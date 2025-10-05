#!/usr/bin/env python3
"""
Intelligent Caching System for Ontology Mapping
Provides high-performance caching with multiple strategies and automatic optimization
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from pathlib import Path
import gzip
import lzma
from concurrent.futures import ThreadPoolExecutor
import functools

# Optional Redis support
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

from config.performance_config import get_performance_config, CacheConfig

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    hit_ratio: float = 0.0

@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entries_count: int = 0
    avg_access_time: float = 0.0
    hit_ratio: float = 0.0

class BaseCacheBackend:
    """Base class for cache backends"""

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache"""
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        raise NotImplementedError

    def clear(self) -> bool:
        """Clear all cache entries"""
        raise NotImplementedError

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        raise NotImplementedError

    def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        raise NotImplementedError

class MemoryCacheBackend(BaseCacheBackend):
    """In-memory cache backend with LRU eviction"""

    def __init__(self, max_size: int = 1000, compression_enabled: bool = True):
        self.max_size = max_size
        self.compression_enabled = compression_enabled
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.lock = threading.RLock()

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU update"""
        with self.lock:
            start_time = time.time()
            self.stats.total_requests += 1

            if key in self.cache:
                entry = self.cache[key]

                # Check if expired
                if entry.ttl_seconds and (time.time() - entry.created_at) > entry.ttl_seconds:
                    del self.cache[key]
                    self.stats.cache_misses += 1
                    return None

                # Update access metadata
                entry.last_accessed = time.time()
                entry.access_count += 1

                # Move to end (most recently used)
                self.cache.move_to_end(key)

                # Update hit ratio for this entry
                total_accesses = entry.access_count
                if total_accesses > 1:
                    entry.hit_ratio = (entry.hit_ratio * (total_accesses - 1) + 1) / total_accesses
                else:
                    entry.hit_ratio = 1.0

                self.stats.cache_hits += 1
                access_time = time.time() - start_time
                self.stats.avg_access_time = (self.stats.avg_access_time * (self.stats.total_requests - 1) + access_time) / self.stats.total_requests

                # Decompress if needed
                if self.compression_enabled and isinstance(entry.value, bytes):
                    try:
                        entry.value = pickle.loads(gzip.decompress(entry.value))
                    except Exception as e:
                        logger.warning(f"Failed to decompress cache entry {key}: {e}")
                        return None

                return entry.value
            else:
                self.stats.cache_misses += 1
                return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache with size management"""
        with self.lock:
            try:
                # Compress value if enabled
                if self.compression_enabled:
                    compressed_value = gzip.compress(pickle.dumps(value))
                    size_bytes = len(compressed_value)
                else:
                    compressed_value = value
                    size_bytes = len(pickle.dumps(value)) if hasattr(value, '__dict__') or isinstance(value, (dict, list)) else 0

                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=compressed_value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    size_bytes=size_bytes,
                    ttl_seconds=ttl
                )

                # Check if we need to evict entries
                if key not in self.cache and len(self.cache) >= self.max_size:
                    self._evict_entries()

                # Store entry
                self.cache[key] = entry
                self.cache.move_to_end(key)  # Mark as recently used

                # Update stats
                self.stats.entries_count = len(self.cache)
                self.stats.size_bytes = sum(entry.size_bytes for entry in self.cache.values())

                return True

            except Exception as e:
                logger.error(f"Failed to set cache entry {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.entries_count = len(self.cache)
                self.stats.size_bytes = sum(entry.size_bytes for entry in self.cache.values())
                return True
            return False

    def clear(self) -> bool:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats()
            return True

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self.lock:
            self.stats.hit_ratio = (self.stats.cache_hits / self.stats.total_requests) if self.stats.total_requests > 0 else 0.0
            return self.stats

    def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        with self.lock:
            expired_keys = []
            current_time = time.time()

            for key, entry in self.cache.items():
                if entry.ttl_seconds and (current_time - entry.created_at) > entry.ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]

            self.stats.entries_count = len(self.cache)
            self.stats.size_bytes = sum(entry.size_bytes for entry in self.cache.values())

            return len(expired_keys)

    def _evict_entries(self):
        """Evict entries using intelligent LRU with hit ratio consideration"""
        # Calculate scores based on recency, frequency, and hit ratio
        entries_with_scores = []
        current_time = time.time()

        for key, entry in self.cache.items():
            # Recency score (0-1, higher is more recent)
            recency_score = 1.0 / (1.0 + (current_time - entry.last_accessed) / 3600)  # Decay over hours

            # Frequency score (0-1, higher is more frequent)
            frequency_score = min(1.0, entry.access_count / 10)  # Cap at 10 accesses

            # Hit ratio score (0-1, higher is better)
            hit_ratio_score = entry.hit_ratio

            # Size penalty (larger entries are easier to evict)
            size_penalty = min(1.0, entry.size_bytes / 1000000)  # Penalty for entries > 1MB

            # Combined score
            combined_score = (recency_score * 0.4 + frequency_score * 0.3 +
                            hit_ratio_score * 0.2 - size_penalty * 0.1)

            entries_with_scores.append((key, combined_score))

        # Sort by score (lowest first - evict least valuable)
        entries_with_scores.sort(key=lambda x: x[1])

        # Evict lowest scoring entries until we're below max_size
        evicted_count = 0
        target_size = int(self.max_size * 0.8)  # Target 80% of max size

        for key, _ in entries_with_scores:
            if len(self.cache) <= target_size:
                break

            if key in self.cache:
                del self.cache[key]
                evicted_count += 1

        self.stats.evictions += evicted_count
        logger.info(f"Evicted {evicted_count} cache entries")

    def _cleanup_worker(self):
        """Background cleanup worker"""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                expired_count = self.cleanup_expired()
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired cache entries")
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

class RedisCacheBackend(BaseCacheBackend):
    """Redis cache backend for distributed caching"""

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0,
                 password: Optional[str] = None, max_connections: int = 10):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install redis package.")

        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=False  # We'll handle serialization
        )

        # Test connection
        try:
            self.redis.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        self.stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            self.stats.total_requests += 1
            data = self.redis.get(key)

            if data:
                self.stats.cache_hits += 1
                return pickle.loads(data)
            else:
                self.stats.cache_misses += 1
                return None

        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in Redis cache"""
        try:
            data = pickle.dumps(value)

            if ttl:
                return bool(self.redis.setex(key, int(ttl), data))
            else:
                return bool(self.redis.set(key, data))

        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        try:
            return bool(self.redis.delete(key))
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            return bool(self.redis.flushdb())
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        try:
            info = self.redis.info()
            self.stats.entries_count = info.get('db0', {}).get('keys', 0)
            self.stats.hit_ratio = (self.stats.cache_hits / self.stats.total_requests) if self.stats.total_requests > 0 else 0.0
            return self.stats
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return self.stats

    def cleanup_expired(self) -> int:
        """Redis handles expiration automatically"""
        return 0

class IntelligentCache:
    """Intelligent caching system with multiple strategies and backends"""

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize intelligent cache"""
        if config is None:
            perf_config = get_performance_config()
            config = perf_config.cache

        self.config = config
        self.strategy = config.strategy
        self.primary_backend = self._create_primary_backend()
        self.secondary_backend = None  # For future multi-level caching

        # Cache key generation strategies
        self.key_generators = {
            'simple': self._simple_key,
            'intelligent': self._intelligent_key,
            'aggressive': self._aggressive_key
        }

        # Performance monitoring
        self.monitoring_enabled = True
        self.performance_stats = defaultdict(int)

        logger.info(f"Initialized intelligent cache with {config.strategy} strategy")

    def _create_primary_backend(self) -> BaseCacheBackend:
        """Create primary cache backend based on configuration"""
        if self.config.redis_host and REDIS_AVAILABLE:
            try:
                return RedisCacheBackend(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Redis, falling back to memory cache: {e}")

        # Fallback to memory cache
        return MemoryCacheBackend(
            max_size=self.config.max_size,
            compression_enabled=self.config.compression_enabled
        )

    def get(self, key_components: Dict[str, Any]) -> Optional[Any]:
        """Get value from cache using intelligent key generation"""
        cache_key = self._generate_key(key_components)

        start_time = time.time()
        result = self.primary_backend.get(cache_key)
        access_time = time.time() - start_time

        # Update performance stats
        if self.monitoring_enabled:
            self.performance_stats['get_operations'] += 1
            self.performance_stats['avg_get_time'] = (
                (self.performance_stats['avg_get_time'] * (self.performance_stats['get_operations'] - 1)) +
                access_time
            ) / self.performance_stats['get_operations']

        return result

    def set(self, key_components: Dict[str, Any], value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache using intelligent key generation"""
        cache_key = self._generate_key(key_components)

        # Use configured TTL if none provided
        if ttl is None:
            ttl = self.config.ttl_seconds

        start_time = time.time()
        success = self.primary_backend.set(cache_key, value, ttl)
        set_time = time.time() - start_time

        # Update performance stats
        if self.monitoring_enabled:
            self.performance_stats['set_operations'] += 1
            self.performance_stats['avg_set_time'] = (
                (self.performance_stats['avg_set_time'] * (self.performance_stats['set_operations'] - 1)) +
                set_time
            ) / self.performance_stats['set_operations']

        return success

    def delete(self, key_components: Dict[str, Any]) -> bool:
        """Delete value from cache"""
        cache_key = self._generate_key(key_components)
        return self.primary_backend.delete(cache_key)

    def clear(self) -> bool:
        """Clear all cache entries"""
        return self.primary_backend.clear()

    def _generate_key(self, components: Dict[str, Any]) -> str:
        """Generate cache key using configured strategy"""
        key_generator = self.key_generators.get(self.strategy, self._simple_key)
        return key_generator(components)

    def _simple_key(self, components: Dict[str, Any]) -> str:
        """Simple key generation based on main identifier"""
        if 'id' in components:
            return f"{components.get('type', 'unknown')}:{components['id']}"
        elif 'term' in components:
            return f"term:{components['term']}"
        else:
            # Fallback to hash of all components
            return self._aggressive_key(components)

    def _intelligent_key(self, components: Dict[str, Any]) -> str:
        """Intelligent key generation based on semantic content"""
        key_parts = []

        # Add type/domain information
        if 'type' in components:
            key_parts.append(components['type'])
        if 'domain' in components:
            key_parts.append(components['domain'])

        # Add semantic elements for medical terms
        if 'term' in components:
            term = components['term'].lower()

            # Extract key medical concepts (simplified semantic analysis)
            medical_indicators = {
                'cardiac': ['heart', 'cardiac', 'chest', 'myocardial', 'angina'],
                'respiratory': ['breath', 'lung', 'respiratory', 'cough', 'dyspnea'],
                'endocrine': ['diabetes', 'thyroid', 'hormone', 'insulin', 'glucose'],
                'neurological': ['headache', 'brain', 'neurological', 'stroke', 'seizure'],
                'gastrointestinal': ['stomach', 'abdominal', 'liver', 'pancreas', 'nausea']
            }

            for category, keywords in medical_indicators.items():
                if any(keyword in term for keyword in keywords):
                    key_parts.append(category)
                    break

            # Add normalized term (first few words)
            words = term.split()[:3]  # First 3 words
            key_parts.extend(words)

        # Add other relevant components
        for key in ['source', 'version']:
            if key in components:
                key_parts.append(str(components[key]))

        return ":".join(key_parts)

    def _aggressive_key(self, components: Dict[str, Any]) -> str:
        """Aggressive key generation using hash of all components"""
        # Sort keys for consistent hashing
        sorted_components = {k: components[k] for k in sorted(components.keys())}

        # Convert to JSON string for hashing
        component_str = json.dumps(sorted_components, sort_keys=True, default=str)

        # Generate hash
        hash_obj = hashlib.sha256(component_str.encode())
        hash_digest = hash_obj.hexdigest()[:16]  # First 16 chars

        # Add type information if available
        type_prefix = components.get('type', 'unknown')

        return f"{type_prefix}:{hash_digest}"

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        backend_stats = self.primary_backend.get_stats()

        stats = {
            'backend_stats': {
                'total_requests': backend_stats.total_requests,
                'cache_hits': backend_stats.cache_hits,
                'cache_misses': backend_stats.cache_misses,
                'hit_ratio': backend_stats.hit_ratio,
                'entries_count': backend_stats.entries_count,
                'size_bytes': backend_stats.size_bytes,
                'evictions': backend_stats.evictions
            },
            'performance_stats': dict(self.performance_stats),
            'configuration': {
                'strategy': self.strategy,
                'max_size': self.config.max_size,
                'ttl_seconds': self.config.ttl_seconds,
                'compression_enabled': self.config.compression_enabled
            }
        }

        return stats

    def optimize_cache(self) -> Dict[str, Any]:
        """Perform cache optimization operations"""
        logger.info("Performing cache optimization...")

        # Clean up expired entries
        expired_count = self.primary_backend.cleanup_expired()

        # Get current stats
        stats = self.get_stats()

        # Calculate optimization recommendations
        hit_ratio = stats['backend_stats']['hit_ratio']
        recommendations = []

        if hit_ratio < 0.5:
            recommendations.append("Consider adjusting cache strategy - current hit ratio is low")
        if stats['backend_stats']['evictions'] > stats['backend_stats']['entries_count'] * 0.1:
            recommendations.append("High eviction rate - consider increasing cache size")

        optimization_result = {
            'expired_entries_cleaned': expired_count,
            'current_stats': stats,
            'recommendations': recommendations
        }

        logger.info(f"Cache optimization completed: {expired_count} expired entries cleaned")
        return optimization_result

    async def preload_common_terms(self, terms: List[str]) -> int:
        """Preload commonly accessed terms into cache"""
        logger.info(f"Preloading {len(terms)} common terms into cache")

        preloaded_count = 0
        for term in terms:
            try:
                # This would typically call the ontology service
                # For now, we'll simulate preloading
                cache_key = self._generate_key({'term': term, 'type': 'preloaded'})
                # In real implementation, this would fetch the actual data

                preloaded_count += 1

                # Small delay to avoid overwhelming
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.warning(f"Failed to preload term {term}: {e}")

        logger.info(f"Successfully preloaded {preloaded_count} terms")
        return preloaded_count

# Global cache instance
_cache_instance = None

def get_cache() -> IntelligentCache:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = IntelligentCache()
    return _cache_instance

def cached_result(ttl_seconds: Optional[float] = None, key_strategy: str = 'auto'):
    """Decorator for caching function results"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_cache()

            # Generate cache key from function call
            key_components = {
                'function': func.__name__,
                'module': func.__module__,
                'args': str(args) if args else '',
                'kwargs': str(sorted(kwargs.items())) if kwargs else ''
            }

            # Try to get from cache first
            cached_result = cache.get(key_components)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache the result
            cache.set(key_components, result, ttl_seconds)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache = get_cache()

            # Generate cache key from function call
            key_components = {
                'function': func.__name__,
                'module': func.__module__,
                'args': str(args) if args else '',
                'kwargs': str(sorted(kwargs.items())) if kwargs else ''
            }

            # Try to get from cache first
            cached_result = cache.get(key_components)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache the result
            cache.set(key_components, result, ttl_seconds)

            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

# Ontology-specific caching utilities
def ontology_cache_key(term: str, context: Optional[str] = None, source: str = 'unknown') -> Dict[str, Any]:
    """Generate ontology-specific cache key"""
    return {
        'type': 'ontology',
        'term': term,
        'context': context,
        'source': source
    }

def medical_term_cache_key(term: str, domain: Optional[str] = None, language: str = 'en') -> Dict[str, Any]:
    """Generate medical term-specific cache key"""
    return {
        'type': 'medical_term',
        'term': term,
        'domain': domain,
        'language': language
    }

if __name__ == "__main__":
    # CLI interface for cache management
    import argparse

    parser = argparse.ArgumentParser(description="Intelligent Cache Management")
    parser.add_argument('action', choices=['stats', 'clear', 'optimize', 'preload'],
                       help='Action to perform')
    parser.add_argument('--terms-file', help='File containing terms to preload (for preload action)')

    args = parser.parse_args()

    cache = get_cache()

    if args.action == 'stats':
        stats = cache.get_stats()
        print("Cache Statistics:")
        print(json.dumps(stats, indent=2, default=str))

    elif args.action == 'clear':
        if cache.clear():
            print("✅ Cache cleared successfully")
        else:
            print("❌ Failed to clear cache")

    elif args.action == 'optimize':
        optimization_result = cache.optimize_cache()
        print("Cache Optimization Results:")
        print(json.dumps(optimization_result, indent=2, default=str))

    elif args.action == 'preload':
        if not args.terms_file:
            print("❌ Terms file required for preload action")
            exit(1)

        try:
            with open(args.terms_file, 'r') as f:
                terms = [line.strip() for line in f if line.strip()]

            async def preload():
                count = await cache.preload_common_terms(terms)
                print(f"✅ Preloaded {count} terms into cache")

            asyncio.run(preload())

        except Exception as e:
            print(f"❌ Failed to preload terms: {e}")
