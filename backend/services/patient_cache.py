#!/usr/bin/env python3
"""
Patient Cache Service for Ultra-Fast Data Retrieval
Provides instant patient data loading through intelligent caching
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)

@dataclass
class CachedPatient:
    """Cached patient data structure"""
    patient_id: str
    patient_info: Dict[str, Any]
    basic_data: Dict[str, Any]
    diagnosis_summary: List[Dict[str, Any]]
    concern_data: Dict[str, Any]
    cached_at: float
    ttl_seconds: int = 300  # 5 minutes default TTL

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() > (self.cached_at + self.ttl_seconds)

class PatientCacheService:
    """Ultra-fast patient data caching service"""
    
    def __init__(self, max_cache_size: int = 1000, default_ttl: int = 300):
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        
        # Multi-tier cache structure
        self.patient_cache: OrderedDict[str, CachedPatient] = OrderedDict()
        self.basic_info_cache: Dict[str, Dict] = {}  # Ultra-fast basic info
        self.diagnosis_cache: Dict[str, List] = {}   # Diagnosis summaries
        
        self.lock = threading.RLock()
        
        # Performance tracking
        self.stats = {
            'hits': 0,
            'misses': 0,
            'fast_hits': 0,  # Basic info cache hits
            'evictions': 0
        }
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("🚀 PatientCacheService initialized for ultra-fast data retrieval")

    def get_patient_fast(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get basic patient info instantly (microsecond response)"""
        with self.lock:
            # Check ultra-fast basic info cache first
            if patient_id in self.basic_info_cache:
                self.stats['fast_hits'] += 1
                return self.basic_info_cache[patient_id]
            
            # Check main cache
            if patient_id in self.patient_cache:
                cached = self.patient_cache[patient_id]
                if not cached.is_expired():
                    # Move to end (LRU)
                    self.patient_cache.move_to_end(patient_id)
                    self.stats['hits'] += 1
                    
                    # Store in ultra-fast cache for next time
                    self.basic_info_cache[patient_id] = cached.basic_data
                    return cached.basic_data
                else:
                    # Remove expired entry
                    del self.patient_cache[patient_id]
            
            self.stats['misses'] += 1
            return None

    def get_patient_full(self, patient_id: str) -> Optional[CachedPatient]:
        """Get full patient data from cache"""
        with self.lock:
            if patient_id in self.patient_cache:
                cached = self.patient_cache[patient_id]
                if not cached.is_expired():
                    # Move to end (LRU)
                    self.patient_cache.move_to_end(patient_id)
                    self.stats['hits'] += 1
                    return cached
                else:
                    # Remove expired entry
                    del self.patient_cache[patient_id]
                    if patient_id in self.basic_info_cache:
                        del self.basic_info_cache[patient_id]
            
            self.stats['misses'] += 1
            return None

    def cache_patient(self, patient_id: str, patient_info: Dict, diagnosis_history: List = None, 
                     concern_data: Dict = None, ttl_seconds: int = None) -> None:
        """Cache patient data with intelligent optimization"""
        with self.lock:
            if ttl_seconds is None:
                ttl_seconds = self.default_ttl
            
            # Create optimized basic data for ultra-fast access
            basic_data = {
                'patient_id': patient_id,
                'patient_name': patient_info.get('patient_name', f'Patient {patient_id}'),
                'current_status': patient_info.get('current_status', 'active'),
                'admission_date': patient_info.get('admission_date'),
                'date_of_birth': patient_info.get('date_of_birth'),
                'gender': patient_info.get('gender'),
                'fast_mode': True,
                'cached': True
            }
            
            # Create diagnosis summary (only essential fields)
            diagnosis_summary = []
            if diagnosis_history:
                for diag in diagnosis_history[:5]:  # Only keep last 5
                    summary = {
                        'session_id': diag.get('session_id'),
                        'created_at': diag.get('created_at'),
                        'primary_diagnosis': diag.get('primary_diagnosis'),
                        'confidence_score': diag.get('confidence_score'),
                        'status': diag.get('status')
                    }
                    diagnosis_summary.append(summary)
            
            # Optimize concern data
            optimized_concern = {
                'risk_level': concern_data.get('risk_level', 'low') if concern_data else 'low',
                'concern_score': concern_data.get('concern_score', 0.0) if concern_data else 0.0,
                'cached': True
            }
            
            # Create cached patient
            cached_patient = CachedPatient(
                patient_id=patient_id,
                patient_info=patient_info,
                basic_data=basic_data,
                diagnosis_summary=diagnosis_summary,
                concern_data=optimized_concern,
                cached_at=time.time(),
                ttl_seconds=ttl_seconds
            )
            
            # Add to caches
            self.patient_cache[patient_id] = cached_patient
            self.basic_info_cache[patient_id] = basic_data
            self.diagnosis_cache[patient_id] = diagnosis_summary
            
            # Enforce size limit
            self._enforce_size_limit()
            
            logger.debug(f"✅ Cached patient {patient_id} for ultra-fast access")

    def get_diagnosis_summary(self, patient_id: str) -> Optional[List]:
        """Get cached diagnosis summary"""
        with self.lock:
            return self.diagnosis_cache.get(patient_id)

    def invalidate_patient(self, patient_id: str) -> None:
        """Invalidate cached patient data"""
        with self.lock:
            self.patient_cache.pop(patient_id, None)
            self.basic_info_cache.pop(patient_id, None)
            self.diagnosis_cache.pop(patient_id, None)
            logger.debug(f"🗑️ Invalidated cache for patient {patient_id}")

    def preload_patients(self, patient_ids: List[str], db_instance) -> None:
        """Preload multiple patients into cache (background operation)"""
        def _preload():
            for patient_id in patient_ids:
                try:
                    if patient_id not in self.patient_cache:
                        # Load from database
                        patient = db_instance.get_patient(patient_id)
                        if patient:
                            diagnosis_history = db_instance.get_patient_diagnosis_sessions(patient_id, limit=3)
                            severity_data = db_instance.get_patient_severity(patient_id)
                            concern_data = {'risk_level': 'low', 'concern_score': 0.0}
                            if severity_data:
                                concern_data = {
                                    'risk_level': severity_data.get('risk_level', 'low'),
                                    'concern_score': severity_data.get('risk_score', 0.0)
                                }
                            
                            self.cache_patient(patient_id, patient, diagnosis_history, concern_data)
                except Exception as e:
                    logger.warning(f"Failed to preload patient {patient_id}: {e}")
        
        # Run in background thread
        threading.Thread(target=_preload, daemon=True).start()
        logger.info(f"🔄 Preloading {len(patient_ids)} patients in background")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_ratio = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            fast_hit_ratio = (self.stats['fast_hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'cache_size': len(self.patient_cache),
                'basic_cache_size': len(self.basic_info_cache),
                'diagnosis_cache_size': len(self.diagnosis_cache),
                'hit_ratio': round(hit_ratio, 2),
                'fast_hit_ratio': round(fast_hit_ratio, 2),
                'total_hits': self.stats['hits'],
                'fast_hits': self.stats['fast_hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions']
            }

    def warm_cache(self, db_instance) -> None:
        """Warm up cache with active patients"""
        try:
            # Get list of active patients
            patients = db_instance.get_all_patients() or []
            patient_ids = [p.get('patient_id') for p in patients[:50]]  # Top 50 patients
            
            if patient_ids:
                self.preload_patients(patient_ids, db_instance)
                logger.info(f"🔥 Cache warming started for {len(patient_ids)} patients")
        except Exception as e:
            logger.warning(f"Cache warming failed: {e}")

    def _enforce_size_limit(self) -> None:
        """Enforce cache size limits using LRU eviction"""
        while len(self.patient_cache) > self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.patient_cache))
            del self.patient_cache[oldest_key]
            self.basic_info_cache.pop(oldest_key, None)
            self.diagnosis_cache.pop(oldest_key, None)
            self.stats['evictions'] += 1

    def _cleanup_worker(self) -> None:
        """Background cleanup of expired entries"""
        while True:
            try:
                time.sleep(60)  # Cleanup every minute
                current_time = time.time()
                expired_keys = []
                
                with self.lock:
                    for key, cached in self.patient_cache.items():
                        if cached.is_expired():
                            expired_keys.append(key)
                    
                    # Remove expired entries
                    for key in expired_keys:
                        del self.patient_cache[key]
                        self.basic_info_cache.pop(key, None)
                        self.diagnosis_cache.pop(key, None)
                
                if expired_keys:
                    logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
                    
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

# Global cache instance
_patient_cache = None

def get_patient_cache() -> PatientCacheService:
    """Get global patient cache instance"""
    global _patient_cache
    if _patient_cache is None:
        _patient_cache = PatientCacheService()
    return _patient_cache

def initialize_patient_cache(db_instance) -> None:
    """Initialize and warm up patient cache"""
    cache = get_patient_cache()
    cache.warm_cache(db_instance)
    logger.info("🚀 Patient cache initialized and warmed up")