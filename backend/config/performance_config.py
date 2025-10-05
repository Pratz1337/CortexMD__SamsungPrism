#!/usr/bin/env python3
"""
Performance Configuration Module  
Centralized configuration management for performance optimization
UPDATED: Added optimized FOL verification and explanation generation settings
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

# FOL Verification Performance Settings  
OPTIMIZED_FOL_CONFIG = {
    "enable_caching": True,
    "cache_size": 500,
    "max_threads": 4,
    "batch_processing": True,
    "fast_text_matching": True,
    "skip_complex_nlp": True,
    "verification_timeout": 10.0,  # seconds
    "confidence_threshold": 0.5,
    "enable_parallel_processing": True
}

# Explanation Generation Performance Settings
OPTIMIZED_EXPLANATION_CONFIG = {
    "max_explanations": 5,
    "single_pass_generation": True,
    "skip_verification_loops": True,
    "enable_batch_verification": True,
    "fast_confidence_calculation": True,
    "simple_verification_only": True,
    "max_generation_time": 15.0  # seconds
}

# System Performance Settings
SYSTEM_PERFORMANCE_CONFIG = {
    "disable_slow_verification": True,
    "enable_fast_mode": True,
    "skip_redundant_processing": True,
    "use_optimized_services": True,
    "parallel_processing": True,
    "cache_everything": True
}

# Feature Flags for Performance
PERFORMANCE_FEATURES = {
    "use_optimized_fol_service": True,
    "use_optimized_explanation_generator": True,
    "skip_advanced_fol_verification": True,
    "enable_fast_predicate_extraction": True,
    "use_compiled_regex": True,
    "enable_lru_caching": True,
    "batch_process_predicates": True
}

@dataclass
class CacheConfig:
    """Caching configuration"""
    enabled: bool = True
    strategy: str = "intelligent"  # 'simple', 'intelligent', 'aggressive'
    max_size: int = 1000
    ttl_seconds: int = 3600  # 1 hour default
    compression_enabled: bool = True
    persistence_enabled: bool = False
    redis_host: Optional[str] = None
    redis_port: int = 6379
    redis_db: int = 0

@dataclass
class PerformanceThresholds:
    """Performance threshold configurations"""
    max_response_time: float = 2.0  # seconds
    min_success_rate: float = 0.95  # 95%
    max_error_rate: float = 0.05  # 5%
    target_p95_response_time: float = 1.0  # seconds
    concurrent_users_warning: int = 50
    concurrent_users_critical: int = 100

@dataclass
class EndpointConfig:
    """Configuration for specific API endpoints"""
    name: str
    path: str
    priority: int = 1  # 1=low, 2=medium, 3=high
    cacheable: bool = True
    timeout_seconds: float = 30.0
    max_retries: int = 3
    circuit_breaker_enabled: bool = False
    rate_limit_per_minute: int = 100

@dataclass
class PerformanceConfig:
    """Main performance configuration"""
    cache: CacheConfig = field(default_factory=CacheConfig)
    thresholds: PerformanceThresholds = field(default_factory=PerformanceThresholds)
    endpoints: List[EndpointConfig] = field(default_factory=list)
    monitoring_enabled: bool = True
    auto_scaling_enabled: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        """Initialize default endpoints if not provided"""
        if not self.endpoints:
            self.endpoints = self._get_default_endpoints()

    def _get_default_endpoints(self) -> List[EndpointConfig]:
        """Get default endpoint configurations"""
        return [
            EndpointConfig(
                name="Predicate Extraction",
                path="/api/predicates/extract",
                priority=3,
                cacheable=True,
                timeout_seconds=45.0,
                rate_limit_per_minute=50
            ),
            EndpointConfig(
                name="Predicate Validation",
                path="/api/predicates/validate",
                priority=2,
                cacheable=True,
                timeout_seconds=30.0,
                rate_limit_per_minute=100
            ),
            EndpointConfig(
                name="Health Check",
                path="/api/health",
                priority=1,
                cacheable=True,
                timeout_seconds=5.0,
                rate_limit_per_minute=1000
            ),
            EndpointConfig(
                name="Full Diagnosis",
                path="/diagnose",
                priority=3,
                cacheable=False,
                timeout_seconds=120.0,
                rate_limit_per_minute=20,
                circuit_breaker_enabled=True
            ),
            EndpointConfig(
                name="Chat Interaction",
                path="/chat",
                priority=2,
                cacheable=False,
                timeout_seconds=60.0,
                rate_limit_per_minute=30
            ),
            EndpointConfig(
                name="Ontology Normalize",
                path="/ontology/normalize",
                priority=2,
                cacheable=True,
                timeout_seconds=15.0,
                rate_limit_per_minute=200
            ),
            EndpointConfig(
                name="Ontology Search",
                path="/ontology/search",
                priority=2,
                cacheable=True,
                timeout_seconds=20.0,
                rate_limit_per_minute=100
            )
        ]

class ConfigManager:
    """Configuration manager for performance settings"""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager"""
        self.config_file = config_file or self._find_config_file()
        self.config = self._load_config()

    def _find_config_file(self) -> str:
        """Find configuration file in standard locations"""
        search_paths = [
            Path.cwd() / "performance_config.json",
            Path.cwd() / "config" / "performance_config.json",
            Path.home() / ".cortexmd" / "performance_config.json",
            Path(__file__).parent / "performance_config.json"
        ]

        for path in search_paths:
            if path.exists():
                return str(path)

        # Return default path if no config found
        return str(Path.cwd() / "performance_config.json")

    def _load_config(self) -> PerformanceConfig:
        """Load configuration from file or environment"""
        config_data = {}

        # Try to load from file
        if Path(self.config_file).exists():
            try:
                import json
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")

        # Override with environment variables
        config_data = self._apply_environment_overrides(config_data)

        # Convert to PerformanceConfig object
        return self._dict_to_config(config_data)

    def _apply_environment_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""

        # Cache configuration
        if 'cache' not in config_data:
            config_data['cache'] = {}

        cache_config = config_data['cache']
        cache_config['enabled'] = os.getenv('PERF_CACHE_ENABLED', str(cache_config.get('enabled', True))).lower() == 'true'
        cache_config['strategy'] = os.getenv('PERF_CACHE_STRATEGY', cache_config.get('strategy', 'intelligent'))
        cache_config['max_size'] = int(os.getenv('PERF_CACHE_MAX_SIZE', cache_config.get('max_size', 1000)))
        cache_config['ttl_seconds'] = int(os.getenv('PERF_CACHE_TTL', cache_config.get('ttl_seconds', 3600)))

        # Redis configuration
        cache_config['redis_host'] = os.getenv('REDIS_HOST', cache_config.get('redis_host'))
        cache_config['redis_port'] = int(os.getenv('REDIS_PORT', cache_config.get('redis_port', 6379)))
        cache_config['redis_db'] = int(os.getenv('REDIS_DB', cache_config.get('redis_db', 0)))

        # Performance thresholds
        if 'thresholds' not in config_data:
            config_data['thresholds'] = {}

        thresholds = config_data['thresholds']
        thresholds['max_response_time'] = float(os.getenv('PERF_MAX_RESPONSE_TIME', thresholds.get('max_response_time', 2.0)))
        thresholds['min_success_rate'] = float(os.getenv('PERF_MIN_SUCCESS_RATE', thresholds.get('min_success_rate', 0.95)))
        thresholds['target_p95_response_time'] = float(os.getenv('PERF_TARGET_P95', thresholds.get('target_p95_response_time', 1.0)))

        # General settings
        config_data['monitoring_enabled'] = os.getenv('PERF_MONITORING_ENABLED', str(config_data.get('monitoring_enabled', True))).lower() == 'true'
        config_data['auto_scaling_enabled'] = os.getenv('PERF_AUTO_SCALING_ENABLED', str(config_data.get('auto_scaling_enabled', False))).lower() == 'true'
        config_data['log_level'] = os.getenv('PERF_LOG_LEVEL', config_data.get('log_level', 'INFO'))

        return config_data

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> PerformanceConfig:
        """Convert dictionary to PerformanceConfig object"""
        try:
            # Extract nested configurations
            cache_data = config_dict.get('cache', {})
            thresholds_data = config_dict.get('thresholds', {})
            endpoints_data = config_dict.get('endpoints', [])

            # Convert endpoint data to EndpointConfig objects
            endpoints = []
            for ep_data in endpoints_data:
                endpoints.append(EndpointConfig(**ep_data))

            # Create nested config objects
            cache_config = CacheConfig(**cache_data)
            thresholds_config = PerformanceThresholds(**thresholds_data)

            # Create main config
            return PerformanceConfig(
                cache=cache_config,
                thresholds=thresholds_config,
                endpoints=endpoints,
                monitoring_enabled=config_dict.get('monitoring_enabled', True),
                auto_scaling_enabled=config_dict.get('auto_scaling_enabled', False),
                log_level=config_dict.get('log_level', 'INFO')
            )

        except Exception as e:
            print(f"Warning: Error parsing configuration: {e}. Using defaults.")
            return PerformanceConfig()

    def save_config(self, config_file: Optional[str] = None):
        """Save current configuration to file"""
        save_path = config_file or self.config_file

        try:
            import json
            config_dict = self._config_to_dict(self.config)

            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)

            print(f"Configuration saved to {save_path}")

        except Exception as e:
            print(f"Error saving configuration: {e}")

    def _config_to_dict(self, config: PerformanceConfig) -> Dict[str, Any]:
        """Convert PerformanceConfig to dictionary"""
        return {
            'cache': {
                'enabled': config.cache.enabled,
                'strategy': config.cache.strategy,
                'max_size': config.cache.max_size,
                'ttl_seconds': config.cache.ttl_seconds,
                'compression_enabled': config.cache.compression_enabled,
                'persistence_enabled': config.cache.persistence_enabled,
                'redis_host': config.cache.redis_host,
                'redis_port': config.cache.redis_port,
                'redis_db': config.cache.redis_db
            },
            'thresholds': {
                'max_response_time': config.thresholds.max_response_time,
                'min_success_rate': config.thresholds.min_success_rate,
                'max_error_rate': config.thresholds.max_error_rate,
                'target_p95_response_time': config.thresholds.target_p95_response_time,
                'concurrent_users_warning': config.thresholds.concurrent_users_warning,
                'concurrent_users_critical': config.thresholds.concurrent_users_critical
            },
            'endpoints': [
                {
                    'name': ep.name,
                    'path': ep.path,
                    'priority': ep.priority,
                    'cacheable': ep.cacheable,
                    'timeout_seconds': ep.timeout_seconds,
                    'max_retries': ep.max_retries,
                    'circuit_breaker_enabled': ep.circuit_breaker_enabled,
                    'rate_limit_per_minute': ep.rate_limit_per_minute
                }
                for ep in config.endpoints
            ],
            'monitoring_enabled': config.monitoring_enabled,
            'auto_scaling_enabled': config.auto_scaling_enabled,
            'log_level': config.log_level
        }

    def get_endpoint_config(self, endpoint_path: str) -> Optional[EndpointConfig]:
        """Get configuration for specific endpoint"""
        for endpoint in self.config.endpoints:
            if endpoint.path == endpoint_path:
                return endpoint
        return None

    def update_endpoint_config(self, endpoint_path: str, **updates):
        """Update configuration for specific endpoint"""
        for endpoint in self.config.endpoints:
            if endpoint.path == endpoint_path:
                for key, value in updates.items():
                    if hasattr(endpoint, key):
                        setattr(endpoint, key, value)
                break

    def add_endpoint_config(self, endpoint_config: EndpointConfig):
        """Add new endpoint configuration"""
        # Check if endpoint already exists
        existing = self.get_endpoint_config(endpoint_config.path)
        if existing:
            # Update existing
            idx = self.config.endpoints.index(existing)
            self.config.endpoints[idx] = endpoint_config
        else:
            # Add new
            self.config.endpoints.append(endpoint_config)

    def validate_config(self) -> List[str]:
        """Validate current configuration and return list of issues"""
        issues = []

        # Validate cache configuration
        if self.config.cache.max_size <= 0:
            issues.append("Cache max_size must be positive")

        if self.config.cache.ttl_seconds <= 0:
            issues.append("Cache TTL must be positive")

        if self.config.cache.strategy not in ['simple', 'intelligent', 'aggressive']:
            issues.append("Invalid cache strategy. Must be 'simple', 'intelligent', or 'aggressive'")

        # Validate thresholds
        if not 0 < self.config.thresholds.min_success_rate <= 1:
            issues.append("Min success rate must be between 0 and 1")

        if self.config.thresholds.max_response_time <= 0:
            issues.append("Max response time must be positive")

        # Validate endpoints
        for endpoint in self.config.endpoints:
            if endpoint.priority not in [1, 2, 3]:
                issues.append(f"Endpoint {endpoint.name}: priority must be 1, 2, or 3")

            if endpoint.timeout_seconds <= 0:
                issues.append(f"Endpoint {endpoint.name}: timeout must be positive")

            if endpoint.rate_limit_per_minute <= 0:
                issues.append(f"Endpoint {endpoint.name}: rate limit must be positive")

        return issues

    def get_performance_alerts(self, current_metrics: Dict[str, Any]) -> List[str]:
        """Check current metrics against thresholds and return alerts"""
        alerts = []

        # Check response time
        avg_response_time = current_metrics.get('avg_response_time', 0)
        if avg_response_time > self.config.thresholds.max_response_time:
            alerts.append(".2f")

        # Check success rate
        success_rate = current_metrics.get('success_rate', 1.0)
        if success_rate < self.config.thresholds.min_success_rate:
            alerts.append(".1%")

        # Check P95 response time
        p95_response_time = current_metrics.get('p95_response_time', 0)
        if p95_response_time > self.config.thresholds.target_p95_response_time:
            alerts.append(".2f")

        # Check concurrent users
        concurrent_users = current_metrics.get('concurrent_users', 0)
        if concurrent_users >= self.config.thresholds.concurrent_users_critical:
            alerts.append(f"🚨 CRITICAL: {concurrent_users} concurrent users (exceeds critical threshold)")
        elif concurrent_users >= self.config.thresholds.concurrent_users_warning:
            alerts.append(f"⚠️  WARNING: {concurrent_users} concurrent users (approaching critical threshold)")

        return alerts

# Global configuration instance
_config_manager = None

def get_performance_config() -> PerformanceConfig:
    """Get global performance configuration"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config

def get_config_manager() -> ConfigManager:
    """Get global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

# Utility functions for common configurations
def create_production_config() -> PerformanceConfig:
    """Create production-optimized configuration"""
    return PerformanceConfig(
        cache=CacheConfig(
            enabled=True,
            strategy="intelligent",
            max_size=5000,
            ttl_seconds=7200,  # 2 hours
            compression_enabled=True,
            persistence_enabled=True
        ),
        thresholds=PerformanceThresholds(
            max_response_time=1.0,
            min_success_rate=0.99,
            target_p95_response_time=0.5,
            concurrent_users_warning=100,
            concurrent_users_critical=200
        ),
        monitoring_enabled=True,
        auto_scaling_enabled=True,
        log_level="WARNING"
    )

def create_development_config() -> PerformanceConfig:
    """Create development-friendly configuration"""
    return PerformanceConfig(
        cache=CacheConfig(
            enabled=True,
            strategy="simple",
            max_size=500,
            ttl_seconds=1800,  # 30 minutes
            compression_enabled=False,
            persistence_enabled=False
        ),
        thresholds=PerformanceThresholds(
            max_response_time=5.0,
            min_success_rate=0.90,
            target_p95_response_time=2.0,
            concurrent_users_warning=20,
            concurrent_users_critical=50
        ),
        monitoring_enabled=True,
        auto_scaling_enabled=False,
        log_level="DEBUG"
    )

def create_high_performance_config() -> PerformanceConfig:
    """Create high-performance configuration"""
    return PerformanceConfig(
        cache=CacheConfig(
            enabled=True,
            strategy="aggressive",
            max_size=10000,
            ttl_seconds=10800,  # 3 hours
            compression_enabled=True,
            persistence_enabled=True
        ),
        thresholds=PerformanceThresholds(
            max_response_time=0.5,
            min_success_rate=0.995,
            target_p95_response_time=0.2,
            concurrent_users_warning=200,
            concurrent_users_critical=500
        ),
        monitoring_enabled=True,
        auto_scaling_enabled=True,
        log_level="ERROR"
    )

if __name__ == "__main__":
    # CLI interface for configuration management
    import argparse

    parser = argparse.ArgumentParser(description="Performance Configuration Manager")
    parser.add_argument('action', choices=['show', 'validate', 'save', 'create-prod', 'create-dev', 'create-hp'],
                       help='Action to perform')
    parser.add_argument('--config-file', help='Configuration file path')
    parser.add_argument('--endpoint', help='Endpoint path for endpoint-specific operations')

    args = parser.parse_args()

    manager = ConfigManager(args.config_file)

    if args.action == 'show':
        print("Current Configuration:")
        print(f"Cache Enabled: {manager.config.cache.enabled}")
        print(f"Cache Strategy: {manager.config.cache.strategy}")
        print(f"Max Cache Size: {manager.config.cache.max_size}")
        print(f"Cache TTL: {manager.config.cache.ttl_seconds}s")
        print(f"Monitoring Enabled: {manager.config.monitoring_enabled}")
        print(f"Auto-scaling Enabled: {manager.config.auto_scaling_enabled}")
        print(f"Log Level: {manager.config.log_level}")
        print(f"\nEndpoints ({len(manager.config.endpoints)}):")
        for ep in manager.config.endpoints:
            print(f"  {ep.name}: {ep.path} (Priority: {ep.priority}, Cacheable: {ep.cacheable})")

    elif args.action == 'validate':
        issues = manager.validate_config()
        if issues:
            print("Configuration Issues Found:")
            for issue in issues:
                print(f"  ❌ {issue}")
        else:
            print("✅ Configuration is valid")

    elif args.action == 'save':
        manager.save_config(args.config_file)
        print("Configuration saved")

    elif args.action == 'create-prod':
        manager.config = create_production_config()
        print("Production configuration created")

    elif args.action == 'create-dev':
        manager.config = create_development_config()
        print("Development configuration created")

    elif args.action == 'create-hp':
        manager.config = create_high_performance_config()
        print("High-performance configuration created")

    if args.endpoint:
        ep_config = manager.get_endpoint_config(args.endpoint)
        if ep_config:
            print(f"\nEndpoint Configuration for {args.endpoint}:")
            print(f"  Name: {ep_config.name}")
            print(f"  Priority: {ep_config.priority}")
            print(f"  Cacheable: {ep_config.cacheable}")
            print(f"  Timeout: {ep_config.timeout_seconds}s")
            print(f"  Max Retries: {ep_config.max_retries}")
            print(f"  Circuit Breaker: {ep_config.circuit_breaker_enabled}")
            print(f"  Rate Limit: {ep_config.rate_limit_per_minute}/min")
        else:
            print(f"Endpoint {args.endpoint} not found")
