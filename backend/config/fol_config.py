# Configuration for CortexMD FOL Verification System

import os
from typing import Dict, Any

# Database Configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "username": os.getenv("DB_USERNAME", "cortexmd"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "cortexmd_db")
}

# Redis Configuration (for caching)
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "password": os.getenv("REDIS_PASSWORD", ""),
    "db": int(os.getenv("REDIS_DB", 0))
}

# FOL Verification Configuration
FOL_CONFIG = {
    "cache_ttl": int(os.getenv("FOL_CACHE_TTL", 1800)),  # 30 minutes
    "max_predicates": int(os.getenv("FOL_MAX_PREDICATES", 50)),
    "verification_timeout": int(os.getenv("FOL_TIMEOUT", 30)),  # seconds
    "batch_size": int(os.getenv("FOL_BATCH_SIZE", 5)),
    "max_workers": int(os.getenv("FOL_MAX_WORKERS", 10))
}

# Ontology Configuration
ONTOLOGY_CONFIG = {
    "umls_api_key": os.getenv("UMLS_API_KEY", "4563e39c-b5ba-4994-b288-8c45269f5d88"),
    "umls_base_url": "https://uts-ws.nlm.nih.gov/rest",
    "enable_cache": os.getenv("ONTOLOGY_CACHE", "true").lower() == "true",
    "cache_ttl": int(os.getenv("ONTOLOGY_CACHE_TTL", 3600))  # 1 hour
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.getenv("LOG_FILE", "cortexmd_fol.log")
}

# Medical Validation Thresholds
VALIDATION_THRESHOLDS = {
    "symptom_confidence": float(os.getenv("SYMPTOM_THRESHOLD", 0.4)),
    "condition_confidence": float(os.getenv("CONDITION_THRESHOLD", 0.5)),
    "medication_confidence": float(os.getenv("MEDICATION_THRESHOLD", 0.6)),
    "lab_confidence": float(os.getenv("LAB_THRESHOLD", 0.6)),
    "vital_confidence": float(os.getenv("VITAL_THRESHOLD", 0.6)),
    "overall_confidence": float(os.getenv("OVERALL_THRESHOLD", 0.5))
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration"""
    return {
        "database": DATABASE_CONFIG,
        "redis": REDIS_CONFIG,
        "fol": FOL_CONFIG,
        "ontology": ONTOLOGY_CONFIG,
        "logging": LOGGING_CONFIG,
        "validation": VALIDATION_THRESHOLDS
    }
