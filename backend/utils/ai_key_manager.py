"""
Simple API key manager for Groq and Google (Gemini).

Usage:
 - Put GROQ_API_KEYS or GROQ_API_KEY in your .env (comma or semicolon separated)
 - Put GOOGLE_API_KEYS or GOOGLE_API_KEY in your .env

This module provides round-robin selection of keys and helpers to return
configured clients/models. It's intentionally lightweight and thread-safe.
"""
import os
import threading
import logging
import time
from typing import List, Optional, Dict
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Performance tracking
_key_usage_stats = defaultdict(lambda: {'requests': 0, 'total_time': 0, 'errors': 0})
_stats_lock = threading.Lock()

def _parse_keys(env_name: str) -> List[str]:
    raw = os.getenv(env_name)
    if not raw:
        return []
    # Allow comma or semicolon separated lists
    parts = [p.strip() for p in raw.replace(';', ',').split(',') if p.strip()]
    return parts

# Load keys once
_GROQ_KEYS = _parse_keys('GROQ_API_KEYS') or (_parse_keys('GROQ_API_KEY') if os.getenv('GROQ_API_KEY') else [])
_GOOGLE_KEYS = _parse_keys('GOOGLE_API_KEYS') or (_parse_keys('GOOGLE_API_KEY') if os.getenv('GOOGLE_API_KEY') else [])

# Log available keys (without exposing the actual keys)
logger.info(f"🔑 Loaded {len(_GROQ_KEYS)} Groq API keys")
logger.info(f"🔑 Loaded {len(_GOOGLE_KEYS)} Google API keys")

# Round-robin indices and locks
_groq_index = 0
_google_index = 0
_groq_lock = threading.Lock()
_google_lock = threading.Lock()

def _next_index(keys: List[str], lock: threading.Lock, idx_name: str) -> Optional[int]:
    if not keys:
        return None
    global _groq_index, _google_index
    with lock:
        if idx_name == 'groq':
            i = _groq_index
            _groq_index = (_groq_index + 1) % len(keys)
            return i
        else:
            i = _google_index
            _google_index = (_google_index + 1) % len(keys)
            return i

def get_next_groq_key() -> Optional[str]:
    i = _next_index(_GROQ_KEYS, _groq_lock, 'groq')
    if i is not None:
        logger.debug(f"🔄 Using Groq API key #{i+1}/{len(_GROQ_KEYS)}")
        return _GROQ_KEYS[i]
    logger.warning("⚠️ No Groq API keys available")
    return None

def get_next_google_key() -> Optional[str]:
    i = _next_index(_GOOGLE_KEYS, _google_lock, 'google')
    if i is not None:
        logger.debug(f"🔄 Using Google API key #{i+1}/{len(_GOOGLE_KEYS)}")
        return _GOOGLE_KEYS[i]
    logger.warning("⚠️ No Google API keys available")
    return None

# Helper factories (import heavy libraries lazily to avoid startup failures)
def get_groq_client():
    try:
        from groq import Groq
    except Exception:
        return None

    key = get_next_groq_key()
    if not key:
        return None
    try:
        return Groq(api_key=key)
    except Exception:
        # Last-resort: try constructing with first key (if any)
        try:
            return Groq(api_key=_GROQ_KEYS[0]) if _GROQ_KEYS else None
        except Exception:
            return None

def get_gemini_model(model_name: str = 'gemini-2.5-flash'):
    """Configure google.generativeai with the next key and return a GenerativeModel.

    Note: genai.configure is global; we re-configure it on each call which is
    acceptable for short-lived functions and simple round-robin load balancing.
    """
    try:
        import google.generativeai as genai
    except Exception as e:
        logger.error(f"❌ Failed to import google.generativeai: {e}")
        return None

    key = get_next_google_key()
    if not key:
        logger.error("❌ No Google API keys available for Gemini model")
        return None
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        logger.info(f"✅ Successfully configured Gemini model: {model_name}")
        return model
    except Exception as e:
        logger.warning(f"⚠️ Failed to configure Gemini with current key, trying fallback: {e}")
        # Try first configured key as fallback
        try:
            if _GOOGLE_KEYS:
                genai.configure(api_key=_GOOGLE_KEYS[0])
                model = genai.GenerativeModel(model_name)
                logger.info(f"✅ Successfully configured Gemini model with fallback key: {model_name}")
                return model
        except Exception as fallback_e:
            logger.error(f"❌ Failed to configure Gemini with fallback key: {fallback_e}")
            return None

def get_available_key_counts():
    """Return the number of available keys for each service."""
    return {
        'groq_keys': len(_GROQ_KEYS),
        'google_keys': len(_GOOGLE_KEYS)
    }

def get_api_key_with_fallback(service='google'):
    """Get API key with fallback to environment variables.
    
    Args:
        service: 'google' or 'groq'
    
    Returns:
        str: API key or None if not available
    """
    if service.lower() == 'google':
        key = get_next_google_key()
        if not key:
            # Fallback to single environment variable
            key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if key:
                logger.info("🔄 Using fallback Google API key from environment")
        return key
    elif service.lower() == 'groq':
        key = get_next_groq_key()
        if not key:
            # Fallback to single environment variable
            key = os.getenv("GROQ_API_KEY")
            if key:
                logger.info("🔄 Using fallback Groq API key from environment")
        return key
    else:
        logger.error(f"❌ Unknown service: {service}")
        return None

def ensure_api_key_available(service='google'):
    """Ensure API key is available, raise error if not.
    
    Args:
        service: 'google' or 'groq'
        
    Raises:
        ValueError: If no API key is available
    """
    key = get_api_key_with_fallback(service)
    if not key:
        service_upper = service.upper()
        raise ValueError(f"{service_upper}_API_KEY not found. Please configure {service_upper}_API_KEYS or {service_upper}_API_KEY in environment variables")
    return key

def track_api_usage(key_suffix: str, response_time: float, success: bool = True):
    """Track API usage statistics.
    
    Args:
        key_suffix: Last 8 characters of the API key for identification
        response_time: Time taken for the API call in seconds
        success: Whether the API call was successful
    """
    with _stats_lock:
        stats = _key_usage_stats[key_suffix]
        stats['requests'] += 1
        stats['total_time'] += response_time
        if not success:
            stats['errors'] += 1

def get_usage_stats() -> Dict:
    """Get API usage statistics."""
    with _stats_lock:
        stats = {}
        for key_suffix, data in _key_usage_stats.items():
            avg_time = data['total_time'] / data['requests'] if data['requests'] > 0 else 0
            stats[f"key_...{key_suffix}"] = {
                'total_requests': data['requests'],
                'average_response_time': round(avg_time, 3),
                'total_errors': data['errors'],
                'error_rate': round(data['errors'] / data['requests'] * 100, 2) if data['requests'] > 0 else 0
            }
        return stats

def reset_usage_stats():
    """Reset all usage statistics."""
    with _stats_lock:
        _key_usage_stats.clear()
        logger.info("📊 Usage statistics reset")
