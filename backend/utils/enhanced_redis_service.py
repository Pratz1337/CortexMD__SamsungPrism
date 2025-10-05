"""
Enhanced Redis Service for CortexMD
Dual storage with PostgreSQL + Redis for optimal performance
"""

import json
import redis
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib

logger = logging.getLogger(__name__)

class EnhancedRedisService:
    """Enhanced Redis service for diagnosis caching and AI context"""
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis_available = False
        self.memory_fallback = {}
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info("✅ Enhanced Redis service connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable, using memory fallback: {e}")
            self.redis_available = False
    
    def _get_key(self, prefix: str, identifier: str) -> str:
        """Generate Redis key with prefix"""
        return f"cortexmd:{prefix}:{identifier}"
    
    def _set_with_fallback(self, key: str, value: Any, expiry: int = 3600) -> bool:
        """Set value with Redis or memory fallback"""
        try:
            if self.redis_available:
                serialized = json.dumps(value, default=str)
                return self.redis_client.setex(key, expiry, serialized)
            else:
                # Memory fallback with expiry
                self.memory_fallback[key] = {
                    'value': value,
                    'expires_at': datetime.now() + timedelta(seconds=expiry)
                }
                return True
        except Exception as e:
            logger.error(f"Redis set error for {key}: {e}")
            return False
    
    def _get_with_fallback(self, key: str) -> Any:
        """Get value with Redis or memory fallback"""
        try:
            if self.redis_available:
                # Add timeout protection for Redis operations
                try:
                    data = self.redis_client.get(key)
                    return json.loads(data) if data else None
                except redis.TimeoutError:
                    logger.warning(f"Redis timeout for key: {key}")
                    return None
            else:
                # Memory fallback with expiry check
                cached = self.memory_fallback.get(key)
                if cached:
                    if datetime.now() < cached['expires_at']:
                        return cached['value']
                    else:
                        del self.memory_fallback[key]
                return None
        except Exception as e:
            logger.error(f"Redis get error for {key}: {e}")
            return None

    # ===== Compatibility helpers (to replace simple_redis_service usages) =====
    def set_data(self, key: str, value: Any, expiry: int = 3600) -> bool:
        """Compatibility wrapper: store arbitrary data under a namespaced key.
        Accepts keys like 'namespace:identifier' or plain keys.
        """
        try:
            if ":" in key:
                prefix, identifier = key.split(":", 1)
            else:
                prefix, identifier = "generic", key
            redis_key = self._get_key(prefix, identifier)
            return self._set_with_fallback(redis_key, value, expiry)
        except Exception as e:
            logger.error(f"Failed to set compatibility key {key}: {e}")
            return False

    def get_data(self, key: str) -> Any:
        """Compatibility wrapper: retrieve arbitrary data by namespaced key.
        Accepts keys like 'namespace:identifier' or plain keys.
        """
        try:
            if ":" in key:
                prefix, identifier = key.split(":", 1)
            else:
                prefix, identifier = "generic", key
            redis_key = self._get_key(prefix, identifier)
            return self._get_with_fallback(redis_key)
        except Exception as e:
            logger.error(f"Failed to get compatibility key {key}: {e}")
            return None

    # Backward-compat alias used by some modules
    def store_data(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Alias for set_data; ttl maps to expiry."""
        try:
            expiry = ttl if ttl is not None else 3600
            return self.set_data(key, value, expiry=expiry)
        except Exception as e:
            logger.error(f"Failed to store_data for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a key from Redis or memory fallback"""
        try:
            if self.redis_available:
                return bool(self.redis_client.delete(key))
            else:
                # Memory fallback
                if key in self.memory_fallback:
                    del self.memory_fallback[key]
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False
    
    def keys(self, pattern: str) -> List[str]:
        """Get keys matching a pattern"""
        try:
            if self.redis_available:
                return self.redis_client.keys(pattern)
            else:
                # Memory fallback - simple pattern matching
                import fnmatch
                return [k for k in self.memory_fallback.keys() if fnmatch.fnmatch(k, pattern)]
        except Exception as e:
            logger.error(f"Failed to get keys for pattern {pattern}: {e}")
            return []
    
    # ===== DIAGNOSIS CACHING =====
    
    def cache_diagnosis_session(self, session_id: str, diagnosis_data: Dict[str, Any], 
                              patient_id: str = None) -> bool:
        """Cache complete diagnosis session for fast AI access"""
        try:
            # Cache full diagnosis data
            diagnosis_key = self._get_key("diagnosis", session_id)
            success = self._set_with_fallback(diagnosis_key, diagnosis_data, expiry=86400)  # 24 hours
            
            # Cache patient's recent diagnoses list
            if patient_id:
                self._update_patient_recent_diagnoses(patient_id, session_id, diagnosis_data)
            
            # Cache diagnosis summary for quick access
            summary = self._extract_diagnosis_summary(diagnosis_data)
            summary_key = self._get_key("diagnosis_summary", session_id)
            self._set_with_fallback(summary_key, summary, expiry=86400)
            
            logger.info(f"✅ Cached diagnosis session {session_id} in Redis")
            return success
        except Exception as e:
            logger.error(f"Failed to cache diagnosis session: {e}")
            return False
    
    def get_diagnosis_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached diagnosis session"""
        diagnosis_key = self._get_key("diagnosis", session_id)
        return self._get_with_fallback(diagnosis_key)
    
    def get_diagnosis_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get quick diagnosis summary"""
        summary_key = self._get_key("diagnosis_summary", session_id)
        return self._get_with_fallback(summary_key)
    
    def _extract_diagnosis_summary(self, diagnosis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key diagnosis information for quick access"""
        diagnosis_result = diagnosis_data.get('diagnosis_result', {})
        
        # Handle DiagnosisResult object vs dictionary
        if hasattr(diagnosis_result, 'primary_diagnosis'):  # Pydantic object
            primary_diagnosis = getattr(diagnosis_result, 'primary_diagnosis', 'Unknown')
            confidence_score = getattr(diagnosis_result, 'confidence_score', 0.0)
            icd_codes = getattr(diagnosis_result, 'icd_codes', []) or []
            snomed_codes = getattr(diagnosis_result, 'snomed_codes', []) or []
        elif isinstance(diagnosis_result, dict):  # Dictionary
            primary_diagnosis = diagnosis_result.get('primary_diagnosis', 'Unknown')
            confidence_score = diagnosis_result.get('confidence_score', 0.0)
            icd_codes = diagnosis_result.get('icd_codes', [])
            snomed_codes = diagnosis_result.get('snomed_codes', [])
        else:
            primary_diagnosis = 'Unknown'
            confidence_score = 0.0
            icd_codes = []
            snomed_codes = []
        
        # Handle patient_input object vs dictionary
        patient_input = diagnosis_data.get('patient_input', {})
        if hasattr(patient_input, 'patient_id'):  # Pydantic object
            patient_id = getattr(patient_input, 'patient_id', None)
        elif isinstance(patient_input, dict):  # Dictionary
            patient_id = patient_input.get('patient_id')
        else:
            patient_id = None
        
        return {
            'session_id': diagnosis_data.get('session_id'),
            'patient_id': patient_id,
            'primary_diagnosis': primary_diagnosis,
            'confidence_score': confidence_score,
            'status': diagnosis_data.get('status', 'unknown'),
            'created_at': diagnosis_data.get('created_at', datetime.now().isoformat()),
            'key_symptoms': self._extract_key_symptoms(diagnosis_data),
            'risk_level': self._determine_risk_level(diagnosis_result),
            'icd_codes': icd_codes,
            'snomed_codes': snomed_codes
        }
    
    def _extract_key_symptoms(self, diagnosis_data: Dict[str, Any]) -> List[str]:
        """Extract key symptoms from patient input"""
        patient_input = diagnosis_data.get('patient_input', {})
        
        # Handle patient_input object vs dictionary
        if hasattr(patient_input, 'text_data'):  # Pydantic object
            symptoms_text = getattr(patient_input, 'text_data', '') or getattr(patient_input, 'symptoms', '')
        elif isinstance(patient_input, dict):  # Dictionary
            symptoms_text = patient_input.get('text_data', '') or patient_input.get('symptoms', '')
        else:
            symptoms_text = ''
        
        # Simple keyword extraction (can be enhanced with NLP)
        key_symptoms = []
        symptom_keywords = ['pain', 'fever', 'headache', 'nausea', 'fatigue', 'cough', 'shortness of breath']
        
        for keyword in symptom_keywords:
            if keyword.lower() in symptoms_text.lower():
                key_symptoms.append(keyword)
        
        return key_symptoms[:5]  # Top 5 symptoms
    
    def _determine_risk_level(self, diagnosis_result: Dict[str, Any]) -> str:
        """Determine risk level based on confidence and diagnosis"""
        # Handle DiagnosisResult object vs dictionary
        if hasattr(diagnosis_result, 'confidence_score'):  # Pydantic object
            confidence = getattr(diagnosis_result, 'confidence_score', 0.0)
            diagnosis = getattr(diagnosis_result, 'primary_diagnosis', '').lower()
        elif isinstance(diagnosis_result, dict):  # Dictionary
            confidence = diagnosis_result.get('confidence_score', 0.0)
            diagnosis = diagnosis_result.get('primary_diagnosis', '').lower()
        else:
            confidence = 0.0
            diagnosis = ''
        
        # High-risk conditions
        high_risk_terms = ['emergency', 'critical', 'severe', 'acute', 'urgent']
        if any(term in diagnosis for term in high_risk_terms):
            return 'high'
        
        # Risk based on confidence
        if confidence >= 0.8:
            return 'medium'
        elif confidence >= 0.6:
            return 'low'
        else:
            return 'very_low'
    
    def _update_patient_recent_diagnoses(self, patient_id: str, session_id: str, 
                                       diagnosis_data: Dict[str, Any]) -> bool:
        """Update patient's recent diagnoses list in Redis"""
        try:
            recent_key = self._get_key("patient_recent_diagnoses", patient_id)
            
            # Get existing recent diagnoses
            recent_diagnoses = self._get_with_fallback(recent_key) or []
            
            # Add new diagnosis summary
            summary = self._extract_diagnosis_summary(diagnosis_data)
            recent_diagnoses.insert(0, summary)  # Add to beginning
            
            # Keep only last 10 diagnoses
            recent_diagnoses = recent_diagnoses[:10]
            
            # Cache updated list
            return self._set_with_fallback(recent_key, recent_diagnoses, expiry=604800)  # 7 days
        except Exception as e:
            logger.error(f"Failed to update patient recent diagnoses: {e}")
            return False
    
    def get_patient_recent_diagnoses(self, patient_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get patient's recent diagnoses for AI context"""
        recent_key = self._get_key("patient_recent_diagnoses", patient_id)
        recent_diagnoses = self._get_with_fallback(recent_key) or []
        return recent_diagnoses[:limit]
    
    # ===== AI CHAT CONTEXT =====
    
    def build_ai_context_for_patient(self, patient_id: str) -> Dict[str, Any]:
        """Build comprehensive AI context including recent diagnoses"""
        try:
            # Get recent diagnoses from Redis
            recent_diagnoses = self.get_patient_recent_diagnoses(patient_id, limit=3)
            
            # Get cached patient context
            patient_context_key = self._get_key("patient_context", patient_id)
            patient_context = self._get_with_fallback(patient_context_key) or {}
            
            # Build comprehensive context
            ai_context = {
                'patient_id': patient_id,
                'patient_info': patient_context.get('patient_info', {}),
                'recent_diagnoses': recent_diagnoses,
                'recent_diagnoses_count': len(recent_diagnoses),
                'last_diagnosis_date': recent_diagnoses[0].get('created_at') if recent_diagnoses else None,
                'concern_data': patient_context.get('concern_data', {}),
                'context_generated_at': datetime.now().isoformat()
            }
            
            # Cache the AI context
            ai_context_key = self._get_key("ai_context", patient_id)
            self._set_with_fallback(ai_context_key, ai_context, expiry=1800)  # 30 minutes
            
            return ai_context
        except Exception as e:
            logger.error(f"Failed to build AI context for patient {patient_id}: {e}")
            return {'patient_id': patient_id, 'error': str(e)}
    
    def get_ai_context(self, patient_id: str) -> Dict[str, Any]:
        """Get cached AI context or build new one"""
        ai_context_key = self._get_key("ai_context", patient_id)
        context = self._get_with_fallback(ai_context_key)
        
        if not context:
            context = self.build_ai_context_for_patient(patient_id)
        
        return context
    
    def generate_ai_prompt_with_diagnosis_context(self, patient_id: str, user_message: str) -> str:
        """Generate AI prompt enriched with recent diagnosis context"""
        context = self.get_ai_context(patient_id)
        
        prompt = f"""You are CortexMD, an advanced medical AI assistant with access to this patient's complete medical history and recent diagnoses.

PATIENT CONTEXT:
- Patient ID: {patient_id}
- Patient Name: {context.get('patient_info', {}).get('patient_name', 'Unknown')}
- Current Status: {context.get('patient_info', {}).get('current_status', 'Active')}

RECENT DIAGNOSES ({context.get('recent_diagnoses_count', 0)} total):"""
        
        for i, diagnosis in enumerate(context.get('recent_diagnoses', [])[:3], 1):
            prompt += f"""
{i}. {diagnosis.get('primary_diagnosis', 'Unknown')} 
   - Date: {diagnosis.get('created_at', 'Unknown')}
   - Confidence: {diagnosis.get('confidence_score', 0) * 100:.1f}%
   - Risk Level: {diagnosis.get('risk_level', 'unknown').upper()}
   - Key Symptoms: {', '.join(diagnosis.get('key_symptoms', []))}
   - ICD Codes: {', '.join(diagnosis.get('icd_codes', []))}"""
        
        if not context.get('recent_diagnoses'):
            prompt += "\nNo recent diagnoses found."
        
        prompt += f"""

CONCERN RISK ASSESSMENT:
- Current Risk Level: {context.get('concern_data', {}).get('current_risk_level', 'Unknown')}
- Concern Score: {context.get('concern_data', {}).get('current_concern_score', 0) * 100:.1f}%

CURRENT USER MESSAGE: {user_message}

Please provide a helpful, contextual response based on this patient's complete medical profile and recent diagnoses. Reference specific diagnoses when relevant and maintain continuity with previous medical assessments."""
        
        return prompt
    
    # ===== ONTOLOGY CACHING =====
    
    def cache_ontology_mapping(self, term: str, mapping_data: Dict[str, Any]) -> bool:
        """Cache ontology mappings (UMLS, SNOMED, ICD) for fast lookup"""
        try:
            # Create hash key for the term
            term_hash = hashlib.md5(term.lower().encode()).hexdigest()
            ontology_key = self._get_key("ontology", term_hash)
            
            # Add metadata
            mapping_data['cached_at'] = datetime.now().isoformat()
            mapping_data['original_term'] = term
            
            return self._set_with_fallback(ontology_key, mapping_data, expiry=2592000)  # 30 days
        except Exception as e:
            logger.error(f"Failed to cache ontology mapping: {e}")
            return False
    
    def get_ontology_mapping(self, term: str) -> Optional[Dict[str, Any]]:
        """Get cached ontology mapping"""
        term_hash = hashlib.md5(term.lower().encode()).hexdigest()
        ontology_key = self._get_key("ontology", term_hash)
        return self._get_with_fallback(ontology_key)
    
    def cache_snomed_codes(self, diagnosis: str, snomed_data: Dict[str, Any]) -> bool:
        """Cache SNOMED CT codes for diagnoses"""
        snomed_key = self._get_key("snomed", diagnosis.lower().replace(' ', '_'))
        return self._set_with_fallback(snomed_key, snomed_data, expiry=2592000)  # 30 days
    
    def get_snomed_codes(self, diagnosis: str) -> Optional[Dict[str, Any]]:
        """Get cached SNOMED CT codes"""
        snomed_key = self._get_key("snomed", diagnosis.lower().replace(' ', '_'))
        return self._get_with_fallback(snomed_key)
    
    # ===== CHAT HISTORY =====
    
    def add_chat_message_with_context(self, patient_id: str, role: str, content: str, 
                                    diagnosis_context: Optional[str] = None) -> bool:
        """Add chat message with diagnosis context"""
        try:
            message = {
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'diagnosis_context': diagnosis_context,
                'patient_id': patient_id
            }
            
            chat_key = self._get_key("chat_history", patient_id)
            
            if self.redis_available:
                # Use Redis list for chat history
                self.redis_client.lpush(chat_key, json.dumps(message, default=str))
                self.redis_client.ltrim(chat_key, 0, 99)  # Keep last 100 messages
                self.redis_client.expire(chat_key, 604800)  # 7 days
            else:
                # Memory fallback
                if chat_key not in self.memory_fallback:
                    self.memory_fallback[chat_key] = {'value': [], 'expires_at': datetime.now() + timedelta(days=7)}
                self.memory_fallback[chat_key]['value'].insert(0, message)
                self.memory_fallback[chat_key]['value'] = self.memory_fallback[chat_key]['value'][:100]
            
            return True
        except Exception as e:
            logger.error(f"Failed to add chat message: {e}")
            return False
    
    def get_chat_history_with_context(self, patient_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get chat history with diagnosis context"""
        try:
            chat_key = self._get_key("chat_history", patient_id)
            
            if self.redis_available:
                messages = self.redis_client.lrange(chat_key, 0, limit - 1)
                return [json.loads(msg) for msg in messages]
            else:
                cached = self.memory_fallback.get(chat_key, {})
                if datetime.now() < cached.get('expires_at', datetime.now()):
                    return cached.get('value', [])[:limit]
                return []
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            return []
    
    # ===== PERFORMANCE MONITORING =====
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        try:
            if self.redis_available:
                info = self.redis_client.info()
                return {
                    'redis_available': True,
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory_human': info.get('used_memory_human', '0B'),
                    'total_commands_processed': info.get('total_commands_processed', 0),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0),
                    'hit_rate': info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1)
                }
            else:
                return {
                    'redis_available': False,
                    'memory_cache_keys': len(self.memory_fallback),
                    'fallback_mode': True
                }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Compatibility alias for simple service get_stats()"""
        return self.get_cache_stats()

    # Simple compatibility aliases for app usage
    def cache_diagnosis(self, session_id: str, diagnosis_data: Dict[str, Any]) -> bool:
        return self.cache_diagnosis_session(session_id, diagnosis_data, diagnosis_data.get('patient_input', {}).get('patient_id'))

    def get_diagnosis(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.get_diagnosis_session(session_id)

    def get_chat_context(self, patient_id: str) -> Dict[str, Any]:
        """Alias to return AI context for a patient"""
        return self.get_ai_context(patient_id)

# Global enhanced Redis service instance
enhanced_redis_service = EnhancedRedisService()

def get_redis_service() -> EnhancedRedisService:
    """Compatibility accessor used across the app"""
    return enhanced_redis_service
