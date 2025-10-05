"""
Redis-enabled Chat Service for CortexMD
Provides fast caching for patient context and chat history
"""

import json
import redis
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RedisChatService:
    """Redis-powered chat service with patient context caching"""
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=1):
        try:
            self.redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                db=redis_db, 
                decode_responses=True,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info("✅ Redis chat service initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable, using in-memory fallback: {e}")
            self.redis_available = False
            self.memory_cache = {}
    
    def get_patient_context_key(self, patient_id: str) -> str:
        """Generate Redis key for patient context"""
        return f"patient_context:{patient_id}"
    
    def get_chat_history_key(self, patient_id: str) -> str:
        """Generate Redis key for chat history"""
        return f"chat_history:{patient_id}"
    
    def cache_patient_context(self, patient_id: str, context: Dict[str, Any]) -> bool:
        """Cache comprehensive patient context in Redis"""
        try:
            if self.redis_available:
                key = self.get_patient_context_key(patient_id)
                context_json = json.dumps(context, default=str)
                self.redis_client.setex(key, 3600, context_json)  # 1 hour cache
                return True
            else:
                # Fallback to memory
                self.memory_cache[f"context_{patient_id}"] = context
                return True
        except Exception as e:
            logger.error(f"Failed to cache patient context: {e}")
            return False
    
    def get_patient_context(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached patient context"""
        try:
            if self.redis_available:
                key = self.get_patient_context_key(patient_id)
                cached_data = self.redis_client.get(key)
                if cached_data:
                    return json.loads(cached_data)
            else:
                # Fallback to memory
                return self.memory_cache.get(f"context_{patient_id}")
        except Exception as e:
            logger.error(f"Failed to get patient context: {e}")
        return None
    
    def add_chat_message(self, patient_id: str, message: Dict[str, Any]) -> bool:
        """Add message to chat history"""
        try:
            message['timestamp'] = datetime.now().isoformat()
            
            if self.redis_available:
                key = self.get_chat_history_key(patient_id)
                # Get existing history
                existing = self.redis_client.get(key)
                history = json.loads(existing) if existing else []
                
                # Add new message
                history.append(message)
                
                # Keep only last 50 messages
                if len(history) > 50:
                    history = history[-50:]
                
                # Save back to Redis
                self.redis_client.setex(key, 7200, json.dumps(history, default=str))  # 2 hours
                return True
            else:
                # Fallback to memory
                key = f"chat_{patient_id}"
                if key not in self.memory_cache:
                    self.memory_cache[key] = []
                self.memory_cache[key].append(message)
                if len(self.memory_cache[key]) > 50:
                    self.memory_cache[key] = self.memory_cache[key][-50:]
                return True
        except Exception as e:
            logger.error(f"Failed to add chat message: {e}")
            return False
    
    def get_chat_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get chat history for patient"""
        try:
            if self.redis_available:
                key = self.get_chat_history_key(patient_id)
                cached_data = self.redis_client.get(key)
                if cached_data:
                    return json.loads(cached_data)
            else:
                # Fallback to memory
                return self.memory_cache.get(f"chat_{patient_id}", [])
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
        return []
    
    def build_patient_context(self, patient_id: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive patient context for AI"""
        context = {
            'patient_id': patient_id,
            'patient_info': patient_data.get('patient_info', {}),
            'diagnosis_history': patient_data.get('diagnosis_history', []),
            'concern_data': patient_data.get('concern_data', {}),
            'clinical_notes': [],
            'recent_visits': [],
            'last_updated': datetime.now().isoformat()
        }
        
        # Add clinical notes from CONCERN system
        try:
            from concern_ews_engine import concern_engine
            dashboard = concern_engine.get_patient_dashboard(patient_id)
            context['concern_dashboard'] = dashboard
        except Exception as e:
            logger.warning(f"Could not get CONCERN data: {e}")
        
        return context
    
    def generate_ai_prompt(self, patient_id: str, user_message: str) -> str:
        """Generate AI prompt with full patient context"""
        context = self.get_patient_context(patient_id)
        chat_history = self.get_chat_history(patient_id)
        
        if not context:
            return f"User message: {user_message}\n\nPlease respond as a medical AI assistant."
        
        prompt = f"""You are CortexMD, an advanced medical AI assistant. You have access to comprehensive patient data and should provide contextual, helpful responses.

PATIENT CONTEXT:
- Patient ID: {context['patient_id']}
- Patient Name: {context.get('patient_info', {}).get('patient_name', 'Unknown')}
- Current Status: {context.get('patient_info', {}).get('current_status', 'Unknown')}
- Admission Date: {context.get('patient_info', {}).get('admission_date', 'Unknown')}

DIAGNOSIS HISTORY:
{self._format_diagnosis_history(context.get('diagnosis_history', []))}

CONCERN RISK ASSESSMENT:
- Current Risk Level: {context.get('concern_data', {}).get('current_risk_level', 'Unknown')}
- Concern Score: {context.get('concern_data', {}).get('current_concern_score', 0) * 100:.1f}%
- Risk Factors: {', '.join(context.get('concern_data', {}).get('risk_factors', []))}

RECENT ACTIVITY (24h):
- Clinical Notes: {context.get('concern_data', {}).get('notes_24h', 0)}
- Patient Visits: {context.get('concern_data', {}).get('visits_24h', 0)}

RECENT CHAT HISTORY:
{self._format_chat_history(chat_history[-5:])}

USER MESSAGE: {user_message}

Please provide a helpful, contextual response based on this patient's complete medical profile. Be professional, accurate, and reference relevant patient data when appropriate."""

        return prompt
    
    def _format_diagnosis_history(self, history: List[Dict]) -> str:
        """Format diagnosis history for AI prompt"""
        if not history:
            return "No previous diagnoses recorded."
        
        formatted = []
        for diag in history[-3:]:  # Last 3 diagnoses
            date = diag.get('created_at', 'Unknown date')
            diagnosis = diag.get('primary_diagnosis', 'Unknown')
            confidence = diag.get('confidence_score', 0)
            formatted.append(f"- {date}: {diagnosis} (Confidence: {confidence * 100:.1f}%)")
        
        return "\n".join(formatted)
    
    def _format_chat_history(self, history: List[Dict]) -> str:
        """Format recent chat history for AI prompt"""
        if not history:
            return "No previous chat messages."
        
        formatted = []
        for msg in history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')[:100]  # Truncate long messages
            timestamp = msg.get('timestamp', '')
            formatted.append(f"- {role}: {content}...")
        
        return "\n".join(formatted)

# Global instance
redis_chat_service = RedisChatService()
