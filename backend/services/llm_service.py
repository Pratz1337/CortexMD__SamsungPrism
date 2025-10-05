#!/usr/bin/env python3
"""
Simple LLM Service for CONCERN Early Warning System
Provides a unified interface for language model operations
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class LLMService:
    """Simple LLM service using available AI backends"""
    
    def __init__(self):
        self.groq_client = None
        self.gemini_model = None
        
        # Initialize with ai_key_manager
        try:
            try:
                from utils.ai_key_manager import get_groq_client, get_gemini_model
            except ImportError:
                from ..utils.ai_key_manager import get_groq_client, get_gemini_model
            
            # Try Groq first
            try:
                self.groq_client = get_groq_client()
                if self.groq_client:
                    logger.info("✅ LLM Service: Groq client initialized")
            except Exception as e:
                logger.warning(f"⚠️ LLM Service: Groq initialization failed: {e}")
            
            # Try Gemini as backup
            try:
                self.gemini_model = get_gemini_model('gemini-1.5-flash')
                if self.gemini_model:
                    logger.info("✅ LLM Service: Gemini model initialized")
            except Exception as e:
                logger.warning(f"⚠️ LLM Service: Gemini initialization failed: {e}")
                
        except ImportError:
            logger.warning("⚠️ LLM Service: ai_key_manager not available")
        
        self.has_llm = bool(self.groq_client or self.gemini_model)
        if not self.has_llm:
            logger.warning("⚠️ LLM Service: No AI backends available")
    
    def generate_response(self, prompt: str, temperature: float = 0.3, max_tokens: int = 500) -> str:
        """
        Generate response using available LLM backend
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        # Try Groq first (faster)
        if self.groq_client:
            try:
                completion = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant specialized in medical analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.8
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"⚠️ Groq generation failed: {e}, trying Gemini")
        
        # Try Gemini as backup
        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': temperature,
                        'max_output_tokens': max_tokens,
                        'top_p': 0.8
                    }
                )
                
                # Parse Gemini response
                if response.candidates and response.candidates[0].content.parts:
                    text = ""
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text'):
                            text += part.text
                    return text.strip()
                elif hasattr(response, 'text'):
                    return response.text.strip()
                else:
                    return str(response).strip()
                    
            except Exception as e:
                logger.error(f"❌ Gemini generation failed: {e}")
        
        # Fallback response
        logger.warning("⚠️ No LLM available, using fallback response")
        return "LLM service not available. Please check your API keys."
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.has_llm
