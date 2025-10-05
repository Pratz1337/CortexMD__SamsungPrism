"""
Hybrid AI Engine for CortexMD
Intelligently uses Groq Llama for medical reasoning and Gemini for multimodal inputs
"""
import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# AI Model imports
import google.generativeai as genai

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import utility functions
try:
    from ..utils.gemini_response_handler import safe_generate_content
except ImportError:
    from utils.gemini_response_handler import safe_generate_content

logger = logging.getLogger(__name__)

class ModelType(Enum):
    GROQ_LLAMA_70B = "llama-3.3-70b-versatile"
    GROQ_LLAMA_8B = "meta-llama/llama-guard-4-12b"
    GEMINI_FLASH = "gemini-2.5-flash"

class TaskType(Enum):
    MEDICAL_REASONING = "medical_reasoning"
    CONFIDENCE_ANALYSIS = "confidence_analysis"
    FOL_EXTRACTION = "fol_extraction"
    MULTIMODAL_DIAGNOSIS = "multimodal_diagnosis"
    IMAGE_ANALYSIS = "image_analysis"
    CLINICAL_EXPLANATION = "clinical_explanation"

@dataclass
class AIResponse:
    content: str
    model_used: ModelType
    processing_time: float
    confidence: Optional[float] = None
    reasoning: Optional[List[str]] = None

class HybridAIEngine:
    """Hybrid AI engine that intelligently routes tasks to optimal models"""
    
    def __init__(self):
        """Initialize both Groq and Gemini clients"""
        
        # Initialize Groq and Gemini clients using ai_key_manager (supports multiple keys)
        get_groq_client = None
        get_gemini_model = None
        
        try:
            from utils.ai_key_manager import get_groq_client, get_gemini_model
        except Exception:
            try:
                from ..utils.ai_key_manager import get_groq_client, get_gemini_model
            except Exception:
                # If we can't import, try to continue without throwing an error immediately
                import logging
                logging.warning("Could not import ai_key_manager, will try alternative initialization")

        if get_groq_client is not None:
            try:
                self.groq_client = get_groq_client()
                if not self.groq_client:
                    raise ValueError("GROQ client initialization failed (no keys or error)")
            except Exception as e:
                raise ValueError(f"GROQ client initialization failed: {str(e)}")
        else:
            raise ValueError("ai_key_manager.get_groq_client not available")

        if get_gemini_model is not None:
            try:
                self.gemini_model = get_gemini_model('gemini-2.5-flash')
                if not self.gemini_model:
                    raise ValueError("Gemini model initialization failed (no keys or error)")
            except Exception as e:
                raise ValueError(f"Gemini model initialization failed: {str(e)}")
        else:
            raise ValueError("ai_key_manager.get_gemini_model not available")
        
        # Model routing logic
        self.model_routing = {
            TaskType.MEDICAL_REASONING: ModelType.GROQ_LLAMA_70B,
            TaskType.CONFIDENCE_ANALYSIS: ModelType.GROQ_LLAMA_70B,
            TaskType.FOL_EXTRACTION: ModelType.GROQ_LLAMA_8B,  # Faster for structured extraction
            TaskType.MULTIMODAL_DIAGNOSIS: ModelType.GEMINI_FLASH,
            TaskType.IMAGE_ANALYSIS: ModelType.GEMINI_FLASH,
            TaskType.CLINICAL_EXPLANATION: ModelType.GROQ_LLAMA_70B,
        }
        
        logger.info("Hybrid AI Engine initialized with Groq and Gemini support")
    
    async def generate_response(
        self, 
        prompt: str, 
        task_type: TaskType,
        images: Optional[List[Any]] = None,
        force_model: Optional[ModelType] = None
    ) -> AIResponse:
        """Generate AI response using the optimal model for the task"""
        
        import time
        start_time = time.time()
        
        # Determine which model to use
        model_to_use = force_model or self.model_routing.get(task_type, ModelType.GROQ_LLAMA_70B)
        
        # If images are provided, force use of Gemini
        if images:
            model_to_use = ModelType.GEMINI_FLASH
            logger.info(f"Images detected, using {model_to_use.value} for multimodal processing")
        
        try:
            # If configured, try parallel calls for non-conflicting tasks to improve latency
            if os.getenv('ENABLE_PARALLEL_AGENTS', '0') in ('1', 'true', 'True') and not images:
                # Run both Groq and Gemini in parallel and use the fastest successful
                response = await self._race_groq_and_gemini(prompt, model_to_use)
            else:
                if model_to_use in [ModelType.GROQ_LLAMA_70B, ModelType.GROQ_LLAMA_8B]:
                    response = await self._generate_groq_response(prompt, model_to_use)
                else:
                    response = await self._generate_gemini_response(prompt, images)
            
            processing_time = time.time() - start_time
            
            return AIResponse(
                content=response,
                model_used=model_to_use,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error with {model_to_use.value}: {e}")
            
            # Fallback logic
            fallback_model = self._get_fallback_model(model_to_use)
            logger.info(f"Falling back to {fallback_model.value}")
            
            try:
                if fallback_model in [ModelType.GROQ_LLAMA_70B, ModelType.GROQ_LLAMA_8B]:
                    response = await self._generate_groq_response(prompt, fallback_model)
                else:
                    response = await self._generate_gemini_response(prompt, images)
                
                processing_time = time.time() - start_time
                
                return AIResponse(
                    content=response,
                    model_used=fallback_model,
                    processing_time=processing_time
                )
                
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return AIResponse(
                    content=f"Error: Both primary and fallback models failed. {str(e)}",
                    model_used=model_to_use,
                    processing_time=time.time() - start_time
                )
    
    async def _generate_groq_response(self, prompt: str, model_type: ModelType) -> str:
        """Generate response using Groq Llama models"""
        
        try:
            # Use asyncio to run the sync Groq call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.groq_client.chat.completions.create(
                    model=model_type.value,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert medical AI assistant specializing in clinical diagnosis and medical reasoning. Provide accurate, evidence-based medical analysis."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=4000,
                    top_p=0.9
                )
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    async def _generate_gemini_response(self, prompt: str, images: Optional[List[Any]] = None) -> str:
        """Generate response using Gemini (supports multimodal)"""
        
        try:
            # Prepare content for Gemini
            if images:
                # Multimodal input
                content = [prompt] + images
            else:
                # Text-only input
                content = prompt
            
            # Use asyncio to run the safe Gemini call
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None,
                lambda: safe_generate_content(self.gemini_model, content)
            )

            return response_text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def _get_fallback_model(self, failed_model: ModelType) -> ModelType:
        """Get fallback model when primary fails"""
        
        fallback_map = {
            ModelType.GROQ_LLAMA_70B: ModelType.GROQ_LLAMA_8B,
            ModelType.GROQ_LLAMA_8B: ModelType.GEMINI_FLASH,
            ModelType.GEMINI_FLASH: ModelType.GROQ_LLAMA_70B,
        }
        
        return fallback_map.get(failed_model, ModelType.GROQ_LLAMA_8B)

    async def _race_groq_and_gemini(self, prompt: str, preferred_model: ModelType) -> str:
        """Run Groq and Gemini simultaneously and return the fastest successful result.

        This helps reduce tail latency by racing both providers. Only used when
        ENABLE_PARALLEL_AGENTS is set and no multimodal images are involved.
        """
        async def run_groq():
            try:
                return await self._generate_groq_response(prompt, preferred_model if preferred_model.name.startswith('GROQ') else ModelType.GROQ_LLAMA_70B)
            except Exception as e:
                return None

        async def run_gemini():
            try:
                return await self._generate_gemini_response(prompt)
            except Exception:
                return None

        tasks = [asyncio.create_task(run_groq()), asyncio.create_task(run_gemini())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Cancel remaining tasks
        for t in pending:
            t.cancel()

        for d in done:
            try:
                res = d.result()
                if res:
                    return res
            except Exception:
                continue

        # If none succeeded, raise
        raise Exception('Parallel Groq/Gemini race failed to produce a result')
    
    async def analyze_medical_confidence(
        self, 
        diagnosis: str, 
        patient_data: Dict[str, Any],
        reasoning_paths: List[str]
    ) -> Dict[str, Any]:
        """Analyze medical confidence using Groq Llama 70B for optimal reasoning"""
        
        prompt = f"""
As an expert medical AI, analyze the confidence of this diagnosis with precision:

DIAGNOSIS: {diagnosis}

PATIENT DATA: {json.dumps(patient_data, indent=2)}

MEDICAL REASONING: {' '.join(reasoning_paths)}

Analyze and provide confidence assessment in this JSON format:
{{
    "symptom_match_score": <0.0-1.0>,
    "evidence_quality_score": <0.0-1.0>,
    "medical_literature_score": <0.0-1.0>,
    "uncertainty_score": <0.0-1.0>,
    "overall_confidence": <0.0-1.0>,
    "reasoning": [
        "Detailed medical reasoning point 1",
        "Detailed medical reasoning point 2",
        "Detailed medical reasoning point 3"
    ],
    "risk_factors": ["risk1", "risk2"],
    "contradictory_evidence": ["evidence1", "evidence2"]
}}

Provide only valid JSON response.
"""
        
        response = await self.generate_response(prompt, TaskType.CONFIDENCE_ANALYSIS)
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                confidence_data = json.loads(json_match.group())
                confidence_data['model_used'] = response.model_used.value
                confidence_data['processing_time'] = response.processing_time
                return confidence_data
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Error parsing confidence analysis: {e}")
            # Return default confidence metrics
            return {
                "symptom_match_score": 0.5,
                "evidence_quality_score": 0.5,
                "medical_literature_score": 0.5,
                "uncertainty_score": 0.5,
                "overall_confidence": 0.5,
                "reasoning": [f"Error in analysis: {str(e)}"],
                "risk_factors": [],
                "contradictory_evidence": [],
                "model_used": response.model_used.value,
                "processing_time": response.processing_time
            }
    
    async def extract_fol_predicates(self, medical_text: str) -> List[Dict[str, Any]]:
        """Extract FOL predicates using fast Groq Llama 8B"""
        
        prompt = f"""
Extract medical First-Order Logic predicates from this clinical text:

TEXT: {medical_text}

Extract predicates in this JSON format:
[
    {{
        "predicate_type": "has_symptom|has_condition|takes_medication|has_lab_value|has_vital_sign",
        "subject": "patient",
        "predicate": "predicate_name",
        "object": "object_value",
        "confidence": <0.0-1.0>,
        "negated": false,
        "temporal": "current|past|future"
    }}
]

Focus on clear medical facts. Provide only valid JSON array.
"""
        
        response = await self.generate_response(prompt, TaskType.FOL_EXTRACTION)
        
        try:
            import re
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                predicates = json.loads(json_match.group())
                return predicates
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error extracting FOL predicates: {e}")
            return []
    
    async def generate_multimodal_diagnosis(
        self, 
        text_data: str,
        images: Optional[List[Any]] = None,
        fhir_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate diagnosis using multimodal Gemini"""
        
        prompt = f"""
Analyze this multimodal medical case and provide diagnosis:

CLINICAL TEXT: {text_data}

"""
        
        if fhir_data:
            prompt += f"FHIR DATA: {json.dumps(fhir_data, indent=2)}\n\n"
        
        if images:
            prompt += f"MEDICAL IMAGES: {len(images)} images provided for analysis\n\n"
        
        prompt += """
Provide comprehensive diagnosis in JSON format:
{
    "primary_diagnosis": "most likely diagnosis",
    "differential_diagnoses": ["diff1", "diff2", "diff3"],
    "reasoning_paths": ["reasoning1", "reasoning2"],
    "confidence_estimate": <0.0-1.0>,
    "recommended_tests": ["test1", "test2"],
    "clinical_notes": "additional observations"
}

Provide only valid JSON response.
"""
        
        response = await self.generate_response(
            prompt, 
            TaskType.MULTIMODAL_DIAGNOSIS,
            images=images
        )
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                diagnosis_data = json.loads(json_match.group())
                diagnosis_data['model_used'] = response.model_used.value
                diagnosis_data['processing_time'] = response.processing_time
                return diagnosis_data
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Error parsing diagnosis: {e}")
            return {
                "primary_diagnosis": "Analysis error",
                "differential_diagnoses": [],
                "reasoning_paths": [f"Error: {str(e)}"],
                "confidence_estimate": 0.0,
                "recommended_tests": [],
                "clinical_notes": "Unable to process due to error",
                "model_used": response.model_used.value,
                "processing_time": response.processing_time
            }
