"""
Enhanced FOL Service with Proper Gemini Pro Integration
Optimized for medical diagnosis verification
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import google.generativeai as genai
import re
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class FOLPredicate:
    """FOL Predicate with verification data"""
    predicate_type: str
    subject: str
    object: str
    confidence: float
    verified: bool = False
    evidence: List[str] = None
    reasoning: str = ""

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

    def to_fol_string(self) -> str:
        return f"{self.predicate_type}({self.subject}, {self.object})"

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class FOLVerificationReport:
    """FOL Verification Report with comprehensive results"""
    total_predicates: int
    verified_predicates: int
    failed_predicates: int
    overall_confidence: float
    verification_time: float
    detailed_results: List[Dict]
    medical_reasoning_summary: str
    disease_probabilities: Dict[str, float]
    clinical_recommendations: List[str]

    def to_dict(self) -> Dict:
        return {
            "total_predicates": self.total_predicates,
            "verified_predicates": self.verified_predicates,
            "failed_predicates": self.failed_predicates,
            "overall_confidence": self.overall_confidence,
            "verification_time": self.verification_time,
            "detailed_results": self.detailed_results,
            "medical_reasoning_summary": self.medical_reasoning_summary,
            "disease_probabilities": self.disease_probabilities,
            "clinical_recommendations": self.clinical_recommendations,
            "verification_summary": {
                "success_rate": self.verified_predicates / self.total_predicates if self.total_predicates > 0 else 0,
                "confidence_category": self._get_confidence_category(),
                "clinical_assessment": self._get_clinical_assessment()
            }
        }

    def _get_confidence_category(self) -> str:
        """Get confidence category based on overall confidence"""
        if self.overall_confidence >= 0.9:
            return "Very High Confidence"
        elif self.overall_confidence >= 0.7:
            return "High Confidence"
        elif self.overall_confidence >= 0.5:
            return "Moderate Confidence"
        elif self.overall_confidence >= 0.3:
            return "Low Confidence"
        else:
            return "Very Low Confidence"

    def _get_clinical_assessment(self) -> str:
        """Get clinical assessment based on verification results"""
        success_rate = self.verified_predicates / self.total_predicates if self.total_predicates > 0 else 0

        if success_rate >= 0.9 and self.overall_confidence >= 0.8:
            return "AI diagnosis is strongly supported by clinical evidence. High reliability for clinical decision-making."
        elif success_rate >= 0.7 and self.overall_confidence >= 0.6:
            return "AI diagnosis is well-supported by available evidence. Suitable for clinical consideration with physician review."
        elif success_rate >= 0.5:
            return "Mixed evidence for AI diagnosis. Requires careful clinical evaluation and additional testing."
        else:
            return "Limited evidence for AI diagnosis. Significant clinical review and additional workup recommended."

class FOLVerificationService:
    """
    Enhanced FOL Service using Gemini Pro for medical predicate extraction
    and verification with optimized prompts and caching
    """
    
    def __init__(self):
        """Initialize Gemini FOL Service"""
        # Prefer ai_key_manager for Gemini model rotation
        try:
            from ai_key_manager import get_gemini_model
            self.model = get_gemini_model('gemini-2.5-flash')
            self.has_gemini = bool(self.model)
            if self.has_gemini:
                logger.info("✅ Gemini FOL Service initialized via ai_key_manager with gemini-2.5-flash")
        except Exception:
            self.api_key = os.getenv('GOOGLE_API_KEY')
            self.has_gemini = bool(self.api_key)
            if self.has_gemini:
                try:
                    genai.configure(api_key=self.api_key)
                    self.model = genai.GenerativeModel(
                        'gemini-2.5-flash',
                        generation_config={
                            'temperature': 0.1,
                            'top_p': 0.8,
                            'top_k': 20,
                            'max_output_tokens': 2048
                        }
                    )
                    logger.info("✅ Gemini FOL Service initialized with gemini-2.5-flash")
                except Exception as e:
                    logger.warning(f"⚠️ Gemini initialization failed: {e}")
                    self.has_gemini = False
                    self.model = None
            else:
                logger.warning("⚠️ GOOGLE_API_KEY not found, using fallback FOL verification")
                self.model = None
        
        # Cache for repeated queries
        self.cache = {}
        self.max_cache_size = 100
    
    async def extract_fol_predicates(
        self,
        clinical_text: str,
        patient_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract FOL predicates from clinical text using Gemini Pro
        
        Args:
            clinical_text: Medical text to analyze
            patient_data: Optional patient data for context
            
        Returns:
            Dictionary with predicates and verification results
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(clinical_text, patient_data)
            if cache_key in self.cache:
                logger.info("📋 Using cached FOL results")
                return self.cache[cache_key]
            
            # Extract predicates using Gemini
            predicates = await self._extract_predicates_with_gemini(clinical_text)
            
            # Verify against patient data if provided
            if patient_data:
                verification_results = await self._verify_predicates(predicates, patient_data)
            else:
                verification_results = []
            
            result = {
                'predicates': predicates,
                'verification_results': verification_results,
                'extraction_method': 'gemini_2.5_flash',
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            # Cache the result
            self._update_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ FOL extraction failed: {e}")
            return {
                'predicates': [],
                'verification_results': [],
                'extraction_method': 'gemini_2.5_flash',
                'error': str(e),
                'success': False
            }
    
    async def _extract_predicates_with_gemini(self, clinical_text: str) -> List[FOLPredicate]:
        """
        Extract predicates using optimized Gemini prompt
        """
        prompt = f"""You are a medical AI expert specializing in First-Order Logic (FOL) predicate extraction for clinical diagnosis verification.

TASK: Extract precise FOL predicates from the following clinical text. Each predicate must be medically accurate and verifiable.

CLINICAL TEXT:
{clinical_text}

REQUIRED FOL PREDICATE TYPES:
1. has_symptom(patient, symptom_name) - Patient symptoms
2. has_condition(patient, condition_name) - Diagnosed conditions
3. has_vital_sign(patient, vital_type, value) - Vital sign measurements
4. takes_medication(patient, medication_name) - Current medications
5. has_lab_value(patient, lab_test, value) - Laboratory results
6. has_finding(patient, clinical_finding) - Clinical examination findings
7. has_risk_factor(patient, risk_factor) - Risk factors
8. has_procedure(patient, procedure_name) - Medical procedures performed

EXTRACTION RULES:
- Only extract information explicitly stated or strongly implied
- Use medical terminology accurately
- Include temporal context when available (acute, chronic, recent)
- Normalize values (e.g., "high blood pressure" → "hypertension")
- Include severity qualifiers (mild, moderate, severe)

OUTPUT FORMAT (JSON):
{{
    "predicates": [
        {{
            "type": "predicate_type",
            "subject": "patient",
            "object": "specific_medical_term",
            "confidence": 0.0-1.0,
            "evidence": "supporting text from input",
            "temporal": "acute/chronic/recent/null",
            "severity": "mild/moderate/severe/null"
        }}
    ],
    "medical_entities": {{
        "symptoms": [],
        "conditions": [],
        "medications": [],
        "lab_values": [],
        "vital_signs": [],
        "procedures": [],
        "risk_factors": []
    }},
    "clinical_reasoning": "Brief reasoning about the clinical picture"
}}

Respond ONLY with valid JSON. No additional text."""

        try:
            # Check if Gemini is available
            if not self.has_gemini or not self.model:
                logger.warning("⚠️ Gemini not available, using fallback extraction")
                return self._fallback_extraction(clinical_text)
            
            # Generate response with Gemini
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            json_text = response.text.strip()
            # Remove markdown formatting if present
            json_text = re.sub(r'```json?\s*', '', json_text)
            json_text = re.sub(r'```\s*$', '', json_text)
            
            result = json.loads(json_text)
            
            # Convert to FOLPredicate objects
            predicates = []
            for pred_data in result.get('predicates', []):
                predicate = FOLPredicate(
                    predicate_type=pred_data.get('type', 'unknown'),
                    subject=pred_data.get('subject', 'patient'),
                    object=pred_data.get('object', ''),
                    confidence=float(pred_data.get('confidence', 0.8)),
                    evidence=[pred_data.get('evidence', '')],
                    reasoning=pred_data.get('clinical_reasoning', '')
                )
                predicates.append(predicate)
            
            logger.info(f"✅ Extracted {len(predicates)} predicates with Gemini")
            return predicates
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse Gemini response: {e}")
            # Fallback to regex extraction
            return self._fallback_extraction(clinical_text)
        except Exception as e:
            logger.error(f"❌ Gemini extraction error: {e}")
            return self._fallback_extraction(clinical_text)
    
    def _fallback_extraction(self, clinical_text: str) -> List[FOLPredicate]:
        """
        Enhanced fallback extraction using regex patterns and medical knowledge
        """
        predicates = []
        text_lower = clinical_text.lower()
        
        logger.info("🔄 Using fallback FOL extraction (regex-based)")
        
        # Enhanced patterns for medical entities with better matching
        symptom_patterns = [
            r'(?:patient\s+has|experiencing|reports|complains\s+of|suffers\s+from)\s+([^.;,]+?)(?:\s+symptoms?)?(?:\.|,|;|and|$)',
            r'(chest\s+pain|shortness\s+of\s+breath|dyspnea|fever|headache|nausea|vomiting|dizziness|fatigue|weakness|thigh\s+mass|mass|pain)',
            r'symptoms?\s+include\s*:?\s*([^.;]+)',
        ]
        
        condition_patterns = [
            r'(?:diagnosed\s+with|diagnosis\s+(?:of|:)\s*|primary\s+diagnosis\s*:?\s*|history\s+of|has\s+(?:a\s+)?(?:history\s+of\s+)?)\s*([^.;,\n]+?)(?:\s*\(|$|\.|,|;)',
            r'(soft\s+tissue\s+sarcoma|myxofibrosarcoma|sarcoma|hypertension|diabetes|asthma|copd|heart\s+failure|myocardial\s+infarction|high-grade)',
            r'(?:condition|disease)\s*:?\s*([^.;,\n]+)',
        ]
        
        medication_patterns = [
            r'(?:taking|prescribed|on|current\s+medications?)\s*:?\s*([a-zA-Z][a-zA-Z0-9\-\s]{2,30})(?:\s+\d+\s*mg|\.|,|;|and|$)',
            r'\b(aspirin|metformin|lisinopril|atorvastatin|insulin|warfarin|prednisone)\b',
            r'(?:patient\s+(?:is\s+)?(?:taking|on))\s+([a-zA-Z][a-zA-Z0-9\-\s]{2,20})(?:\s+\d+\s*mg|\.|$)',
        ]
        
        # Extract symptoms
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    object_text = match.group(1).strip()
                    if len(object_text) > 2 and object_text not in ['the', 'and', 'or', 'with']:
                        # Clean up the text
                        object_text = re.sub(r'\s+', ' ', object_text).strip()
                        predicate = FOLPredicate(
                            predicate_type='has_symptom',
                            subject='patient',
                            object=object_text,
                            confidence=0.75,
                            evidence=[match.group(0).strip()],
                            reasoning='Extracted via enhanced symptom pattern matching'
                        )
                        predicates.append(predicate)
        
        # Extract conditions
        for pattern in condition_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    object_text = match.group(1).strip()
                    if len(object_text) > 2 and object_text not in ['the', 'and', 'or', 'with']:
                        # Clean up the text
                        object_text = re.sub(r'\s+', ' ', object_text).strip()
                        predicate = FOLPredicate(
                            predicate_type='has_condition',
                            subject='patient',
                            object=object_text,
                            confidence=0.8,
                            evidence=[match.group(0).strip()],
                            reasoning='Extracted via enhanced condition pattern matching'
                        )
                        predicates.append(predicate)
        
        # Extract medications
        for pattern in medication_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    object_text = match.group(1).strip()
                    # Filter out clearly non-medication text
                    if (len(object_text) > 2 and len(object_text) < 30 and 
                        object_text not in ['the', 'and', 'or', 'with', 'diagnosis', 'assessment', 'based', 'imaging', 'characteristics', 'patient', 'presentation']):
                        # Clean up the text
                        object_text = re.sub(r'\s+', ' ', object_text).strip()
                        # Skip if it contains non-medication words
                        if not any(bad_word in object_text.lower() for bad_word in ['diagnosis', 'assessment', 'characteristics', 'presentation', 'evaluation', 'reasoning']):
                            predicate = FOLPredicate(
                                predicate_type='takes_medication',
                                subject='patient',
                                object=object_text,
                                confidence=0.7,
                                evidence=[match.group(0).strip()],
                                reasoning='Extracted via enhanced medication pattern matching'
                            )
                            predicates.append(predicate)
        
        # Remove duplicates and filter
        seen = set()
        unique_predicates = []
        for pred in predicates:
            key = f"{pred.predicate_type}:{pred.object.lower()}"
            if key not in seen:
                seen.add(key)
                unique_predicates.append(pred)
        
        logger.info(f"📝 Fallback extraction found {len(unique_predicates)} predicates")
        return unique_predicates
    
    async def _verify_predicates(
        self,
        predicates: List[FOLPredicate],
        patient_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Verify extracted predicates against patient data
        """
        verification_results = []
        
        for predicate in predicates:
            result = await self._verify_single_predicate(predicate, patient_data)
            verification_results.append(result)
        
        return verification_results
    
    async def _verify_single_predicate(
        self,
        predicate: FOLPredicate,
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify a single predicate against patient data
        """
        verified = False
        supporting_evidence = []
        contradicting_evidence = []
        
        object_text = predicate.object.replace('_', ' ')
        
        if predicate.predicate_type == 'has_symptom':
            # Check symptoms
            symptoms = patient_data.get('symptoms', [])
            for symptom in symptoms:
                if self._fuzzy_match(object_text, symptom.lower()):
                    verified = True
                    supporting_evidence.append(f"Patient reported: {symptom}")
                    break
            
            if not verified:
                contradicting_evidence.append(f"No record of symptom: {object_text}")
        
        elif predicate.predicate_type == 'has_condition':
            # Check conditions
            conditions = (patient_data.get('medical_history', []) + 
                         patient_data.get('current_conditions', []))
            
            for condition in conditions:
                if self._fuzzy_match(object_text, condition.lower()):
                    verified = True
                    supporting_evidence.append(f"Medical history: {condition}")
                    break
            
            if not verified:
                contradicting_evidence.append(f"No record of condition: {object_text}")
        
        elif predicate.predicate_type == 'takes_medication':
            # Check medications
            medications = patient_data.get('current_medications', [])
            
            for medication in medications:
                if self._fuzzy_match(object_text.split()[0], medication.lower().split()[0]):
                    verified = True
                    supporting_evidence.append(f"Current medication: {medication}")
                    break
            
            if not verified:
                contradicting_evidence.append(f"No record of medication: {object_text}")
        
        elif predicate.predicate_type == 'has_lab_value':
            # Check lab results
            lab_results = patient_data.get('lab_results', {})
            
            for lab_name, lab_value in lab_results.items():
                if self._fuzzy_match(object_text.split(':')[0], lab_name.lower()):
                    verified = True
                    supporting_evidence.append(f"Lab result: {lab_name} = {lab_value}")
                    break
            
            if not verified:
                contradicting_evidence.append(f"No lab result for: {object_text}")
        
        elif predicate.predicate_type == 'has_vital_sign':
            # Check vital signs
            vitals = patient_data.get('vitals', {})
            
            for vital_name, vital_value in vitals.items():
                if self._fuzzy_match(object_text.split(':')[0], vital_name.lower()):
                    verified = True
                    supporting_evidence.append(f"Vital sign: {vital_name} = {vital_value}")
                    break
            
            if not verified:
                contradicting_evidence.append(f"No vital sign for: {object_text}")
        
        # Update predicate verification status
        predicate.verified = verified
        if verified:
            predicate.confidence = min(0.95, predicate.confidence + 0.1)
        else:
            predicate.confidence = max(0.3, predicate.confidence - 0.2)
        
        return {
            'predicate': predicate.to_fol_string(),
            'verified': verified,
            'confidence': predicate.confidence,
            'supporting_evidence': supporting_evidence,
            'contradicting_evidence': contradicting_evidence,
            'predicate_type': predicate.predicate_type,
            'object': predicate.object
        }
    
    def _fuzzy_match(self, text1: str, text2: str, threshold: float = 0.5) -> bool:
        """
        Enhanced fuzzy matching for medical terms with better flexibility
        """
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Exact match
        if text1 == text2:
            return True
        
        # Substring match (more permissive)
        if text1 in text2 or text2 in text1:
            return True
        
        # Handle common medical term variations
        medical_synonyms = {
            'sarcoma': ['soft tissue sarcoma', 'myxofibrosarcoma'],
            'chest pain': ['chest discomfort', 'thoracic pain'],
            'shortness of breath': ['dyspnea', 'breathlessness', 'difficulty breathing'],
            'high-grade': ['high grade', 'aggressive'],
            'thigh mass': ['thigh tumor', 'leg mass', 'mass'],
        }
        
        # Check synonyms
        for term, synonyms in medical_synonyms.items():
            if (term in text1 and any(syn in text2 for syn in synonyms)) or \
               (term in text2 and any(syn in text1 for syn in synonyms)):
                return True
        
        # Word overlap (more permissive threshold)
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0
        
        # Also check if any significant word matches
        significant_words = words1 & words2
        if significant_words:
            # If there's at least one significant match and reasonable overlap
            return similarity >= max(0.3, threshold * 0.7)
        
        return similarity >= threshold
    
    def _get_cache_key(self, text: str, data: Optional[Dict]) -> str:
        """Generate cache key"""
        import hashlib
        content = f"{text}_{json.dumps(data, sort_keys=True) if data else ''}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _update_cache(self, key: str, value: Dict):
        """Update cache with LRU eviction"""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    async def verify_medical_explanation(
        self,
        explanation_text: str,
        patient_data: Dict[str, Any],
        patient_id: Optional[str] = None,
        diagnosis: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete FOL verification for medical explanations
        
        Args:
            explanation_text: Medical explanation to verify
            patient_data: Patient data for verification
            patient_id: Optional patient identifier
            diagnosis: Optional diagnosis for context
            context: Optional additional context (including diagnosis)
            
        Returns:
            Comprehensive verification report
        """
        import time
        self._verification_start_time = time.time()
        
        logger.info("🔬 Starting comprehensive FOL verification with Gemini")
        
        try:
            # Extract diagnosis from context if not provided directly
            if not diagnosis and context and 'diagnosis' in context:
                diagnosis = context['diagnosis']
            
            # Add patient_id to patient_data if provided
            if patient_id:
                patient_data = dict(patient_data)
                patient_data['patient_id'] = patient_id
            # Extract predicates
            extraction_result = await self.extract_fol_predicates(
                explanation_text, patient_data
            )
            
            predicates = extraction_result.get('predicates', [])
            verification_results = extraction_result.get('verification_results', [])
            
            # Calculate metrics
            total_predicates = len(predicates)
            verified_count = sum(1 for v in verification_results if v.get('verified', False))
            
            if total_predicates > 0:
                success_rate = verified_count / total_predicates
                avg_confidence = sum(p.confidence for p in predicates) / total_predicates
            else:
                success_rate = 0.0
                avg_confidence = 0.0
            
            # Generate medical reasoning
            reasoning = await self._generate_medical_reasoning(
                predicates, verification_results, diagnosis
            )
            
            # Determine overall status
            if success_rate >= 0.8:
                status = 'FULLY_VERIFIED'
                summary = f"High confidence: {verified_count}/{total_predicates} predicates verified"
            elif success_rate >= 0.6:
                status = 'PARTIALLY_VERIFIED'
                summary = f"Moderate confidence: {verified_count}/{total_predicates} predicates verified"
            else:
                status = 'LOW_CONFIDENCE'
                summary = f"Low confidence: {verified_count}/{total_predicates} predicates verified"
            
            # Create FOLVerificationReport object
            import time
            verification_time = time.time() - (getattr(self, '_verification_start_time', time.time()) or time.time())
            
            failed_count = total_predicates - verified_count
            
            report = FOLVerificationReport(
                total_predicates=total_predicates,
                verified_predicates=verified_count,
                failed_predicates=failed_count,
                overall_confidence=avg_confidence,
                verification_time=verification_time,
                detailed_results=verification_results,
                medical_reasoning_summary=reasoning[:500] if reasoning else "Medical reasoning analysis completed.",
                disease_probabilities={},  # Can be populated later if needed
                clinical_recommendations=[]  # Can be populated later if needed
            )
            
            logger.info(f"✅ FOL verification completed: {verified_count}/{total_predicates} predicates verified ({success_rate:.1%})")
            return report
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            # Return error report
            error_report = FOLVerificationReport(
                total_predicates=0,
                verified_predicates=0,
                failed_predicates=0,
                overall_confidence=0.0,
                verification_time=0.0,
                detailed_results=[],
                medical_reasoning_summary=f"FOL verification encountered an error: {str(e)}",
                disease_probabilities={},
                clinical_recommendations=[]
            )
            return error_report
    
    def _get_assessment_text(self, success_rate: float, confidence: float) -> str:
        """Get clinical assessment text based on metrics"""
        if success_rate >= 0.9 and confidence >= 0.8:
            return "AI diagnosis is strongly supported by clinical evidence. High reliability for clinical decision-making."
        elif success_rate >= 0.7 and confidence >= 0.6:
            return "AI diagnosis is well-supported by available evidence. Suitable for clinical consideration with physician review."
        elif success_rate >= 0.5:
            return "Mixed evidence for AI diagnosis. Requires careful clinical evaluation and additional testing."
        else:
            return "Limited evidence for AI diagnosis. Significant clinical review and additional workup recommended."
    
    async def _generate_medical_reasoning(
        self,
        predicates: List[FOLPredicate],
        verification_results: List[Dict],
        diagnosis: Optional[str] = None
    ) -> str:
        """
        Generate medical reasoning using Gemini
        """
        if not predicates:
            return "No predicates extracted for medical reasoning."
        
        # Prepare predicate summary
        verified_preds = [v for v in verification_results if v.get('verified')]
        unverified_preds = [v for v in verification_results if not v.get('verified')]
        
        prompt = f"""Based on the FOL predicate verification results, provide a brief medical reasoning summary.

VERIFIED PREDICATES ({len(verified_preds)}):
{json.dumps(verified_preds[:5], indent=2)}

UNVERIFIED PREDICATES ({len(unverified_preds)}):
{json.dumps(unverified_preds[:3], indent=2)}

{f"DIAGNOSIS CONTEXT: {diagnosis}" if diagnosis else ""}

Provide a 2-3 sentence medical reasoning summary explaining:
1. How well the clinical findings support the diagnosis
2. Any inconsistencies or missing evidence
3. Overall clinical confidence

Keep it concise and medically accurate."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate reasoning: {e}")
            
            # Fallback reasoning
            verified_ratio = len(verified_preds) / len(verification_results) if verification_results else 0
            if verified_ratio >= 0.8:
                return f"Strong clinical evidence with {len(verified_preds)} verified findings supporting the diagnosis."
            elif verified_ratio >= 0.5:
                return f"Moderate clinical evidence with {len(verified_preds)} verified and {len(unverified_preds)} unverified findings."
            else:
                return f"Limited clinical evidence with only {len(verified_preds)} verified findings. Additional assessment recommended."


# Wrapper class for backward compatibility
class EnhancedFOLService:
    """
    Wrapper class that provides both the new Gemini service and fallback options
    """
    
    def __init__(self):
        """Initialize Enhanced FOL Service"""
        try:
            # Try to initialize Gemini service
            self.gemini_service = FOLVerificationService()
            self.use_gemini = True
            logger.info("✅ Enhanced FOL Service initialized with Gemini 2.5 Flash")
        except Exception as e:
            logger.warning(f"⚠️ Gemini initialization failed: {e}")
            self.gemini_service = None
            self.use_gemini = False
    
    async def verify_medical_explanation(
        self,
        explanation_text: str,
        patient_data: Dict[str, Any],
        patient_id: str = "default",
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Main verification method with Gemini or fallback
        """
        if self.use_gemini and self.gemini_service:
            # Use Gemini service
            diagnosis = context.get('diagnosis') if context else None
            return await self.gemini_service.verify_medical_explanation(
                explanation_text=explanation_text, 
                patient_data=patient_data, 
                patient_id=patient_id,
                diagnosis=diagnosis,
                context=context
            )
        else:
            # Use fallback basic extraction
            return await self._basic_verification(explanation_text, patient_data)
    
    async def _basic_verification(
        self,
        explanation_text: str,
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Basic verification without Gemini
        """
        # Simple pattern-based extraction
        predicates = []
        text_lower = explanation_text.lower()
        
        # Extract basic predicates
        if 'chest pain' in text_lower:
            predicates.append(FOLPredicate(
                predicate_type='has_symptom',
                subject='patient',
                object='chest_pain',
                confidence=0.7
            ))
        
        if 'hypertension' in text_lower:
            predicates.append(FOLPredicate(
                predicate_type='has_condition',
                subject='patient',
                object='hypertension',
                confidence=0.7
            ))
        
        # Basic verification
        verified_count = 0
        for predicate in predicates:
            if predicate.predicate_type == 'has_symptom':
                symptoms = patient_data.get('symptoms', [])
                if any('chest' in s.lower() or 'pain' in s.lower() for s in symptoms):
                    verified_count += 1
                    predicate.verified = True
        
        success_rate = verified_count / len(predicates) if predicates else 0
        
        return {
            'status': 'BASIC_VERIFICATION',
            'summary': f"Basic verification: {verified_count}/{len(predicates)} predicates verified",
            'total_predicates': len(predicates),
            'verified_predicates': verified_count,
            'success_rate': success_rate,
            'average_confidence': 0.6,
            'predicates': [p.to_dict() for p in predicates],
            'extraction_method': 'basic_patterns'
        }
    
    def extract_medical_predicates(self, clinical_text: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for predicate extraction
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if self.use_gemini and self.gemini_service:
            result = loop.run_until_complete(
                self.gemini_service.extract_fol_predicates(clinical_text)
            )
            
            # Format for compatibility
            predicates = result.get('predicates', [])
            return {
                'predicates': [p.to_fol_string() for p in predicates],
                'entities': self._extract_entities_from_predicates(predicates),
                'logic_rules': [],
                'confidence_scores': {
                    'overall': sum(p.confidence for p in predicates) / len(predicates) if predicates else 0
                },
                'extraction_method': 'gemini_2.5_flash'
            }
        else:
            # Fallback
            return {
                'predicates': [],
                'entities': [],
                'logic_rules': [],
                'confidence_scores': {'overall': 0.0},
                'extraction_method': 'none'
            }
    
    def _extract_entities_from_predicates(self, predicates: List[FOLPredicate]) -> List[str]:
        """Extract entities from predicates"""
        entities = []
        for pred in predicates:
            entity_type = pred.predicate_type.replace('has_', '').replace('takes_', '')
            entities.append(f"{entity_type}:{pred.object}")
        return entities


# Quick test function
async def test_gemini_fol_service():
    """Test the Gemini FOL service"""
    service = FOLVerificationService()
    
    clinical_text = """
    Patient presents with severe chest pain and shortness of breath. 
    History of hypertension and diabetes. Currently taking metformin 500mg 
    and lisinopril 10mg. Blood pressure 160/95, heart rate 110 bpm.
    Troponin elevated at 0.8 ng/ml. ECG shows ST elevation.
    """
    
    patient_data = {
        'symptoms': ['chest pain', 'shortness of breath'],
        'medical_history': ['hypertension', 'diabetes'],
        'current_medications': ['metformin 500mg', 'lisinopril 10mg'],
        'vitals': {
            'blood_pressure': '160/95',
            'heart_rate': 110
        },
        'lab_results': {
            'troponin': 0.8
        }
    }
    
    result = await service.verify_medical_explanation(
        clinical_text, 
        patient_data,
        diagnosis="Myocardial Infarction"
    )
    
    print(json.dumps(result, indent=2))
    return result


def create_fast_patient_data_structure(patient_input):
    """
    Create optimized patient data structure for FOL verification.
    Converts patient input to standardized format for efficient processing.
    
    Args:
        patient_input: Patient data object with symptoms, history, etc.
        
    Returns:
        Dict: Standardized patient data structure for FOL verification
    """
    try:
        patient_data = {}
        
        # Handle symptoms
        if hasattr(patient_input, 'symptoms') and patient_input.symptoms:
            patient_data['symptoms'] = patient_input.symptoms
        elif hasattr(patient_input, 'chief_complaint') and patient_input.chief_complaint:
            patient_data['symptoms'] = [patient_input.chief_complaint]
        else:
            patient_data['symptoms'] = []
        
        # Handle medical history
        if hasattr(patient_input, 'medical_history') and patient_input.medical_history:
            patient_data['medical_history'] = patient_input.medical_history
        elif hasattr(patient_input, 'past_medical_history') and patient_input.past_medical_history:
            patient_data['medical_history'] = patient_input.past_medical_history
        else:
            patient_data['medical_history'] = []
        
        # Handle current medications
        if hasattr(patient_input, 'current_medications') and patient_input.current_medications:
            patient_data['current_medications'] = patient_input.current_medications
        elif hasattr(patient_input, 'medications') and patient_input.medications:
            patient_data['current_medications'] = patient_input.medications
        else:
            patient_data['current_medications'] = []
        
        # Handle vital signs
        vitals = {}
        if hasattr(patient_input, 'vital_signs') and patient_input.vital_signs:
            vitals = patient_input.vital_signs
        else:
            # Try to extract individual vital signs
            if hasattr(patient_input, 'blood_pressure') and patient_input.blood_pressure:
                vitals['blood_pressure'] = patient_input.blood_pressure
            if hasattr(patient_input, 'heart_rate') and patient_input.heart_rate:
                vitals['heart_rate'] = patient_input.heart_rate
            if hasattr(patient_input, 'temperature') and patient_input.temperature:
                vitals['temperature'] = patient_input.temperature
            if hasattr(patient_input, 'respiratory_rate') and patient_input.respiratory_rate:
                vitals['respiratory_rate'] = patient_input.respiratory_rate
        patient_data['vitals'] = vitals
        
        # Handle lab results
        if hasattr(patient_input, 'lab_results') and patient_input.lab_results:
            patient_data['lab_results'] = patient_input.lab_results
        else:
            patient_data['lab_results'] = {}
        
        # Handle imaging results
        if hasattr(patient_input, 'imaging_results') and patient_input.imaging_results:
            patient_data['imaging_results'] = patient_input.imaging_results
        else:
            patient_data['imaging_results'] = {}
        
        # Handle demographics
        demographics = {}
        if hasattr(patient_input, 'age') and patient_input.age:
            demographics['age'] = patient_input.age
        if hasattr(patient_input, 'gender') and patient_input.gender:
            demographics['gender'] = patient_input.gender
        if hasattr(patient_input, 'weight') and patient_input.weight:
            demographics['weight'] = patient_input.weight
        if hasattr(patient_input, 'height') and patient_input.height:
            demographics['height'] = patient_input.height
        patient_data['demographics'] = demographics
        
        # Handle additional fields that might be present
        if hasattr(patient_input, 'allergies') and patient_input.allergies:
            patient_data['allergies'] = patient_input.allergies
        else:
            patient_data['allergies'] = []
            
        if hasattr(patient_input, 'family_history') and patient_input.family_history:
            patient_data['family_history'] = patient_input.family_history
        else:
            patient_data['family_history'] = []
        
        # Add patient ID if available
        if hasattr(patient_input, 'patient_id') and patient_input.patient_id:
            patient_data['patient_id'] = patient_input.patient_id
            
        logger.info(f"✅ Created fast patient data structure with {len(patient_data)} fields")
        return patient_data
        
    except Exception as e:
        logger.error(f"❌ Error creating fast patient data structure: {str(e)}")
        # Fallback to basic structure
        return {
            'symptoms': [],
            'medical_history': [],
            'current_medications': [],
            'vitals': {},
            'lab_results': {},
            'imaging_results': {},
            'demographics': {},
            'allergies': [],
            'family_history': []
        }


# Create alias for backward compatibility with app.py
OptimizedFOLVerificationService = FOLVerificationService

# Run test if executed directly
if __name__ == "__main__":
    asyncio.run(test_gemini_fol_service())