"""
FOL Verification System V2 - Complete Rebuild
A robust, error-free implementation of First-Order Logic verification for medical diagnoses
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import google.generativeai as genai
from groq import Groq

logger = logging.getLogger(__name__)

# Configure API keys
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDuTFCoDcTULjSANmMvQlR_yYYD8WSZerQ')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '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')


@dataclass
class FOLPredicate:
    """Represents a single FOL predicate"""
    predicate_type: str  # has_symptom, has_condition, takes_medication, etc.
    subject: str = "patient"
    object: str = ""
    value: Optional[str] = None
    confidence: float = 0.0
    verified: bool = False
    evidence: List[str] = field(default_factory=list)
    
    def to_string(self) -> str:
        """Convert predicate to FOL string format"""
        if self.value:
            return f"{self.predicate_type}({self.subject}, {self.object}, {self.value})"
        return f"{self.predicate_type}({self.subject}, {self.object})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert predicate to dictionary"""
        return {
            'fol_string': self.to_string(),
            'type': self.predicate_type,
            'subject': self.subject,
            'object': self.object,
            'value': self.value,
            'confidence': self.confidence,
            'verified': self.verified,
            'evidence': self.evidence
        }


@dataclass
class FOLVerificationResult:
    """Complete FOL verification result"""
    total_predicates: int = 0
    verified_predicates: int = 0
    failed_predicates: int = 0
    predicates: List[FOLPredicate] = field(default_factory=list)
    overall_confidence: float = 0.0
    verification_time: float = 0.0
    success_rate: float = 0.0
    confidence_level: str = "LOW"
    clinical_assessment: str = "UNKNOWN"
    medical_reasoning: str = ""
    recommendations: List[str] = field(default_factory=list)
    ai_service_used: str = "unknown"
    status: str = "PENDING"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for API response"""
        return {
            'status': self.status,
            'total_predicates': self.total_predicates,
            'verified_predicates': self.verified_predicates,
            'failed_predicates': self.failed_predicates,
            'success_rate': self.success_rate,
            'overall_confidence': self.overall_confidence,
            'verification_time': self.verification_time,
            'confidence_level': self.confidence_level,
            'clinical_assessment': self.clinical_assessment,
            'medical_reasoning_summary': self.medical_reasoning,
            'clinical_recommendations': self.recommendations,
            'ai_service_used': self.ai_service_used,
            'predicates': [p.to_dict() for p in self.predicates],
            'verified_explanations': self.verified_predicates,
            'total_explanations': self.total_predicates,
            'verification_summary': f'FOL verification: {self.verified_predicates}/{self.total_predicates} predicates verified ({self.success_rate:.1%})',
            'detailed_results': [p.to_dict() for p in self.predicates]
        }


class FOLVerificationV2:
    """
    Rebuilt FOL Verification System with robust error handling
    """
    
    def __init__(self):
        """Initialize FOL verification system with AI services"""
        # Initialize Gemini and Groq, prefer ai_key_manager for key rotation
        self.gemini_available = False
        self.gemini_model = None
        self.groq_available = False
        self.groq_client = None

        try:
            try:
                from utils.ai_key_manager import get_gemini_model, get_groq_client
            except ImportError:
                from ..utils.ai_key_manager import get_gemini_model, get_groq_client
        except Exception:
            get_gemini_model = None
            get_groq_client = None

        if get_gemini_model:
            try:
                self.gemini_model = get_gemini_model('gemini-1.5-flash')
                if self.gemini_model:
                    self.gemini_available = True
                    logger.info("✅ Gemini initialized via ai_key_manager for FOL verification")
            except Exception as e:
                logger.warning(f"Gemini initialization via ai_key_manager failed: {e}")

        if not self.gemini_available:
            try:
                if GEMINI_API_KEY:
                    genai.configure(api_key=GEMINI_API_KEY)
                    self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                    self.gemini_available = True
                    logger.info("✅ Gemini initialized via env for FOL verification")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}")

        if get_groq_client:
            try:
                self.groq_client = get_groq_client()
                if self.groq_client:
                    self.groq_available = True
                    logger.info("✅ Groq initialized via ai_key_manager for FOL verification")
            except Exception as e:
                logger.warning(f"Groq initialization via ai_key_manager failed: {e}")

        if not self.groq_available:
            try:
                if GROQ_API_KEY:
                    self.groq_client = Groq(api_key=GROQ_API_KEY)
                    self.groq_available = True
                    logger.info("✅ Groq initialized via env for FOL verification")
            except Exception as e:
                logger.warning(f"Groq initialization failed: {e}")
    
    async def verify_medical_explanation(
        self,
        explanation_text: str,
        patient_data: Dict[str, Any],
        patient_id: Optional[str] = None,
        diagnosis: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main verification method with XAI reasoning - ENHANCED
        
        This method now includes XAI (Explainable AI) capabilities:
        1. Uses LLM to generate detailed medical reasoning by comparing diagnosis with user input
        2. Extracts FOL predicates from both the reasoning and original text
        3. Verifies predicates against patient data
        4. Provides transparent, explainable verification results
        
        Args:
            explanation_text: Medical explanation text (can be string or list of strings)
            patient_data: Patient data dictionary
            patient_id: Patient identifier
            diagnosis: Primary diagnosis
            context: Additional context
            
        Returns:
            Complete FOL verification result with XAI reasoning as dictionary
        """
        import time
        start_time = time.time()
        
        try:
            # Handle both string and list inputs gracefully
            if isinstance(explanation_text, list):
                # Join list elements into a single string
                explanation_text = ' '.join(str(item) for item in explanation_text)
            elif not isinstance(explanation_text, str):
                # Convert to string if not already
                explanation_text = str(explanation_text)
            
            logger.info(f"🔬 Starting XAI-enhanced FOL verification for patient {patient_id}")
            
            # ===== NEW XAI STEP: Generate Medical Reasoning using LLM =====
            logger.info(f"🧠 Step 1: Generating XAI medical reasoning")
            xai_reasoning = await self._generate_xai_reasoning(
                diagnosis=diagnosis,
                patient_data=patient_data,
                explanation_text=explanation_text,
                context=context
            )
            
            # ===== Step 2: Extract predicates from XAI reasoning AND original text =====
            logger.info(f"📋 Step 2: Extracting FOL predicates from XAI reasoning and original text")
            
            # Combine XAI reasoning with original explanation for predicate extraction
            combined_text = f"{xai_reasoning}\n\n{explanation_text}"
            predicates = await self._extract_predicates(combined_text, diagnosis)
            
            if not predicates:
                logger.warning("No predicates extracted from explanation")
                # Create default result with XAI reasoning even if no predicates
                return self._create_empty_result_with_xai(time.time() - start_time, xai_reasoning)
            
            logger.info(f"📋 Extracted {len(predicates)} predicates from XAI-enhanced text")
            
            # ===== Step 3: Verify each predicate against patient data =====
            logger.info(f"✅ Step 3: Verifying predicates against patient data")
            for predicate in predicates:
                self._verify_predicate(predicate, patient_data)
            
            # ===== Step 4: Calculate verification metrics =====
            verified_count = sum(1 for p in predicates if p.verified)
            failed_count = len(predicates) - verified_count
            success_rate = verified_count / len(predicates) if predicates else 0.0
            avg_confidence = sum(p.confidence for p in predicates) / len(predicates) if predicates else 0.0
            
            # ===== Step 5: Determine confidence level and assessment =====
            confidence_level = self._get_confidence_level(avg_confidence)
            clinical_assessment = self._get_clinical_assessment(success_rate)
            
            # ===== Step 6: Generate medical reasoning summary =====
            medical_reasoning = await self._generate_medical_reasoning(
                predicates, verified_count, len(predicates), diagnosis
            )
            
            # Integrate XAI reasoning with medical reasoning
            integrated_reasoning = f"XAI Analysis: {xai_reasoning}\n\nVerification Summary: {medical_reasoning}"
            
            # ===== Step 7: Generate recommendations =====
            recommendations = self._generate_recommendations(success_rate, predicates)
            
            # Determine which AI service was used
            ai_service = "Gemini" if self.gemini_available else "Groq" if self.groq_available else "Fallback"
            
            # Create enhanced result with XAI reasoning
            result = FOLVerificationResult(
                total_predicates=len(predicates),
                verified_predicates=verified_count,
                failed_predicates=failed_count,
                predicates=predicates,
                overall_confidence=avg_confidence,
                verification_time=time.time() - start_time,
                success_rate=success_rate,
                confidence_level=confidence_level,
                clinical_assessment=clinical_assessment,
                medical_reasoning=integrated_reasoning,
                recommendations=recommendations,
                ai_service_used=ai_service,
                status="COMPLETED"
            )
            
            # Add XAI-specific fields to result
            result_dict = result.to_dict()
            result_dict['xai_reasoning'] = xai_reasoning
            result_dict['xai_enabled'] = True
            
            logger.info(f"✅ XAI-enhanced FOL verification completed: {verified_count}/{len(predicates)} verified ({success_rate:.1%})")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"FOL verification failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_result(str(e), time.time() - start_time)
    
    async def _extract_predicates(self, text: str, diagnosis: Optional[str] = None) -> List[FOLPredicate]:
        """
        Extract FOL predicates from medical text using AI
        """
        predicates = []
        
        # Try Gemini first
        if self.gemini_available:
            try:
                predicates = await self._extract_with_gemini(text, diagnosis)
                if predicates:
                    return predicates
            except Exception as e:
                logger.warning(f"Gemini extraction failed: {e}")
        
        # Try Groq as backup
        if self.groq_available:
            try:
                predicates = await self._extract_with_groq(text, diagnosis)
                if predicates:
                    return predicates
            except Exception as e:
                logger.warning(f"Groq extraction failed: {e}")
        
        # Fallback to regex extraction
        return self._extract_with_regex(text)
    
    async def _extract_with_gemini(self, text: str, diagnosis: Optional[str] = None) -> List[FOLPredicate]:
        """Extract predicates using Gemini"""
        prompt = f"""
        Extract medical FOL predicates from this clinical text.
        
        Text: {text}
        {f"Diagnosis: {diagnosis}" if diagnosis else ""}
        
        Extract predicates in these categories:
        1. has_symptom(patient, symptom_name) - for symptoms
        2. has_condition(patient, condition_name) - for conditions/diseases
        3. takes_medication(patient, medication_name) - for medications
        4. has_vital_sign(patient, vital_type, value) - for vital signs
        5. has_lab_value(patient, lab_test, value) - for lab results
        
        Return ONLY a JSON array of predicates:
        [
            {{"type": "has_symptom", "object": "chest pain", "confidence": 0.9}},
            {{"type": "has_condition", "object": "myxofibrosarcoma", "confidence": 0.95}}
        ]
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            json_text = response.text
            
            # Clean JSON response
            json_text = self._clean_json(json_text)
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Convert to predicates
            predicates = []
            for item in data:
                if isinstance(item, dict):
                    pred = FOLPredicate(
                        predicate_type=item.get('type', 'unknown'),
                        object=item.get('object', ''),
                        value=item.get('value'),
                        confidence=float(item.get('confidence', 0.8))
                    )
                    predicates.append(pred)
            
            return predicates
            
        except Exception as e:
            logger.error(f"Gemini extraction error: {e}")
            return []
    
    async def _extract_with_groq(self, text: str, diagnosis: Optional[str] = None) -> List[FOLPredicate]:
        """Extract predicates using Groq"""
        prompt = f"""
        Extract medical FOL predicates from this clinical text.
        
        Text: {text}
        {f"Diagnosis: {diagnosis}" if diagnosis else ""}
        
        Return a JSON array of medical predicates with type, object, and confidence.
        Types: has_symptom, has_condition, takes_medication, has_vital_sign, has_lab_value
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Extract medical FOL predicates as JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            json_text = response.choices[0].message.content
            json_text = self._clean_json(json_text)
            
            data = json.loads(json_text)
            
            predicates = []
            for item in data:
                if isinstance(item, dict):
                    pred = FOLPredicate(
                        predicate_type=item.get('type', 'unknown'),
                        object=item.get('object', ''),
                        value=item.get('value'),
                        confidence=float(item.get('confidence', 0.8))
                    )
                    predicates.append(pred)
            
            return predicates
            
        except Exception as e:
            logger.error(f"Groq extraction error: {e}")
            return []
    
    def _extract_with_regex(self, text: str) -> List[FOLPredicate]:
        """Fallback regex-based extraction"""
        predicates = []
        text_lower = text.lower()
        
        # Extract symptoms
        symptom_patterns = [
            r'\b(pain|ache|discomfort|tenderness)\b',
            r'\b(fever|chills|sweating)\b',
            r'\b(cough|shortness of breath|dyspnea)\b',
            r'\b(nausea|vomiting|dizziness)\b',
            r'\b(fatigue|weakness|malaise)\b',
            r'\b(mass|lump|swelling|lesion)\b'
        ]
        
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                symptom = match.group(0)
                predicates.append(FOLPredicate(
                    predicate_type='has_symptom',
                    object=symptom,
                    confidence=0.7
                ))
        
        # Extract conditions
        condition_patterns = [
            r'\b(sarcoma|carcinoma|cancer|tumor|neoplasm)\b',
            r'\b(myxofibrosarcoma|liposarcoma|fibrosarcoma)\b',
            r'\b(diabetes|hypertension|asthma|copd)\b',
            r'\b(infection|inflammation|disease|syndrome)\b'
        ]
        
        for pattern in condition_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                condition = match.group(0)
                predicates.append(FOLPredicate(
                    predicate_type='has_condition',
                    object=condition,
                    confidence=0.75
                ))
        
        # Extract medications
        med_patterns = [
            r'\b(aspirin|ibuprofen|acetaminophen|paracetamol)\b',
            r'\b(metformin|insulin|glipizide)\b',
            r'\b(lisinopril|atenolol|amlodipine)\b',
            r'\b(chemotherapy|radiation|therapy)\b'
        ]
        
        for pattern in med_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                med = match.group(0)
                predicates.append(FOLPredicate(
                    predicate_type='takes_medication',
                    object=med,
                    confidence=0.7
                ))
        
        # Remove duplicates
        seen = set()
        unique_predicates = []
        for pred in predicates:
            key = f"{pred.predicate_type}:{pred.object}"
            if key not in seen:
                seen.add(key)
                unique_predicates.append(pred)
        
        return unique_predicates
    
    def _verify_predicate(self, predicate: FOLPredicate, patient_data: Dict[str, Any]) -> None:
        """
        Verify a predicate against patient data
        """
        predicate.verified = False
        predicate.evidence = []
        
        # Clean the object for matching
        obj_lower = predicate.object.lower().strip()
        
        if predicate.predicate_type == 'has_symptom':
            # Check symptoms
            symptoms = patient_data.get('symptoms', [])
            for symptom in symptoms:
                if self._fuzzy_match(obj_lower, str(symptom).lower()):
                    predicate.verified = True
                    predicate.evidence.append(f"Found in symptoms: {symptom}")
                    predicate.confidence = min(1.0, predicate.confidence + 0.1)
                    return
            
            # Check in chief complaint
            chief_complaint = patient_data.get('chief_complaint', '')
            if chief_complaint and obj_lower in chief_complaint.lower():
                predicate.verified = True
                predicate.evidence.append("Found in chief complaint")
                predicate.confidence = min(1.0, predicate.confidence + 0.05)
                return
        
        elif predicate.predicate_type == 'has_condition':
            # Check medical history
            history = patient_data.get('medical_history', [])
            for condition in history:
                if self._fuzzy_match(obj_lower, str(condition).lower()):
                    predicate.verified = True
                    predicate.evidence.append(f"Found in medical history: {condition}")
                    predicate.confidence = min(1.0, predicate.confidence + 0.15)
                    return
            
            # Check diagnoses
            diagnoses = patient_data.get('diagnoses', [])
            for diagnosis in diagnoses:
                if self._fuzzy_match(obj_lower, str(diagnosis).lower()):
                    predicate.verified = True
                    predicate.evidence.append(f"Found in diagnoses: {diagnosis}")
                    predicate.confidence = min(1.0, predicate.confidence + 0.2)
                    return
            
            # Check primary diagnosis
            primary = patient_data.get('primary_diagnosis', '')
            if primary and self._fuzzy_match(obj_lower, primary.lower()):
                predicate.verified = True
                predicate.evidence.append("Matches primary diagnosis")
                predicate.confidence = min(1.0, predicate.confidence + 0.25)
                return
        
        elif predicate.predicate_type == 'takes_medication':
            # Check medications
            medications = patient_data.get('current_medications', [])
            for med in medications:
                if self._fuzzy_match(obj_lower, str(med).lower()):
                    predicate.verified = True
                    predicate.evidence.append(f"Found in medications: {med}")
                    predicate.confidence = min(1.0, predicate.confidence + 0.1)
                    return
        
        elif predicate.predicate_type == 'has_vital_sign':
            # Check vitals
            vitals = patient_data.get('vitals', {})
            if obj_lower in str(vitals).lower():
                predicate.verified = True
                predicate.evidence.append(f"Found in vitals")
                predicate.confidence = min(1.0, predicate.confidence + 0.15)
                return
        
        elif predicate.predicate_type == 'has_lab_value':
            # Check lab results
            labs = patient_data.get('lab_results', {})
            if obj_lower in str(labs).lower():
                predicate.verified = True
                predicate.evidence.append(f"Found in lab results")
                predicate.confidence = min(1.0, predicate.confidence + 0.15)
                return
        
        # If not verified, add reason
        if not predicate.verified:
            predicate.evidence.append(f"Not found in patient data")
    
    def _fuzzy_match(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Fuzzy string matching for medical terms
        """
        # Exact match
        if text1 == text2:
            return True
        
        # Substring match
        if text1 in text2 or text2 in text1:
            return True
        
        # Word overlap for multi-word terms
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if words1 and words2:
            overlap = len(words1 & words2)
            union = len(words1 | words2)
            if union > 0 and (overlap / union) >= threshold:
                return True
        
        # Check medical synonyms
        synonyms = {
            'chest pain': ['angina', 'thoracic pain', 'chest discomfort'],
            'shortness of breath': ['dyspnea', 'sob', 'breathing difficulty'],
            'mass': ['tumor', 'lump', 'lesion', 'growth'],
            'sarcoma': ['soft tissue sarcoma', 'cancer', 'malignancy'],
            'myxofibrosarcoma': ['mfs', 'myxoid fibrosarcoma'],
            'liposarcoma': ['dedifferentiated liposarcoma', 'fatty tumor']
        }
        
        for term, syns in synonyms.items():
            if text1 == term:
                if any(syn in text2 for syn in syns):
                    return True
            if text2 == term:
                if any(syn in text1 for syn in syns):
                    return True
        
        return False
    
    async def _generate_medical_reasoning(
        self, 
        predicates: List[FOLPredicate], 
        verified_count: int, 
        total_count: int,
        diagnosis: Optional[str] = None
    ) -> str:
        """Generate medical reasoning summary"""
        success_rate = verified_count / total_count if total_count > 0 else 0.0
        
        if success_rate >= 0.8:
            consistency = "highly consistent"
            assessment = "strong support"
        elif success_rate >= 0.6:
            consistency = "mostly consistent"
            assessment = "moderate support"
        elif success_rate >= 0.4:
            consistency = "partially consistent"
            assessment = "limited support"
        else:
            consistency = "limited consistency"
            assessment = "weak support"
        
        reasoning = f"Clinical findings show {consistency} with patient data "
        reasoning += f"({verified_count}/{total_count} predicates verified, {success_rate:.1%} success rate). "
        
        if diagnosis:
            reasoning += f"The verified findings provide {assessment} for the diagnosis of {diagnosis}. "
        
        # Add specific findings
        verified_symptoms = [p for p in predicates if p.verified and p.predicate_type == 'has_symptom']
        if verified_symptoms:
            symptoms = ', '.join(p.object for p in verified_symptoms[:3])
            reasoning += f"Key symptoms confirmed: {symptoms}. "
        
        verified_conditions = [p for p in predicates if p.verified and p.predicate_type == 'has_condition']
        if verified_conditions:
            conditions = ', '.join(p.object for p in verified_conditions[:2])
            reasoning += f"Medical conditions verified: {conditions}. "
        
        return reasoning
    
    def _generate_recommendations(self, success_rate: float, predicates: List[FOLPredicate]) -> List[str]:
        """Generate clinical recommendations based on verification results"""
        recommendations = []
        
        if success_rate < 0.5:
            recommendations.append("Consider additional clinical assessment to verify unconfirmed findings")
            recommendations.append("Review patient history for missing information")
        
        if success_rate >= 0.7:
            recommendations.append("Clinical findings are well-documented and consistent with diagnosis")
        
        # Check for missing data types
        has_symptoms = any(p.predicate_type == 'has_symptom' and p.verified for p in predicates)
        has_vitals = any(p.predicate_type == 'has_vital_sign' for p in predicates)
        has_labs = any(p.predicate_type == 'has_lab_value' for p in predicates)
        
        if not has_symptoms:
            recommendations.append("Document patient symptoms comprehensively")
        
        if not has_vitals:
            recommendations.append("Record vital signs if not already documented")
        
        if not has_labs:
            recommendations.append("Consider laboratory tests for additional diagnostic support")
        
        if not recommendations:
            recommendations.append("Continue with current clinical management plan")
        
        return recommendations[:4]  # Limit to 4 recommendations
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Determine confidence level category"""
        if confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_clinical_assessment(self, success_rate: float) -> str:
        """Determine clinical assessment based on success rate"""
        if success_rate >= 0.8:
            return "HIGHLY_CONSISTENT"
        elif success_rate >= 0.6:
            return "MOSTLY_CONSISTENT"
        elif success_rate >= 0.4:
            return "PARTIALLY_CONSISTENT"
        else:
            return "INCONSISTENT"
    
    def _clean_json(self, text: str) -> str:
        """Clean JSON response from AI"""
        # Remove markdown formatting
        text = re.sub(r'```json?\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        # Find JSON array or object
        start = text.find('[')
        if start == -1:
            start = text.find('{')
        
        if start != -1:
            end = text.rfind(']')
            if end == -1:
                end = text.rfind('}')
            
            if end != -1:
                text = text[start:end+1]
        
        return text.strip()
    
    def _create_empty_result(self, verification_time: float) -> Dict[str, Any]:
        """Create empty result when no predicates found"""
        result = FOLVerificationResult(
            verification_time=verification_time,
            status="NO_PREDICATES",
            medical_reasoning="No predicates could be extracted from the explanation",
            recommendations=["Provide more detailed clinical information"]
        )
        return result.to_dict()
    
    def _create_empty_result_with_xai(self, verification_time: float, xai_reasoning: str) -> Dict[str, Any]:
        """Create empty result when no predicates found but include XAI reasoning"""
        result = FOLVerificationResult(
            verification_time=verification_time,
            status="NO_PREDICATES",
            medical_reasoning=f"XAI Analysis: {xai_reasoning}\n\nNo predicates could be extracted from the explanation",
            recommendations=["Provide more detailed clinical information"]
        )
        result_dict = result.to_dict()
        result_dict['xai_reasoning'] = xai_reasoning
        result_dict['xai_enabled'] = True
        return result_dict
    
    async def _generate_xai_reasoning(
        self,
        diagnosis: Optional[str],
        patient_data: Dict[str, Any],
        explanation_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate XAI (Explainable AI) reasoning by verifying diagnosis against user input
        
        This is the core XAI component that:
        1. Takes the diagnosis output from the AI model
        2. Compares it with the original patient input/data
        3. Generates transparent reasoning about how the diagnosis relates to the input
        4. Identifies supporting and contradicting evidence
        5. Provides explainable justification for the diagnosis
        
        Args:
            diagnosis: Primary diagnosis from AI model
            patient_data: Original patient data/input
            explanation_text: Clinical explanation text
            context: Additional context
            
        Returns:
            Detailed medical reasoning explaining how diagnosis relates to patient input
        """
        logger.info("🧠 Generating XAI medical reasoning to verify diagnosis against patient input")
        
        # Prepare patient presentation summary
        symptoms_str = ', '.join(patient_data.get('symptoms', [])) if patient_data.get('symptoms') else "None documented"
        history_str = ', '.join(patient_data.get('medical_history', [])) if patient_data.get('medical_history') else "None documented"
        medications_str = ', '.join(patient_data.get('current_medications', [])) if patient_data.get('current_medications') else "None documented"
        chief_complaint = patient_data.get('chief_complaint', 'Not specified')
        vitals = patient_data.get('vitals', {})
        
        # Build comprehensive prompt for XAI reasoning
        xai_prompt = f"""
You are an expert medical AI providing transparent, explainable reasoning for a diagnosis.

TASK: Analyze how the diagnosis relates to the patient's input data and generate transparent reasoning.

DIAGNOSIS OUTPUT: {diagnosis or 'Not specified'}

PATIENT INPUT DATA:
- Chief Complaint: {chief_complaint}
- Reported Symptoms: {symptoms_str}
- Medical History: {history_str}
- Current Medications: {medications_str}
- Vital Signs: {vitals if vitals else 'Not documented'}

CLINICAL CONTEXT: {explanation_text[:500]}

Generate a detailed XAI (Explainable AI) analysis that:
1. **Verification**: Explicitly verify how the diagnosis matches the patient's reported symptoms and input
2. **Supporting Evidence**: List specific patient data points that SUPPORT the diagnosis
3. **Contradicting Evidence**: List any patient data that CONTRADICTS or raises doubts about the diagnosis
4. **Missing Information**: Identify critical information that is missing but needed to confirm the diagnosis
5. **Clinical Reasoning**: Explain the medical logic connecting patient input to the diagnosis
6. **Confidence Assessment**: State confidence level (HIGH/MEDIUM/LOW) based on available patient data

Format your response as clear, transparent medical reasoning in 3-5 sentences.
Focus on being explainable - a patient or clinician should understand WHY this diagnosis was given.
"""
        
        try:
            # Try Gemini first for XAI reasoning
            if self.gemini_available:
                try:
                    response = self.gemini_model.generate_content(xai_prompt)
                    xai_reasoning = response.text.strip()
                    logger.info(f"✅ XAI reasoning generated using Gemini")
                    return xai_reasoning
                except Exception as e:
                    logger.warning(f"Gemini XAI reasoning failed: {e}")
            
            # Try Groq as backup
            if self.groq_available:
                try:
                    response = self.groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "You are an expert medical AI providing transparent, explainable reasoning for diagnoses."},
                            {"role": "user", "content": xai_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=800
                    )
                    xai_reasoning = response.choices[0].message.content.strip()
                    logger.info(f"✅ XAI reasoning generated using Groq")
                    return xai_reasoning
                except Exception as e:
                    logger.warning(f"Groq XAI reasoning failed: {e}")
            
            # Fallback: Generate basic reasoning without AI
            logger.warning("Using fallback XAI reasoning generation")
            return self._generate_fallback_xai_reasoning(diagnosis, patient_data)
            
        except Exception as e:
            logger.error(f"XAI reasoning generation failed: {e}")
            return self._generate_fallback_xai_reasoning(diagnosis, patient_data)
    
    def _generate_fallback_xai_reasoning(
        self,
        diagnosis: Optional[str],
        patient_data: Dict[str, Any]
    ) -> str:
        """Generate basic XAI reasoning when AI services are unavailable"""
        symptoms = patient_data.get('symptoms', [])
        history = patient_data.get('medical_history', [])
        
        reasoning = f"XAI Verification Analysis: The diagnosis of '{diagnosis or 'unknown condition'}' "
        
        if symptoms:
            reasoning += f"is being evaluated based on {len(symptoms)} reported symptoms ({', '.join(symptoms[:3])}). "
        else:
            reasoning += "has limited symptom data for verification. "
        
        if history:
            reasoning += f"Patient's medical history includes {len(history)} documented conditions. "
        else:
            reasoning += "Patient medical history is not documented. "
        
        reasoning += "This preliminary assessment requires clinical validation with additional diagnostic data. "
        reasoning += "Confidence level: MEDIUM based on available patient input data."
        
        return reasoning
    
    def _create_error_result(self, error: str, verification_time: float) -> Dict[str, Any]:
        """Create error result"""
        result = FOLVerificationResult(
            verification_time=verification_time,
            status="ERROR",
            medical_reasoning=f"Verification failed: {error}",
            recommendations=["Please retry the verification"]
        )
        return result.to_dict()


# Global instance for easy access
fol_verifier_v2 = FOLVerificationV2()


async def verify_medical_explanation_v2(
    explanation_text: Any,
    patient_data: Dict[str, Any],
    patient_id: Optional[str] = None,
    diagnosis: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main entry point for FOL verification V2
    Handles any input type gracefully
    """
    return await fol_verifier_v2.verify_medical_explanation(
        explanation_text=explanation_text,
        patient_data=patient_data,
        patient_id=patient_id,
        diagnosis=diagnosis,
        context=context
    )
