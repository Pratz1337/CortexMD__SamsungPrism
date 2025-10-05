"""
Enhanced FOL Service with Gemini (fixed) and Groq Integration
Optimized for medical diagnosis verification with better predicate matching
"""
import os
import re
import json
import time
import hashlib
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

logger = logging.getLogger(__name__)

@dataclass
class FOLPredicate:
    """FOL Predicate with verification data"""
    predicate_type: str
    subject: str
    object: str
    confidence: float
    verified: bool = False
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""
    temporal_modifier: Optional[str] = None
    severity: Optional[str] = None
    original_text: str = ""

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

    def to_fol_string(self) -> str:
        """Convert to FOL string representation"""
        temporal = f"@{self.temporal_modifier}" if self.temporal_modifier else ""
        severity_prefix = f"{self.severity}_" if self.severity else ""
        return f"{self.predicate_type}({self.subject}, {severity_prefix}{self.object}){temporal}"

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'predicate_type': self.predicate_type,
            'subject': self.subject,
            'object': self.object,
            'confidence': self.confidence,
            'verified': self.verified,
            'evidence': self.evidence,
            'reasoning': self.reasoning,
            'fol_string': self.to_fol_string(),
            'temporal_modifier': self.temporal_modifier,
            'severity': self.severity,
            'original_text': self.original_text
        }

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
    clinical_recommendations: List[str]
    ai_service_used: str = "fallback"

    def to_dict(self) -> Dict:
        return {
            'total_predicates': self.total_predicates,
            'verified_predicates': self.verified_predicates,
            'failed_predicates': self.failed_predicates,
            'success_rate': self.verified_predicates / max(self.total_predicates, 1),
            'overall_confidence': self.overall_confidence,
            'verification_time': round(self.verification_time, 2),
            'confidence_level': self._get_confidence_category(),
            'clinical_assessment': self._get_clinical_assessment(),
            'detailed_results': self.detailed_results,
            'medical_reasoning_summary': self.medical_reasoning_summary,
            'clinical_recommendations': self.clinical_recommendations,
            'ai_service_used': self.ai_service_used
        }

    def _get_confidence_category(self) -> str:
        if self.overall_confidence >= 0.8:
            return "HIGH"
        elif self.overall_confidence >= 0.6:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_clinical_assessment(self) -> str:
        success_rate = self.verified_predicates / max(self.total_predicates, 1)
        if success_rate >= 0.8:
            return "HIGHLY_CONSISTENT"
        elif success_rate >= 0.6:
            return "MOSTLY_CONSISTENT"
        elif success_rate >= 0.4:
            return "PARTIALLY_CONSISTENT"
        else:
            return "INCONSISTENT"

class EnhancedFOLService:
    """
    Enhanced FOL Service with Gemini (fixed), Groq, and fallback extraction
    """
    
    def __init__(self):
        """Initialize Enhanced FOL Service with multiple AI backends"""
        # Initialize Gemini and Groq using ai_key_manager (supports multiple keys)
        self.has_gemini = False
        self.gemini_model = None
        self.has_groq = False
        self.groq_client = None

        try:
            try:
                from utils.ai_key_manager import get_gemini_model, get_groq_client
            except ImportError:
                from ..utils.ai_key_manager import get_gemini_model, get_groq_client
        except Exception:
            get_gemini_model = None
            get_groq_client = None

        if GEMINI_AVAILABLE and get_gemini_model:
            try:
                self.gemini_model = get_gemini_model('gemini-1.5-flash')
                if self.gemini_model:
                    self.has_gemini = True
                    logger.info("✅ Gemini initialized via ai_key_manager")
            except Exception as e:
                logger.warning(f"⚠️ Gemini initialization via ai_key_manager failed: {e}")

        if GROQ_AVAILABLE and get_groq_client:
            try:
                self.groq_client = get_groq_client()
                if self.groq_client:
                    self.has_groq = True
                    logger.info("✅ Groq initialized via ai_key_manager")
            except Exception as e:
                logger.warning(f"⚠️ Groq initialization via ai_key_manager failed: {e}")
        
        # Cache
        self.cache = {}
        self.max_cache_size = 100
        
        # Log available services
        services = []
        if self.has_gemini:
            services.append("Gemini")
        if self.has_groq:
            services.append("Groq")
        services.append("Fallback")
        
        logger.info(f"🚀 FOL Service initialized with: {', '.join(services)}")
    
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
        """
        start_time = time.time()
        
        try:
            logger.info("🔬 Starting comprehensive FOL verification")
            
            # Extract predicates using best available service
            predicates, service_used = await self._extract_predicates_best_effort(explanation_text)
            
            if not predicates:
                logger.warning("⚠️ No predicates extracted")
                return self._create_empty_report(service_used, time.time() - start_time)
            
            # Verify predicates against patient data
            verification_results = await self._verify_predicates_enhanced(predicates, patient_data)
            
            # Calculate metrics
            verified_count = sum(1 for result in verification_results if result.get('verified', False))
            avg_confidence = sum(pred.confidence for pred in predicates) / len(predicates)
            
            # Generate medical reasoning
            medical_reasoning = await self._generate_medical_reasoning_multi_service(
                predicates, verification_results, diagnosis
            )
            
            # Create recommendations
            recommendations = self._generate_clinical_recommendations(
                predicates, verification_results, verified_count, len(predicates)
            )
            
            # Create report
            report = FOLVerificationReport(
                total_predicates=len(predicates),
                verified_predicates=verified_count,
                failed_predicates=len(predicates) - verified_count,
                overall_confidence=avg_confidence,
                verification_time=time.time() - start_time,
                detailed_results=verification_results,
                medical_reasoning_summary=medical_reasoning,
                clinical_recommendations=recommendations,
                ai_service_used=service_used
            )
            
            result = report.to_dict()
            result['predicates'] = [pred.to_dict() for pred in predicates]
            result['status'] = 'SUCCESS'
            result['message'] = f"FOL verification completed using {service_used}"
            
            logger.info(f"✅ FOL verification completed: {verified_count}/{len(predicates)} predicates verified ({verified_count/len(predicates)*100:.1f}%) using {service_used}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ FOL verification failed: {e}")
            error_report = self._create_error_report(str(e), time.time() - start_time)
            return error_report
    
    async def _extract_predicates_best_effort(self, clinical_text: str) -> Tuple[List[FOLPredicate], str]:
        """
        Extract predicates using the best available service
        """
        # Try Gemini first (if available)
        if self.has_gemini:
            try:
                predicates = await self._extract_with_gemini_fixed(clinical_text)
                if predicates:
                    return predicates, "Gemini"
                logger.warning("⚠️ Gemini returned no predicates, trying Groq")
            except Exception as e:
                logger.warning(f"⚠️ Gemini extraction failed: {e}, trying Groq")
        
        # Try Groq as backup
        if self.has_groq:
            try:
                predicates = await self._extract_with_groq(clinical_text)
                if predicates:
                    return predicates, "Groq"
                logger.warning("⚠️ Groq returned no predicates, using fallback")
            except Exception as e:
                logger.warning(f"⚠️ Groq extraction failed: {e}, using fallback")
        
        # Fallback to regex extraction
        predicates = self._extract_with_fallback(clinical_text)
        return predicates, "Fallback"
    
    async def _extract_with_gemini_fixed(self, clinical_text: str) -> List[FOLPredicate]:
        """
        Extract predicates using Gemini with fixed response parsing
        """
        if not self.has_gemini:
            raise Exception("Gemini not available")
        
        prompt = self._create_extraction_prompt(clinical_text)
        
        try:
            response = self.gemini_model.generate_content(prompt)
            
            # Fixed: Use parts instead of text for complex responses
            if response.candidates and response.candidates[0].content.parts:
                json_text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text'):
                        json_text += part.text
                    elif hasattr(part, 'function_call'):
                        # Handle function calls if any
                        continue
                    else:
                        json_text += str(part)
            else:
                # Fallback to simple text access if available
                json_text = response.text if hasattr(response, 'text') else str(response)
            
            # Clean JSON
            json_text = self._clean_json_response(json_text)
            
            # Parse and convert to predicates
            result = json.loads(json_text)
            predicates = self._convert_to_predicates(result, "Gemini")
            
            logger.info(f"✅ Gemini extracted {len(predicates)} predicates")
            return predicates
            
        except Exception as e:
            logger.error(f"❌ Gemini extraction error: {e}")
            raise
    
    async def _extract_with_groq(self, clinical_text: str) -> List[FOLPredicate]:
        """
        Extract predicates using Groq
        """
        if not self.has_groq:
            raise Exception("Groq not available")
        
        prompt = self._create_extraction_prompt(clinical_text)
        
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical AI expert specializing in FOL predicate extraction. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2048,
                top_p=0.8
            )
            
            json_text = completion.choices[0].message.content
            json_text = self._clean_json_response(json_text)
            
            result = json.loads(json_text)
            predicates = self._convert_to_predicates(result, "Groq")
            
            logger.info(f"✅ Groq extracted {len(predicates)} predicates")
            return predicates
            
        except Exception as e:
            logger.error(f"❌ Groq extraction error: {e}")
            raise
    
    def _extract_with_fallback(self, clinical_text: str) -> List[FOLPredicate]:
        """
        Enhanced fallback extraction using regex patterns
        """
        predicates = []
        text_lower = clinical_text.lower()
        
        logger.info("🔄 Using enhanced fallback FOL extraction")
        
        # Enhanced medical patterns with better matching
        patterns = {
            'symptoms': [
                r'(?:patient\s+(?:has|presents\s+with|reports|complains\s+of|experiencing))\s+([^.,;]+?)(?:[.,;]|and|$)',
                r'(chest\s+pain|shortness\s+of\s+breath|dyspnea|fever|headache|nausea|vomiting|dizziness|fatigue|weakness|thigh\s+mass|mass\s+in|abdominal\s+pain|back\s+pain)',
                r'(?:symptoms?\s+include)\s*:?\s*([^.;]+)',
                r'(?:chief\s+complaint|cc)\s*:?\s*([^.;,]+)',
            ],
            'conditions': [
                r'(?:diagnosed\s+with|diagnosis\s+(?:of|:)\s*|history\s+of|has\s+(?:a\s+)?(?:history\s+of\s+)?)\s*([^.,;\n\(]+?)(?:\s*\(|$|[.,;])',
                r'(soft\s+tissue\s+sarcoma|myxofibrosarcoma|sarcoma|hypertension|diabetes|asthma|copd|heart\s+failure|high-grade|cancer|tumor)',
                r'(?:condition|disease|disorder)\s*:?\s*([^.,;]+)',
            ],
            'medications': [
                r'(?:taking|prescribed|on|current\s+medications?)\s*:?\s*([a-zA-Z][a-zA-Z0-9\-\s]{2,30})(?:\s+\d+\s*mg|[.,;]|and|$)',
                r'\b(aspirin|metformin|lisinopril|atorvastatin|insulin|warfarin|prednisone|ibuprofen|acetaminophen)\b',
            ],
            'vitals': [
                r'(?:blood\s+pressure|BP)\s*:?\s*(\d+/\d+)',
                r'(?:heart\s+rate|HR|pulse)\s*:?\s*(\d+)',
                r'(?:temperature|temp)\s*:?\s*(\d+[\.\d]*)',
                r'(?:weight)\s*:?\s*(\d+[\.\d]*\s*(?:kg|lbs)?)',
            ],
            'labs': [
                r'(troponin|glucose|creatinine|hemoglobin|hba1c|cholesterol)\s+(?:is|was|level|value|result)?\s*(elevated|high|low|normal|\d+[\.\d]*)',
                r'(?:lab\s+(?:shows|results))\s+([^.,;]+)',
            ]
        }
        
        # Extract symptoms
        for pattern in patterns['symptoms']:
            matches = re.finditer(pattern, clinical_text, re.IGNORECASE)
            for match in matches:
                symptom = self._clean_medical_term(match.group(1))
                if self._is_valid_medical_term(symptom):
                    predicates.append(FOLPredicate(
                        predicate_type='has_symptom',
                        subject='patient',
                        object=symptom.lower(),
                        confidence=0.75,
                        evidence=[match.group(0)],
                        original_text=match.group(0)
                    ))
        
        # Extract conditions
        for pattern in patterns['conditions']:
            matches = re.finditer(pattern, clinical_text, re.IGNORECASE)
            for match in matches:
                condition = self._clean_medical_term(match.group(1))
                if self._is_valid_medical_term(condition):
                    predicates.append(FOLPredicate(
                        predicate_type='has_condition',
                        subject='patient',
                        object=condition.lower(),
                        confidence=0.85,
                        evidence=[match.group(0)],
                        original_text=match.group(0)
                    ))
        
        # Extract medications
        for pattern in patterns['medications']:
            matches = re.finditer(pattern, clinical_text, re.IGNORECASE)
            for match in matches:
                medication = self._clean_medical_term(match.group(1))
                if self._is_valid_medical_term(medication):
                    predicates.append(FOLPredicate(
                        predicate_type='takes_medication',
                        subject='patient',
                        object=medication.lower(),
                        confidence=0.8,
                        evidence=[match.group(0)],
                        original_text=match.group(0)
                    ))
        
        # Extract vitals
        for pattern in patterns['vitals']:
            matches = re.finditer(pattern, clinical_text, re.IGNORECASE)
            for match in matches:
                vital_type = self._identify_vital_type(pattern)
                vital_value = match.group(1)
                predicates.append(FOLPredicate(
                    predicate_type='has_vital_sign',
                    subject='patient',
                    object=f"{vital_type}:{vital_value}".lower(),
                    confidence=0.9,
                    evidence=[match.group(0)],
                    original_text=match.group(0)
                ))
        
        # Extract lab values
        for pattern in patterns['labs']:
            matches = re.finditer(pattern, clinical_text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    lab_test = match.group(1).strip()
                    lab_value = match.group(2).strip()
                    predicates.append(FOLPredicate(
                        predicate_type='has_lab_value',
                        subject='patient',
                        object=f"{lab_test}:{lab_value}".lower(),
                        confidence=0.85,
                        evidence=[match.group(0)],
                        original_text=match.group(0)
                    ))
        
        # Remove duplicates and filter
        predicates = self._filter_and_deduplicate(predicates)
        
        logger.info(f"📝 Fallback extraction found {len(predicates)} predicates")
        print(f"🔍 DEBUG - Fallback extraction details:")
        print(f"   📝 Input text length: {len(clinical_text)} chars")
        print(f"   📝 Text preview: {clinical_text[:200]}...")
        print(f"   📊 Raw predicates found: {len(predicates)}")
        
        if predicates:
            for i, pred in enumerate(predicates[:3]):  # Show first 3 predicates
                print(f"   🔬 Predicate {i+1}: {pred.predicate_type}({pred.subject}, {pred.object}) - confidence: {pred.confidence}")
        else:
            print("   ⚠️ No predicates found in fallback extraction")
        
        return predicates
    
    async def _verify_predicates_enhanced(
        self,
        predicates: List[FOLPredicate],
        patient_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Enhanced predicate verification with more lenient matching
        """
        verification_results = []
        
        for predicate in predicates:
            verified = False
            supporting_evidence = []
            confidence_bonus = 0
            
            object_text = predicate.object.replace('_', ' ').lower()
            
            # Symptom verification
            if predicate.predicate_type == 'has_symptom':
                verified, evidence = self._verify_symptom_enhanced(object_text, patient_data)
                if verified:
                    supporting_evidence.extend(evidence)
                    confidence_bonus = 0.1
            
            # Condition verification
            elif predicate.predicate_type == 'has_condition':
                verified, evidence = self._verify_condition_enhanced(object_text, patient_data)
                if verified:
                    supporting_evidence.extend(evidence)
                    confidence_bonus = 0.15
            
            # Medication verification
            elif predicate.predicate_type == 'takes_medication':
                verified, evidence = self._verify_medication_enhanced(object_text, patient_data)
                if verified:
                    supporting_evidence.extend(evidence)
                    confidence_bonus = 0.1
            
            # Vital signs verification
            elif predicate.predicate_type == 'has_vital_sign':
                verified, evidence = self._verify_vital_enhanced(object_text, patient_data)
                if verified:
                    supporting_evidence.extend(evidence)
                    confidence_bonus = 0.2
            
            # Lab values verification
            elif predicate.predicate_type == 'has_lab_value':
                verified, evidence = self._verify_lab_enhanced(object_text, patient_data)
                if verified:
                    supporting_evidence.extend(evidence)
                    confidence_bonus = 0.15
            
            # Update predicate
            predicate.verified = verified
            if verified:
                predicate.confidence = min(1.0, predicate.confidence + confidence_bonus)
                predicate.evidence.extend(supporting_evidence)
            
            verification_results.append({
                'predicate': predicate.to_fol_string(),
                'predicate_type': predicate.predicate_type,
                'object': predicate.object,
                'verified': verified,
                'confidence': predicate.confidence,
                'supporting_evidence': supporting_evidence,
                'original_text': predicate.original_text
            })
        
        return verification_results
    
    def _verify_symptom_enhanced(self, symptom: str, patient_data: Dict) -> Tuple[bool, List[str]]:
        """Enhanced symptom verification with fuzzy matching"""
        evidence = []
        
        # Check symptoms field
        if 'symptoms' in patient_data and patient_data['symptoms']:
            for patient_symptom in patient_data['symptoms']:
                if self._fuzzy_match_enhanced(symptom, patient_symptom):
                    evidence.append(f"Patient symptoms: {patient_symptom}")
                    return True, evidence
        
        # Check chief complaint
        if 'chief_complaint' in patient_data and patient_data['chief_complaint']:
            if self._fuzzy_match_enhanced(symptom, patient_data['chief_complaint']):
                evidence.append(f"Chief complaint: {patient_data['chief_complaint']}")
                return True, evidence
        
        # Check present illness
        if 'present_illness' in patient_data and patient_data['present_illness']:
            if self._fuzzy_match_enhanced(symptom, patient_data['present_illness']):
                evidence.append(f"Present illness: {patient_data['present_illness']}")
                return True, evidence
        
        return False, []
    
    def _verify_condition_enhanced(self, condition: str, patient_data: Dict) -> Tuple[bool, List[str]]:
        """Enhanced condition verification"""
        evidence = []
        
        # Check medical history
        if 'medical_history' in patient_data and patient_data['medical_history']:
            for patient_condition in patient_data['medical_history']:
                if self._fuzzy_match_enhanced(condition, patient_condition):
                    evidence.append(f"Medical history: {patient_condition}")
                    return True, evidence
        
        # Check diagnoses
        if 'diagnoses' in patient_data and patient_data['diagnoses']:
            for diagnosis in patient_data['diagnoses']:
                if self._fuzzy_match_enhanced(condition, diagnosis):
                    evidence.append(f"Diagnosis: {diagnosis}")
                    return True, evidence
        
        # Check primary diagnosis
        if 'primary_diagnosis' in patient_data and patient_data['primary_diagnosis']:
            if self._fuzzy_match_enhanced(condition, patient_data['primary_diagnosis']):
                evidence.append(f"Primary diagnosis: {patient_data['primary_diagnosis']}")
                return True, evidence
        
        return False, []
    
    def _verify_medication_enhanced(self, medication: str, patient_data: Dict) -> Tuple[bool, List[str]]:
        """Enhanced medication verification"""
        evidence = []
        
        # Check current medications
        if 'current_medications' in patient_data and patient_data['current_medications']:
            for patient_med in patient_data['current_medications']:
                if self._fuzzy_match_enhanced(medication, patient_med):
                    evidence.append(f"Current medication: {patient_med}")
                    return True, evidence
        
        # Check medications list
        if 'medications' in patient_data and patient_data['medications']:
            for patient_med in patient_data['medications']:
                if self._fuzzy_match_enhanced(medication, patient_med):
                    evidence.append(f"Medication: {patient_med}")
                    return True, evidence
        
        return False, []
    
    def _verify_vital_enhanced(self, vital: str, patient_data: Dict) -> Tuple[bool, List[str]]:
        """Enhanced vital signs verification"""
        evidence = []
        
        if 'vitals' in patient_data or 'vital_signs' in patient_data:
            vitals_data = patient_data.get('vitals', patient_data.get('vital_signs', {}))
            
            # Parse vital type and value
            if ':' in vital:
                vital_type, vital_value = vital.split(':', 1)
                
                # Check if vital exists in data
                for key, value in vitals_data.items():
                    if self._fuzzy_match_enhanced(vital_type, key):
                        evidence.append(f"Vital sign {key}: {value}")
                        return True, evidence
        
        return False, []
    
    def _verify_lab_enhanced(self, lab: str, patient_data: Dict) -> Tuple[bool, List[str]]:
        """Enhanced lab values verification"""
        evidence = []
        
        if 'lab_results' in patient_data or 'labs' in patient_data:
            labs_data = patient_data.get('lab_results', patient_data.get('labs', {}))
            
            # Parse lab test and value
            if ':' in lab:
                lab_test, lab_value = lab.split(':', 1)
                
                # Check if lab exists in data
                for key, value in labs_data.items():
                    if self._fuzzy_match_enhanced(lab_test, key):
                        evidence.append(f"Lab result {key}: {value}")
                        return True, evidence
        
        return False, []
    
    def _fuzzy_match_enhanced(self, text1: str, text2: str, threshold: float = 0.4) -> bool:
        """Enhanced fuzzy matching with medical synonyms"""
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Exact match
        if text1 == text2:
            return True
        
        # Substring match
        if text1 in text2 or text2 in text1:
            return True
        
        # Medical synonyms
        medical_synonyms = {
            'sarcoma': ['soft tissue sarcoma', 'myxofibrosarcoma', 'tumor', 'mass'],
            'chest pain': ['chest discomfort', 'thoracic pain', 'angina'],
            'shortness of breath': ['dyspnea', 'breathlessness', 'difficulty breathing', 'sob'],
            'high-grade': ['high grade', 'aggressive', 'malignant'],
            'thigh mass': ['thigh tumor', 'leg mass', 'mass', 'lump'],
            'hypertension': ['high blood pressure', 'htn', 'elevated bp'],
            'diabetes': ['dm', 'diabetes mellitus', 'high blood sugar'],
            'mass': ['tumor', 'lesion', 'growth', 'lump']
        }
        
        # Check synonyms
        for term, synonyms in medical_synonyms.items():
            if (text1 == term and any(syn in text2 for syn in synonyms)) or \
               (text2 == term and any(syn in text1 for syn in synonyms)):
                return True
        
        # Word overlap
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def _create_extraction_prompt(self, clinical_text: str) -> str:
        """Create standardized extraction prompt"""
        return f"""You are a medical AI expert specializing in First-Order Logic (FOL) predicate extraction for clinical diagnosis verification.

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

EXTRACTION RULES:
- Only extract information explicitly stated or strongly implied
- Use medical terminology accurately
- Include temporal context when available (acute, chronic, recent)
- Normalize values (e.g., "high blood pressure" → "hypertension")
- Include severity qualifiers (mild, moderate, severe)

OUTPUT FORMAT (JSON ONLY):
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
    "clinical_reasoning": "Brief reasoning about the clinical picture"
}}

Respond ONLY with valid JSON. No additional text."""
    
    def _clean_json_response(self, json_text: str) -> str:
        """Clean JSON response from AI services"""
        # Remove markdown formatting
        json_text = re.sub(r'```json?\s*', '', json_text)
        json_text = re.sub(r'```\s*$', '', json_text)
        
        # Remove extra whitespace
        json_text = json_text.strip()
        
        # Find JSON content between braces
        start = json_text.find('{')
        end = json_text.rfind('}')
        
        if start != -1 and end != -1:
            json_text = json_text[start:end+1]
        
        return json_text
    
    def _convert_to_predicates(self, result: Dict, source: str) -> List[FOLPredicate]:
        """Convert AI response to FOL predicates"""
        predicates = []
        
        for pred_data in result.get('predicates', []):
            predicate = FOLPredicate(
                predicate_type=pred_data.get('type', 'unknown'),
                subject=pred_data.get('subject', 'patient'),
                object=pred_data.get('object', ''),
                confidence=float(pred_data.get('confidence', 0.8)),
                evidence=[pred_data.get('evidence', '')],
                reasoning=result.get('clinical_reasoning', ''),
                temporal_modifier=pred_data.get('temporal'),
                severity=pred_data.get('severity'),
                original_text=pred_data.get('evidence', '')
            )
            predicates.append(predicate)
        
        return predicates
    
    def _clean_medical_term(self, term: str) -> str:
        """Clean and standardize medical term"""
        if not term:
            return ""
        
        # Remove extra whitespace and punctuation
        cleaned = ' '.join(term.strip().split())
        
        # Remove common non-medical words at the beginning
        stop_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = cleaned.split()
        while words and words[0].lower() in stop_words:
            words.pop(0)
        
        # Remove trailing punctuation
        result = ' '.join(words).strip()
        result = re.sub(r'[,;\.!?\s]+$', '', result)
        
        return result if len(result) > 2 else ""
    
    def _is_valid_medical_term(self, term: str) -> bool:
        """Check if term is a valid medical term"""
        if not term or len(term) < 3:
            return False
        
        # Check for meaningful medical content
        medical_keywords = {
            'pain', 'ache', 'discomfort', 'difficulty', 'shortness', 'breath', 
            'nausea', 'vomiting', 'fever', 'headache', 'dizziness', 'fatigue',
            'mass', 'tumor', 'sarcoma', 'hypertension', 'diabetes', 'cancer',
            'chest', 'thigh', 'abdomen', 'back', 'leg', 'arm'
        }
        
        term_lower = term.lower()
        return any(keyword in term_lower for keyword in medical_keywords) or len(term.split()) > 1
    
    def _identify_vital_type(self, pattern: str) -> str:
        """Identify vital sign type from regex pattern"""
        pattern_lower = pattern.lower()
        if 'blood pressure' in pattern_lower or 'bp' in pattern_lower:
            return 'blood_pressure'
        elif 'heart rate' in pattern_lower or 'hr' in pattern_lower or 'pulse' in pattern_lower:
            return 'heart_rate'
        elif 'temperature' in pattern_lower or 'temp' in pattern_lower:
            return 'temperature'
        elif 'weight' in pattern_lower:
            return 'weight'
        else:
            return 'vital_sign'
    
    def _filter_and_deduplicate(self, predicates: List[FOLPredicate]) -> List[FOLPredicate]:
        """Filter and deduplicate predicates"""
        seen = set()
        filtered = []
        
        for predicate in predicates:
            # Create a unique key
            key = f"{predicate.predicate_type}:{predicate.object}"
            
            if key not in seen:
                seen.add(key)
                filtered.append(predicate)
        
        # Sort by confidence
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered
    
    async def _generate_medical_reasoning_multi_service(
        self,
        predicates: List[FOLPredicate],
        verification_results: List[Dict],
        diagnosis: Optional[str] = None
    ) -> str:
        """Generate medical reasoning using available AI services"""
        
        if not predicates:
            return "No clinical predicates available for reasoning analysis."
        
        verified_count = sum(1 for result in verification_results if result.get('verified', False))
        total_count = len(predicates)
        
        # Try AI services for reasoning
        if self.has_groq:
            try:
                return await self._generate_reasoning_with_groq(predicates, verification_results, diagnosis)
            except Exception as e:
                logger.warning(f"⚠️ Groq reasoning failed: {e}")
        
        if self.has_gemini:
            try:
                return await self._generate_reasoning_with_gemini(predicates, verification_results, diagnosis)
            except Exception as e:
                logger.warning(f"⚠️ Gemini reasoning failed: {e}")
        
        # Fallback reasoning
        return self._generate_fallback_reasoning(verified_count, total_count, diagnosis)
    
    async def _generate_reasoning_with_groq(
        self,
        predicates: List[FOLPredicate],
        verification_results: List[Dict],
        diagnosis: Optional[str]
    ) -> str:
        """Generate reasoning with Groq"""
        verified_preds = [v for v in verification_results if v.get('verified')][:5]
        
        prompt = f"""Based on the FOL predicate verification, provide a brief clinical reasoning summary:

VERIFIED PREDICATES ({len(verified_preds)}):
{json.dumps(verified_preds, indent=2)}

{f"DIAGNOSIS CONTEXT: {diagnosis}" if diagnosis else ""}

Provide 2-3 sentences explaining:
1. Clinical consistency of findings
2. Support for diagnosis
3. Overall confidence assessment

Keep it concise and medically accurate."""

        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a medical AI providing clinical reasoning. Be concise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ Groq reasoning error: {e}")
            raise
    
    async def _generate_reasoning_with_gemini(
        self,
        predicates: List[FOLPredicate],
        verification_results: List[Dict],
        diagnosis: Optional[str]
    ) -> str:
        """Generate reasoning with Gemini"""
        verified_preds = [v for v in verification_results if v.get('verified')][:5]
        
        prompt = f"""Based on the FOL predicate verification, provide a brief clinical reasoning summary:

VERIFIED PREDICATES ({len(verified_preds)}):
{json.dumps(verified_preds, indent=2)}

{f"DIAGNOSIS CONTEXT: {diagnosis}" if diagnosis else ""}

Provide 2-3 sentences explaining clinical consistency and diagnostic support."""

        try:
            response = self.gemini_model.generate_content(prompt)
            
            # Use fixed response parsing
            if response.candidates and response.candidates[0].content.parts:
                text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text'):
                        text += part.text
            else:
                text = response.text if hasattr(response, 'text') else str(response)
            
            return text.strip()
        except Exception as e:
            logger.error(f"❌ Gemini reasoning error: {e}")
            raise
    
    def _generate_fallback_reasoning(self, verified_count: int, total_count: int, diagnosis: Optional[str]) -> str:
        """Generate fallback reasoning"""
        success_rate = verified_count / max(total_count, 1)
        
        if success_rate >= 0.8:
            consistency = "highly consistent"
        elif success_rate >= 0.6:
            consistency = "mostly consistent"
        elif success_rate >= 0.4:
            consistency = "partially consistent"
        else:
            consistency = "limited consistency"
        
        reasoning = f"Clinical findings show {consistency} verification ({verified_count}/{total_count} predicates verified, {success_rate:.1%} success rate)."
        
        if diagnosis:
            if success_rate >= 0.6:
                reasoning += f" The verified findings provide reasonable support for the diagnosis of {diagnosis}."
            else:
                reasoning += f" Additional clinical evidence may be needed to fully support the diagnosis of {diagnosis}."
        
        return reasoning
    
    def _generate_clinical_recommendations(
        self,
        predicates: List[FOLPredicate],
        verification_results: List[Dict],
        verified_count: int,
        total_count: int
    ) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []
        success_rate = verified_count / max(total_count, 1)
        
        if success_rate < 0.5:
            recommendations.append("Consider additional clinical assessment to verify unconfirmed findings")
            recommendations.append("Review patient history and symptoms for missing information")
        
        if success_rate >= 0.7:
            recommendations.append("Clinical findings are well-documented and consistent")
        
        # Check for specific findings
        has_symptoms = any(p.predicate_type == 'has_symptom' and p.verified for p in predicates)
        has_conditions = any(p.predicate_type == 'has_condition' and p.verified for p in predicates)
        has_vitals = any(p.predicate_type == 'has_vital_sign' and p.verified for p in predicates)
        
        if not has_symptoms:
            recommendations.append("Document patient symptoms more comprehensively")
        
        if not has_vitals:
            recommendations.append("Consider recording vital signs if not documented")
        
        if not recommendations:
            recommendations.append("Clinical documentation appears comprehensive and well-verified")
        
        return recommendations
    
    def _create_empty_report(self, service_used: str, verification_time: float) -> Dict[str, Any]:
        """Create empty report when no predicates found"""
        return {
            'status': 'SUCCESS',
            'message': 'No medical predicates found in the provided text',
            'total_predicates': 0,
            'verified_predicates': 0,
            'failed_predicates': 0,
            'success_rate': 0.0,
            'overall_confidence': 0.0,
            'verification_time': verification_time,
            'confidence_level': 'LOW',
            'clinical_assessment': 'INSUFFICIENT_DATA',
            'detailed_results': [],
            'predicates': [],
            'medical_reasoning_summary': 'No clinical predicates could be extracted from the provided explanation text.',
            'clinical_recommendations': ['Provide more detailed clinical information for better analysis'],
            'ai_service_used': service_used
        }
    
    def _create_error_report(self, error_message: str, verification_time: float) -> Dict[str, Any]:
        """Create error report"""
        return {
            'status': 'ERROR',
            'message': f'FOL verification failed: {error_message}',
            'total_predicates': 0,
            'verified_predicates': 0,
            'failed_predicates': 0,
            'success_rate': 0.0,
            'overall_confidence': 0.0,
            'verification_time': verification_time,
            'confidence_level': 'LOW',
            'clinical_assessment': 'ERROR',
            'detailed_results': [],
            'predicates': [],
            'medical_reasoning_summary': f'Verification failed due to: {error_message}',
            'clinical_recommendations': ['Please try again or contact support'],
            'ai_service_used': 'error'
        }

# Wrapper functions for backward compatibility
class FOLVerificationService(EnhancedFOLService):
    """Backward compatibility wrapper"""
    pass

# Quick test function
async def test_enhanced_fol_service():
    """Test the enhanced FOL service"""
    service = EnhancedFOLService()
    
    clinical_text = """
    62-year-old male patient presents with a large thigh mass and history of soft tissue sarcoma. 
    Patient reports chest pain and shortness of breath. Current medications include aspirin and metformin.
    Vital signs: BP 140/90, HR 95 bpm. Recent CT shows high-grade myxofibrosarcoma.
    """
    
    patient_data = {
        'symptoms': ['chest pain', 'shortness of breath', 'thigh mass'],
        'medical_history': ['soft tissue sarcoma'],
        'current_medications': ['aspirin', 'metformin'],
        'vitals': {
            'blood_pressure': '140/90',
            'heart_rate': 95
        },
        'primary_diagnosis': 'High-grade myxofibrosarcoma'
    }
    
    result = await service.verify_medical_explanation(
        clinical_text, 
        patient_data,
        diagnosis="High-grade myxofibrosarcoma"
    )
    
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    asyncio.run(test_enhanced_fol_service())
