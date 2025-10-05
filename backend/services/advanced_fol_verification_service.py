"""
Advanced FOL Verification Service
Enhanced deterministic FOL verification system with NLP predicate extraction
"""

import asyncio
import time
import logging
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import os

from services.fol_logic_engine import DeterministicFOLVerifier
from services.ontology_mapper import OntologyMapper
from services.enhanced_fol_verifier_neo4j import EnhancedFOLVerifierWithNeo4j

logger = logging.getLogger(__name__)

@dataclass
class AdvancedFOLVerificationReport:
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

class AdvancedFOLVerificationService:
    """Enhanced deterministic FOL verification service with NLP predicate extraction"""

    def __init__(self):
        logger.info("Initializing Enhanced Advanced FOL Verification Service")

        # Check if Neo4j is enabled
        neo4j_enabled = os.getenv('NEO4J_ENABLED', 'false').lower() == 'true'
        
        # Initialize appropriate FOL verifier based on Neo4j availability
        if neo4j_enabled:
            try:
                self.fol_verifier = EnhancedFOLVerifierWithNeo4j()
                logger.info("✅ Using Enhanced FOL Verifier with Neo4j integration")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Neo4j verifier: {e}")
                logger.info("Falling back to deterministic FOL verifier")
                self.fol_verifier = DeterministicFOLVerifier()
        else:
            self.fol_verifier = DeterministicFOLVerifier()
            logger.info("Using deterministic FOL verifier (Neo4j disabled)")
        
        self.ontology_mapper = OntologyMapper()

        # Medical reasoning engine
        self.medical_reasoner = MedicalReasoningEngine()

        # Simple in-memory cache (in production, use Redis)
        self._cache = {}
        self._cache_ttl = 1800  # 30 minutes

    async def verify_medical_explanation(
        self,
        explanation_text: str,
        patient_data: Dict,
        patient_id: str,
        context: Optional[Dict] = None
    ) -> AdvancedFOLVerificationReport:
        """
        Enhanced deterministic FOL verification pipeline with NLP extraction

        Args:
            explanation_text: AI-generated medical explanation
            patient_data: Comprehensive patient data
            patient_id: Patient identifier
            context: Additional context (imaging, genetics, etc.)

        Returns:
            Comprehensive verification report with medical reasoning
        """
        start_time = time.time()

        logger.info(f"Starting enhanced advanced FOL verification for patient {patient_id}")
        logger.info(f"Explanation: {explanation_text[:200]}...")

        # Check cache first
        cache_key = self._generate_cache_key(explanation_text, patient_data, patient_id)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.info("Returning cached advanced verification result")
            return cached_result

        try:
            # Step 1: Extract FOL predicates using enhanced NLP techniques
            logger.info("Step 1: Extracting FOL predicates with NLP techniques...")
            predicate_strings = await self._extract_predicates_from_text(explanation_text)
            
            # Ensure all predicates are strings, not lists
            cleaned_predicates = []
            for pred in predicate_strings:
                if isinstance(pred, list):
                    # If it's a list, join its elements or take the first one
                    cleaned_predicates.append(str(pred[0]) if pred else "")
                elif isinstance(pred, str):
                    cleaned_predicates.append(pred)
                else:
                    cleaned_predicates.append(str(pred))
            predicate_strings = [p for p in cleaned_predicates if p]  # Remove empty strings
            
            logger.info(f"Extracted {len(predicate_strings)} predicate strings")

            if not predicate_strings:
                return self._create_empty_report(time.time() - start_time)

            # Step 2: Verify predicates using deterministic engine
            logger.info("Step 2: Verifying predicates with deterministic logic engine...")
            verification_tasks = [
                self._verify_single_predicate_deterministic(predicate_str, patient_data)
                for predicate_str in predicate_strings
            ]

            verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)

            # Handle any exceptions and convert to verification results
            processed_results = []
            for i, result in enumerate(verification_results):
                if isinstance(result, Exception):
                    logger.error(f"Verification failed for predicate {predicate_strings[i]}: {result}")
                    processed_results.append({
                        "predicate": predicate_strings[i],
                        "verified": False,
                        "confidence_score": 0.0,
                        "evaluation_method": "error",
                        "error": str(result),
                        "medical_reasoning": "Verification error occurred"
                    })
                else:
                    processed_results.append(result)

            # Step 3: Medical reasoning and disease probability analysis
            logger.info("Step 3: Generating medical reasoning and disease probabilities...")
            disease_probabilities = await self.medical_reasoner.analyze_disease_probabilities(
                predicate_strings, processed_results, patient_data
            )

            # Step 4: Generate clinical recommendations
            logger.info("Step 4: Generating clinical recommendations...")
            recommendations = await self.medical_reasoner.generate_clinical_recommendations(
                disease_probabilities, patient_data, processed_results
            )

            # Step 5: Create comprehensive report
            verification_time = time.time() - start_time
            report = self._generate_advanced_report(
                predicate_strings,
                processed_results,
                disease_probabilities,
                recommendations,
                verification_time
            )

            # Cache results
            self._cache_result(cache_key, report)

            logger.info(f"Advanced verification completed in {verification_time:.2f}s")
            logger.info(f"Overall confidence: {report.overall_confidence:.2f}")
            logger.info(f"Success rate: {report.verified_predicates}/{report.total_predicates}")

            return report

        except Exception as e:
            logger.error(f"Advanced verification failed: {str(e)}")
            return await self._create_error_report(str(e), time.time() - start_time)

    async def _extract_predicates_from_text(self, explanation_text: str) -> List[str]:
        """Conservative FOL predicate extraction using only well-defined medical terms"""
        predicates = []

        # Step 1: Preprocess text
        processed_text = self._preprocess_medical_text(explanation_text)

        # Step 2: Extract only well-defined medical terms using strict pattern matching
        extracted_terms = self._extract_with_patterns(processed_text)

        # Step 3: Normalize terms using OntologyMapper (async call)
        normalized_terms = await self._normalize_terms_async(extracted_terms)

        # Step 4: Convert to FOL predicates using simple rules
        for term_info in normalized_terms:
            logger.debug(f"Processing term_info: {term_info} (type: {type(term_info)})")
            if isinstance(term_info, dict):  # OntologyMapper result
                term = term_info.get('preferred_name', '').lower()
                semantic_type = term_info.get('semantic_types', ['Unknown'])[0]
                logger.debug(f"OntologyMapper result for '{term}': semantic_type='{semantic_type}'")

                # Override OntologyMapper result for known medical conditions
                # that are incorrectly classified as "Medical Concept"
                if semantic_type == 'Medical Concept' and term in self._get_known_conditions():
                    semantic_type = 'Disease or Syndrome'
                    logger.debug(f"Overriding OntologyMapper result: '{term}' -> 'Disease or Syndrome'")
            else:  # Fallback string
                term = term_info.lower()
                logger.debug(f"Using fallback for term: '{term}'")
                semantic_type = self._infer_semantic_type_simple(term)
                logger.debug(f"Fallback inference for '{term}': semantic_type='{semantic_type}'")

            # Convert to FOL predicate based on simple semantic mapping
            predicate = self._convert_term_to_predicate_simple(term, semantic_type)
            logger.debug(f"Created predicate: '{predicate}' for term '{term}' with type '{semantic_type}'")
            if predicate and predicate not in predicates:
                predicates.append(predicate)

        return predicates

    async def _normalize_terms_async(self, extracted_terms: List[str]) -> List:
        """Asynchronously normalize medical terms using OntologyMapper"""
        normalized_terms = []

        for term in extracted_terms:
            if len(term.strip()) > 2:
                try:
                    # Use OntologyMapper asynchronously
                    normalized = await self.ontology_mapper.normalize_medical_term(term.strip())
                    if normalized:
                        # Convert MedicalConcept to dict for easier processing
                        normalized_terms.append({
                            'preferred_name': normalized.preferred_name,
                            'semantic_types': normalized.semantic_types,
                            'cui': normalized.cui,
                            'original_term': term.strip()
                        })
                    else:
                        # Fallback to original term if not found in ontology
                        normalized_terms.append(term.strip())
                except Exception as e:
                    logger.warning(f"Ontology mapping failed for term '{term}': {e}")
                    normalized_terms.append(term.strip())

        return normalized_terms

    def _preprocess_medical_text(self, text: str) -> str:
        """Preprocess medical text for better term extraction"""
        # Convert to lowercase for consistent processing
        processed = text.lower()

        # Normalize whitespace
        processed = re.sub(r'\s+', ' ', processed)

        # Handle medical abbreviations
        abbreviations = {
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'temp': 'temperature',
            'rr': 'respiratory rate',
            'sob': 'shortness of breath',
            'cp': 'chest pain',
            'htn': 'hypertension',
            'dm': 'diabetes mellitus',
            'mi': 'myocardial infarction',
            'cad': 'coronary artery disease',
            'chf': 'congestive heart failure'
        }

        for abbr, full in abbreviations.items():
            processed = re.sub(r'\b' + re.escape(abbr) + r'\b', full, processed)

        return processed.strip()

    def _extract_with_patterns(self, text: str) -> List[str]:
        """Extract terms using conservative medical pattern matching"""
        terms = []

        # Highly conservative approach - only extract terms that are clearly medical
        # and avoid false positives by being very specific

        # Step 1: Extract only well-defined medical terms using strict patterns
        strict_medical_patterns = [
            # Specific conditions with strict boundaries - EXPANDED to include cancers and sarcomas
            r'\b(?:urolithiasis|nephrolithiasis|myocardial infarction|pneumonia|diabetes mellitus|hypertension|asthma|copd|bronchitis)\b',
            r'\b(?:sarcoma|liposarcoma|dedifferentiated liposarcoma|myxofibrosarcoma|cancer|carcinoma|tumor|neoplasm|malignancy)\b',
            r'\b(?:soft tissue sarcoma|bone sarcoma|undifferentiated pleomorphic sarcoma)\b',

            # Specific symptoms with strict boundaries
            r'\b(?:flank pain|chest pain|abdominal pain|back pain|nausea|vomiting|hematuria|shortness of breath|dyspnea|fever|cough|fatigue|dizziness|headache)\b',
            r'\b(?:mass|swelling|lump|lesion|pain|weakness|weight loss|night sweats)\b',

            # Specific lab tests
            r'\b(?:troponin|glucose|creatinine|hemoglobin|hba1c|bilirubin|albumin)\b',

            # Specific procedures
            r'\b(?:ct scan|mri|ultrasound|x-ray|echocardiogram|ekg|ecg|biopsy)\b',
            r'\b(?:chemotherapy|radiation therapy|surgery|resection|excision)\b'
        ]

        # Extract terms using strict patterns
        for pattern in strict_medical_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(0).strip()
                # Only add if it's a reasonable length and not a common word
                if 4 <= len(term) <= 50 and term.lower() not in ['this', 'that', 'with', 'from']:
                    terms.append(term)

        # Step 2: Extract medication names using conservative patterns
        medication_patterns = [
            r'\b(?:aspirin|lisinopril|metformin|insulin|amoxicillin|azithromycin|atorvastatin|amlodipine|hydrochlorothiazide|warfarin|heparin)\b'
        ]

        for pattern in medication_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(0).strip()
                if term not in terms:
                    terms.append(term)

        # Step 3: Remove any remaining suspicious terms
        filtered_terms = []
        suspicious_patterns = [
            r'^(?:response|question|doctor|consultation|follow)$',  # Only exclude if it's JUST these words
            r'^(?:the|and|for|with|this|that|from|have|been|were|will|can|may)$',  # Only exact matches of common words
            r'^\w{1,2}$'  # Only 1-2 letter words
        ]

        for term in terms:
            is_suspicious = False
            for pattern in suspicious_patterns:
                if re.match(pattern, term, re.IGNORECASE):
                    is_suspicious = True
                    break
            if not is_suspicious:
                filtered_terms.append(term)

        return filtered_terms

    def _infer_semantic_type_simple(self, term: str) -> str:
        """Simple semantic type inference based on term characteristics"""
        term_lower = term.lower()

            # Direct condition mappings - these are definitely conditions
        condition_terms = [
            'urolithiasis', 'nephrolithiasis', 'myocardial infarction', 'pneumonia',
            'diabetes', 'hypertension', 'asthma', 'copd', 'bronchitis', 'cancer',
            'infection', 'disease', 'syndrome', 'disorder', 'stone', 'calculi',
            'sarcoma', 'liposarcoma', 'carcinoma', 'tumor', 'neoplasm', 'malignancy',
            'myxofibrosarcoma', 'undifferentiated', 'dedifferentiated'
        ]

        # Check for exact matches first
        if term_lower in condition_terms:
            return "Disease or Syndrome"

        # Check for partial matches
        if any(condition in term_lower for condition in condition_terms):
            return "Disease or Syndrome"

        # Symptom indicators
        if any(word in term_lower for word in ['pain', 'ache', 'nausea', 'vomiting', 'fever', 'cough', 'fatigue']):
            return "Sign or Symptom"

        # Medical suffix-based classification
        elif any(word in term_lower for word in ['itis', 'osis', 'emia', 'opathy', 'algia', 'oma', 'sarcoma', 'carcinoma']):
            return "Disease or Syndrome"

        # Medication indicators
        elif any(word in term_lower for word in ['aspirin', 'lisinopril', 'metformin', 'insulin']):
            return "Pharmacologic Substance"

        # Lab test indicators
        elif any(word in term_lower for word in ['troponin', 'glucose', 'creatinine', 'hemoglobin']):
            return "Laboratory or Test Result"
        else:
            return "Medical Concept"

    def _convert_term_to_predicate(self, term: str, semantic_type: str, context: str = "") -> Optional[str]:
        """Enhanced term to predicate conversion with context awareness"""
        term_clean = term.replace(' ', '_').replace('-', '_')

        # Enhanced mapping based on semantic type and context
        if semantic_type == "Sign or Symptom":
            return f"has_symptom(patient, {term_clean})"
        elif semantic_type == "Disease or Syndrome":
            return f"has_condition(patient, {term_clean})"
        elif semantic_type == "Pharmacologic Substance":
            return f"takes_medication(patient, {term_clean})"
        elif semantic_type == "Laboratory or Test Result":
            # Determine expected value from context
            expected_value = self._infer_lab_value_from_context(term, context)
            return f"has_lab_value(patient, {term_clean}, {expected_value})"
        elif semantic_type == "Organism Function":
            # Handle vital signs
            if 'blood_pressure' in term_clean or 'blood pressure' in term:
                return f"has_vital_sign(patient, blood_pressure, high)"
            elif 'heart_rate' in term_clean or 'heart rate' in term:
                return f"has_vital_sign(patient, heart_rate, normal)"
            elif 'temperature' in term_clean:
                return f"has_vital_sign(patient, temperature, normal)"
            elif 'respiratory_rate' in term_clean or 'respiratory rate' in term:
                return f"has_vital_sign(patient, respiratory_rate, normal)"
        else:
            # Use context to determine predicate type
            return self._convert_term_to_predicate_simple(term, semantic_type)

    def _convert_term_to_predicate_simple(self, term: str, semantic_type: str) -> Optional[str]:
        """Simple term to predicate conversion (fallback method)"""
        term_clean = term.replace(' ', '_').replace('-', '_')

        # Simple mapping based on semantic type
        if semantic_type == "Sign or Symptom":
            return f"has_symptom(patient, {term_clean})"
        elif semantic_type == "Disease or Syndrome":
            return f"has_condition(patient, {term_clean})"
        elif semantic_type == "Pharmacologic Substance":
            return f"takes_medication(patient, {term_clean})"
        elif semantic_type == "Laboratory or Test Result":
            return f"has_lab_value(patient, {term_clean}, normal)"
        else:
            # Default to symptom if unsure
            return f"has_symptom(patient, {term_clean})"

    def _infer_lab_value_from_context(self, term: str, context: str) -> str:
        """Infer expected lab value from context"""
        context_lower = context.lower()
        term_lower = term.lower()

        # Check for elevation indicators
        if any(word in context_lower for word in ['elevated', 'high', 'increased', 'raised']):
            return 'elevated'
        elif any(word in context_lower for word in ['low', 'decreased', 'reduced', 'below']):
            return 'low'
        elif any(word in context_lower for word in ['normal', 'within normal', 'wnl']):
            return 'normal'
        elif any(word in context_lower for word in ['abnormal', 'abnormal']):
            return 'abnormal'
        else:
            # Default based on lab type
            if any(word in term_lower for word in ['troponin', 'ck', 'ldh']):
                return 'elevated'  # Cardiac markers often elevated in disease
            elif any(word in term_lower for word in ['hemoglobin', 'platelets']):
                return 'normal'  # Often normal unless specified
            else:
                return 'normal'  # Safe default

    def _get_known_conditions(self) -> List[str]:
        """Get list of known medical conditions that should be classified as diseases"""
        return [
            'urolithiasis', 'nephrolithiasis', 'myocardial infarction', 'pneumonia',
            'diabetes', 'hypertension', 'asthma', 'copd', 'bronchitis', 'cancer',
            'infection', 'disease', 'syndrome', 'disorder', 'stone', 'calculi',
            'kidney', 'renal', 'cardiac', 'heart', 'lung', 'respiratory', 'liver',
            'hepatic', 'brain', 'neurological', 'endocrine', 'metabolic',
            'sarcoma', 'liposarcoma', 'dedifferentiated liposarcoma', 'myxofibrosarcoma',
            'carcinoma', 'tumor', 'neoplasm', 'malignancy', 'undifferentiated pleomorphic sarcoma',
            'soft tissue sarcoma', 'bone sarcoma'
        ]



    async def _verify_single_predicate_deterministic(self, predicate_str: str, patient_data: Dict) -> Dict:
        """Enhanced verification of a single predicate with comprehensive logic"""
        try:
            # Input validation
            if not predicate_str or not isinstance(predicate_str, str):
                return {
                    "predicate": predicate_str,
                    "verified": False,
                    "confidence_score": 0.0,
                    "evaluation_method": "validation_error",
                    "error": "Invalid predicate string",
                    "medical_reasoning": "Predicate string is empty or invalid"
                }

            if not patient_data or not isinstance(patient_data, dict):
                return {
                    "predicate": predicate_str,
                    "verified": False,
                    "confidence_score": 0.0,
                    "evaluation_method": "validation_error",
                    "error": "Invalid patient data",
                    "medical_reasoning": "Patient data is empty or invalid"
                }

            # Parse predicate to extract components for enhanced validation
            predicate_components = self._parse_predicate_components(predicate_str)
            if not predicate_components:
                return {
                    "predicate": predicate_str,
                    "verified": False,
                    "confidence_score": 0.0,
                    "evaluation_method": "parsing_error",
                    "error": "Could not parse predicate",
                    "medical_reasoning": "Predicate format is invalid"
                }

            # Enhanced validation based on predicate type
            validation_result = self._validate_predicate_data_availability(
                predicate_components, patient_data
            )

            if not validation_result["data_available"]:
                return {
                    "predicate": predicate_str,
                    "verified": False,
                    "confidence_score": 0.0,
                    "evaluation_method": "data_unavailable",
                    "error": validation_result["reason"],
                    "medical_reasoning": validation_result["reason"]
                }

            # Use FOL verifier for core logic
            fol_result = self.fol_verifier.verify_predicate(predicate_str, patient_data)

            # Enhance result with additional context
            enhanced_result = self._enhance_verification_result(
                fol_result, predicate_components, patient_data
            )

            return enhanced_result

        except Exception as e:
            logger.error(f"Error verifying predicate {predicate_str}: {str(e)}")
            return {
                "predicate": predicate_str,
                "verified": False,
                "confidence_score": 0.0,
                "evaluation_method": "error",
                "error": str(e),
                "medical_reasoning": f"Verification error occurred: {str(e)}"
            }

    def _parse_predicate_components(self, predicate_str: str) -> Optional[Dict]:
        """Parse predicate string into components for analysis"""
        try:
            # Match pattern: predicate_name(arg1, arg2, ...)
            match = re.match(r'(\w+)\((.*)\)', predicate_str.strip())
            if not match:
                return None

            predicate_name = match.group(1)
            args_str = match.group(2)

            # Parse arguments
            args = []
            if args_str.strip():
                # Handle nested parentheses and commas
                current_arg = ""
                paren_depth = 0

                for char in args_str:
                    if char == '(':
                        paren_depth += 1
                        current_arg += char
                    elif char == ')':
                        paren_depth -= 1
                        current_arg += char
                    elif char == ',' and paren_depth == 0:
                        args.append(current_arg.strip())
                        current_arg = ""
                    else:
                        current_arg += char

                if current_arg.strip():
                    args.append(current_arg.strip())

            return {
                "name": predicate_name,
                "arguments": args,
                "full_string": predicate_str
            }

        except Exception as e:
            logger.warning(f"Failed to parse predicate '{predicate_str}': {e}")
            return None

    def _validate_predicate_data_availability(self, predicate_components: Dict, patient_data: Dict) -> Dict:
        """Validate if required data is available for predicate verification"""
        predicate_name = predicate_components["name"]
        args = predicate_components["arguments"]

        # Define data requirements for each predicate type
        data_requirements = {
            "has_symptom": ["symptoms"],
            "has_condition": ["medical_history", "current_conditions"],
            "has_lab_value": ["lab_results"],
            "has_vital_sign": ["vitals"],
            "takes_medication": ["current_medications"]
        }

        if predicate_name not in data_requirements:
            return {
                "data_available": False,
                "reason": f"Unknown predicate type: {predicate_name}"
            }

        required_fields = data_requirements[predicate_name]

        # Check if required data fields are present and not empty
        for field in required_fields:
            if field not in patient_data:
                return {
                    "data_available": False,
                    "reason": f"Required data field '{field}' is missing from patient data"
                }

            field_data = patient_data[field]
            if not field_data or (isinstance(field_data, (list, dict)) and len(field_data) == 0):
                return {
                    "data_available": False,
                    "reason": f"Required data field '{field}' is empty"
                }

        return {
            "data_available": True,
            "reason": "All required data is available"
        }

    def _enhance_verification_result(self, fol_result: Dict, predicate_components: Dict, patient_data: Dict) -> Dict:
        """Enhance FOL verification result with additional context and reasoning"""
        enhanced_result = fol_result.copy()

        # Add clinical context
        predicate_name = predicate_components["name"]
        clinical_context = self._generate_clinical_context(
            predicate_name, predicate_components, patient_data, fol_result.get("verified", False)
        )

        enhanced_result["clinical_context"] = clinical_context
        enhanced_result["data_completeness"] = self._assess_data_completeness(
            predicate_components, patient_data
        )

        # Adjust confidence based on data quality
        data_quality_factor = self._calculate_data_quality_factor(patient_data)
        original_confidence = fol_result.get("confidence_score", 0.0)
        enhanced_result["confidence_score"] = min(1.0, original_confidence * data_quality_factor)

        # Add temporal context if available
        if "timestamp" in patient_data:
            enhanced_result["data_timestamp"] = patient_data["timestamp"]

        return enhanced_result

    def _generate_clinical_context(self, predicate_name: str, predicate_components: Dict,
                                 patient_data: Dict, verified: bool) -> str:
        """Generate clinical context for the verification result"""
        args = predicate_components["arguments"]

        if predicate_name == "has_symptom" and len(args) >= 2:
            symptom = args[1].replace('_', ' ')
            if verified:
                return f"Patient exhibits documented symptom: {symptom}"
            else:
                return f"No documentation found for symptom: {symptom}"

        elif predicate_name == "has_condition" and len(args) >= 2:
            condition = args[1].replace('_', ' ')
            if verified:
                return f"Patient has documented condition: {condition}"
            else:
                return f"No documentation found for condition: {condition}"

        elif predicate_name == "has_lab_value" and len(args) >= 3:
            lab = args[1].replace('_', ' ')
            expected = args[2]
            if verified:
                return f"Laboratory result for {lab} matches expected value: {expected}"
            else:
                return f"Laboratory result for {lab} does not match expected value: {expected}"

        elif predicate_name == "takes_medication" and len(args) >= 2:
            medication = args[1].replace('_', ' ')
            if verified:
                return f"Patient is documented as taking: {medication}"
            else:
                return f"No documentation found for medication: {medication}"

        return "Clinical context analysis completed"

    def _assess_data_completeness(self, predicate_components: Dict, patient_data: Dict) -> str:
        """Assess the completeness of data available for verification"""
        predicate_name = predicate_components["name"]

        # Define completeness criteria for each predicate type
        completeness_criteria = {
            "has_symptom": {
                "required": ["symptoms"],
                "optional": ["clinical_notes", "chief_complaint"]
            },
            "has_condition": {
                "required": ["medical_history"],
                "optional": ["current_conditions", "past_medical_history"]
            },
            "has_lab_value": {
                "required": ["lab_results"],
                "optional": ["lab_date", "reference_ranges"]
            },
            "has_vital_sign": {
                "required": ["vitals"],
                "optional": ["vital_signs_date", "measurement_method"]
            },
            "takes_medication": {
                "required": ["current_medications"],
                "optional": ["medication_dose", "medication_frequency"]
            }
        }

        if predicate_name not in completeness_criteria:
            return "Unknown"

        criteria = completeness_criteria[predicate_name]
        required_present = all(field in patient_data for field in criteria["required"])
        optional_present = sum(1 for field in criteria["optional"] if field in patient_data)

        if required_present and optional_present >= len(criteria["optional"]) * 0.5:
            return "High"
        elif required_present:
            return "Medium"
        else:
            return "Low"

    def _calculate_data_quality_factor(self, patient_data: Dict) -> float:
        """Calculate a quality factor for the patient data"""
        quality_score = 1.0

        # Reduce quality if key data is missing
        key_fields = ["symptoms", "medical_history", "current_medications", "vitals", "lab_results"]
        missing_fields = sum(1 for field in key_fields if field not in patient_data or not patient_data[field])

        if missing_fields > 0:
            quality_score *= (1.0 - missing_fields * 0.1)  # 10% reduction per missing field

        # Reduce quality if data is old (if timestamp available)
        if "timestamp" in patient_data:
            try:
                import datetime
                data_time = datetime.datetime.fromisoformat(patient_data["timestamp"].replace('Z', '+00:00'))
                current_time = datetime.datetime.now(datetime.timezone.utc)
                age_days = (current_time - data_time).days

                if age_days > 30:
                    quality_score *= 0.8  # 20% reduction for data older than 30 days
                elif age_days > 7:
                    quality_score *= 0.9  # 10% reduction for data older than 7 days
            except:
                pass  # Ignore timestamp parsing errors

        return max(0.1, quality_score)  # Minimum quality factor of 0.1
    
    def _generate_advanced_report(
        self,
        predicate_strings,
        verification_results,
        disease_probabilities,
        recommendations,
        verification_time
    ) -> AdvancedFOLVerificationReport:
        """Generate comprehensive verification report"""

        total_predicates = len(predicate_strings)
        verified_count = sum(1 for result in verification_results if result.get("verified", False))
        failed_count = total_predicates - verified_count

        # Calculate overall confidence
        if verification_results:
            total_confidence = sum(result.get("confidence_score", 0.0) for result in verification_results)
            overall_confidence = total_confidence / len(verification_results)
        else:
            overall_confidence = 0.0

        # Generate detailed results
        detailed_results = []
        for i, (predicate_str, result) in enumerate(zip(predicate_strings, verification_results)):
            detailed_result = {
                "predicate_index": i,
                "predicate_string": predicate_str,
                "verification_result": result,
                "verification_status": "VERIFIED" if result.get("verified", False) else "FAILED",
                "confidence_level": self._get_confidence_level(result.get("confidence_score", 0.0)),
                "reasoning": result.get("reasoning", "No reasoning provided"),
                "evaluation_method": result.get("evaluation_method", "deterministic_logic"),
                "clinical_significance": self._assess_clinical_significance(predicate_str, result)
            }
            detailed_results.append(detailed_result)

        # Generate medical reasoning summary
        medical_reasoning_summary = self._generate_medical_reasoning_summary(
            predicate_strings, verification_results, disease_probabilities
        )

        return AdvancedFOLVerificationReport(
            total_predicates=total_predicates,
            verified_predicates=verified_count,
            failed_predicates=failed_count,
            overall_confidence=overall_confidence,
            verification_time=verification_time,
            detailed_results=detailed_results,
            medical_reasoning_summary=medical_reasoning_summary,
            disease_probabilities=disease_probabilities,
            clinical_recommendations=recommendations
        )
    
    def _generate_medical_reasoning_summary(self, predicate_strings, results, disease_probs):
        """Generate comprehensive medical reasoning summary"""
        summary_parts = []

        # Overview
        verified_count = sum(1 for r in results if r.get("verified", False))
        total_count = len(predicate_strings)

        summary_parts.append(
            f"Medical AI verification analyzed {total_count} clinical assertions, "
            f"with {verified_count} ({verified_count/total_count*100:.1f}%) verified by patient data."
        )

        # Top disease probabilities
        if disease_probs:
            top_diseases = sorted(disease_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            disease_text = ", ".join([f"{disease} ({prob:.2f})" for disease, prob in top_diseases])
            summary_parts.append(f"Primary diagnostic considerations: {disease_text}.")

        # Key verified findings
        verified_predicates = [p for p, r in zip(predicate_strings, results) if r.get("verified", False)]
        if verified_predicates:
            key_findings = []
            for predicate_str in verified_predicates[:3]:  # Top 3
                # Ensure predicate_str is a string
                if isinstance(predicate_str, list):
                    predicate_str = str(predicate_str[0]) if predicate_str else ""
                elif not isinstance(predicate_str, str):
                    predicate_str = str(predicate_str)
                    
                if predicate_str.startswith("has_symptom"):
                    key_findings.append(f"symptom '{predicate_str.split(',')[1].strip(' )')}'")
                elif predicate_str.startswith("has_lab_value"):
                    key_findings.append(f"lab finding '{predicate_str.split(',')[1].strip()}'")
                elif predicate_str.startswith("has_condition"):
                    key_findings.append(f"condition '{predicate_str.split(',')[1].strip(' )')}'")

            if key_findings:
                # Ensure all items in key_findings are strings
                key_findings = [str(f) if not isinstance(f, str) else f for f in key_findings]
                summary_parts.append(f"Key verified findings include: {', '.join(key_findings)}.")

        # Clinical correlation
        summary_parts.append(
            "This verification provides evidence-based validation of AI diagnostic reasoning "
            "to support clinical decision-making."
        )

        return " ".join(summary_parts)

    def _assess_clinical_significance(self, predicate_str, result):
        """Assess clinical significance of verification result"""
        if result.get("verified", False):
            if predicate_str.startswith("has_condition"):
                return "High - Diagnostic assertion supported"
            elif predicate_str.startswith("has_lab_value"):
                return "High - Objective data confirmed"
            elif predicate_str.startswith("has_symptom"):
                return "Moderate - Symptom documented"
            elif predicate_str.startswith("takes_medication"):
                return "Moderate - Medication documented"
            elif predicate_str.startswith("has_vital_sign"):
                return "High - Vital sign documented"
            else:
                return "Moderate - Finding supported"
        else:
            return "Low - Finding not supported by available data"



    def _generate_cache_key(self, explanation_text: str, patient_data: Dict, patient_id: str) -> str:
        """Generate cache key for verification results"""
        import hashlib
        import json

        # Create hash of explanation + essential patient data
        essential_data = {
            "symptoms": patient_data.get("symptoms", []),
            "vitals": patient_data.get("vitals", {}),
            "lab_results": patient_data.get("lab_results", {}),
            "medical_history": patient_data.get("medical_history", []),
            "current_medications": patient_data.get("current_medications", [])
        }

        cache_input = f"{patient_id}:{explanation_text}:{json.dumps(essential_data, sort_keys=True)}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str):
        """Get cached verification result"""
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_data
            else:
                # Remove expired cache entry
                del self._cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, report):
        """Cache verification result"""
        try:
            self._cache[cache_key] = (report, time.time())

            # Simple cache size management
            if len(self._cache) > 100:
                # Remove oldest entries
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

        except Exception as e:
            logger.warning(f"Failed to cache verification results: {e}")
    
    def _validate_inputs(self, explanation_text: str, patient_data: Dict, patient_id: str) -> Dict:
        """Validate input parameters for verification"""
        try:
            # Validate explanation text
            if not explanation_text or not isinstance(explanation_text, str):
                return {
                    "valid": False,
                    "reason": "Explanation text is required and must be a non-empty string"
                }

            if len(explanation_text.strip()) < 10:
                return {
                    "valid": False,
                    "reason": "Explanation text is too short (minimum 10 characters required)"
                }

            # Validate patient data
            if not patient_data or not isinstance(patient_data, dict):
                return {
                    "valid": False,
                    "reason": "Patient data is required and must be a dictionary"
                }

            # Validate patient ID
            if not patient_id or not isinstance(patient_id, str):
                return {
                    "valid": False,
                    "reason": "Patient ID is required and must be a non-empty string"
                }

            # Check for minimum required patient data fields
            required_fields = ["symptoms", "medical_history"]
            missing_fields = [field for field in required_fields if field not in patient_data]
            if missing_fields:
                logger.warning(f"Missing recommended patient data fields: {missing_fields}")

            return {
                "valid": True,
                "reason": "All input validation checks passed"
            }

        except Exception as e:
            return {
                "valid": False,
                "reason": f"Input validation failed due to error: {str(e)}"
            }

    def _generate_enhanced_cache_key(self, explanation_text: str, patient_data: Dict,
                                    patient_id: str, context: Optional[Dict] = None) -> str:
        """Generate enhanced cache key that includes context and data quality factors"""
        import hashlib
        import json

        # Create comprehensive cache input
        cache_components = {
            "patient_id": patient_id,
            "explanation_hash": hashlib.md5(explanation_text.encode()).hexdigest(),
            "patient_data_hash": self._hash_patient_data(patient_data),
            "context": context or {},
            "timestamp": patient_data.get("timestamp", ""),
            "data_quality": self._calculate_data_quality_factor(patient_data)
        }

        # Create deterministic JSON string
        cache_input = json.dumps(cache_components, sort_keys=True, default=str)
        return hashlib.sha256(cache_input.encode()).hexdigest()

    def _hash_patient_data(self, patient_data: Dict) -> str:
        """Create a hash of essential patient data for caching"""
        import hashlib
        import json

        # Extract only essential data that affects verification results
        essential_data = {
            "symptoms": patient_data.get("symptoms", []),
            "vitals": patient_data.get("vitals", {}),
            "lab_results": patient_data.get("lab_results", {}),
            "medical_history": patient_data.get("medical_history", []),
            "current_medications": patient_data.get("current_medications", []),
            "current_conditions": patient_data.get("current_conditions", [])
        }

        # Convert to sorted JSON string for consistent hashing
        data_str = json.dumps(essential_data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _log_cache_hit(self, cache_key: str):
        """Log cache hit for monitoring purposes"""
        logger.info(f"Cache hit for key: {cache_key[:16]}...")

    async def _verify_predicates_parallel(self, predicate_strings: List[str], patient_data: Dict) -> List[Dict]:
        """Verify multiple predicates using ThreadPoolExecutor for parallel processing"""
        try:
            import concurrent.futures
            import asyncio

            logger.info(f"Starting parallel verification of {len(predicate_strings)} predicates")

            # Create a thread pool for CPU-bound verification tasks
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(predicate_strings), 8)) as executor:
                # Create tasks for parallel execution
                loop = asyncio.get_event_loop()
                tasks = []

                for predicate_str in predicate_strings:
                    # Wrap the async method call for thread execution
                    task = loop.run_in_executor(
                        executor,
                        self._verify_single_predicate_sync,
                        predicate_str,
                        patient_data
                    )
                    tasks.append(task)

                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle exceptions in results
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Parallel verification failed for predicate {predicate_strings[i]}: {result}")
                        processed_results.append({
                            "predicate": predicate_strings[i],
                            "verified": False,
                            "confidence_score": 0.0,
                            "evaluation_method": "parallel_error",
                            "error": str(result),
                            "medical_reasoning": f"Parallel verification error: {str(result)}"
                        })
                    else:
                        processed_results.append(result)

                logger.info(f"Parallel verification completed for {len(predicate_strings)} predicates")
                return processed_results

        except Exception as e:
            logger.error(f"Parallel verification setup failed: {e}")
            # Fallback to sequential processing
            logger.info("Falling back to sequential verification")
            return await self._verify_predicates_sequential(predicate_strings, patient_data)

    def _verify_single_predicate_sync(self, predicate_str: str, patient_data: Dict) -> Dict:
        """Synchronous wrapper for predicate verification (for use with ThreadPoolExecutor)"""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the async verification method
            result = loop.run_until_complete(
                self._verify_single_predicate_deterministic(predicate_str, patient_data)
            )

            loop.close()
            return result

        except Exception as e:
            logger.error(f"Synchronous verification failed for predicate {predicate_str}: {e}")
            return {
                "predicate": predicate_str,
                "verified": False,
                "confidence_score": 0.0,
                "evaluation_method": "sync_error",
                "error": str(e),
                "medical_reasoning": f"Synchronous verification error: {str(e)}"
            }

    async def _verify_predicates_sequential(self, predicate_strings: List[str], patient_data: Dict) -> List[Dict]:
        """Fallback sequential verification method"""
        logger.info(f"Using sequential verification for {len(predicate_strings)} predicates")
        results = []

        for predicate_str in predicate_strings:
            result = await self._verify_single_predicate_deterministic(predicate_str, patient_data)
            results.append(result)

        return results

    def _get_confidence_level(self, confidence_score: float) -> str:
        """Get confidence level description"""
        if confidence_score >= 0.9:
            return "Very High"
        elif confidence_score >= 0.7:
            return "High"
        elif confidence_score >= 0.5:
            return "Moderate"
        elif confidence_score >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def _create_empty_report(self, verification_time: float) -> AdvancedFOLVerificationReport:
        """Create report when no predicates are found"""
        return AdvancedFOLVerificationReport(
            total_predicates=0,
            verified_predicates=0,
            failed_predicates=0,
            overall_confidence=0.0,
            verification_time=verification_time,
            detailed_results=[],
            medical_reasoning_summary="No medical assertions found in the provided explanation.",
            disease_probabilities={},
            clinical_recommendations=["Review explanation text for medical content"]
        )
    
    def _create_error_report(self, error_message: str, verification_time: float) -> AdvancedFOLVerificationReport:
        """Create report when verification fails"""
        return AdvancedFOLVerificationReport(
            total_predicates=0,
            verified_predicates=0,
            failed_predicates=0,
            overall_confidence=0.0,
            verification_time=verification_time,
            detailed_results=[],
            medical_reasoning_summary=f"Verification failed due to system error: {error_message}",
            disease_probabilities={},
            clinical_recommendations=["System error - please retry verification"]
        )

class MedicalReasoningEngine:
    """Enhanced medical reasoning engine for disease probability analysis"""

    def __init__(self):
        logger.info("Initializing Enhanced Medical Reasoning Engine")

    async def analyze_disease_probabilities(
        self,
        predicate_strings: List[str],
        verification_results: List[Dict],
        patient_data: Dict
    ) -> Dict[str, float]:
        """Async version of disease probability analysis"""
        try:
            # Run the synchronous analysis in a thread pool to avoid blocking
            import concurrent.futures
            import asyncio

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    self.analyze_disease_probabilities_sync,
                    predicate_strings,
                    verification_results,
                    patient_data
                )
            return result
        except Exception as e:
            logger.error(f"Error in async disease probability analysis: {e}")
            return {}

    async def generate_clinical_recommendations(
        self,
        disease_probabilities: Dict[str, float],
        patient_data: Dict,
        verification_results: List[Dict]
    ) -> List[str]:
        """Async version of clinical recommendations generation"""
        try:
            # Run the synchronous analysis in a thread pool to avoid blocking
            import concurrent.futures
            import asyncio

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    self.generate_clinical_recommendations_sync,
                    disease_probabilities,
                    patient_data,
                    verification_results
                )
            return result
        except Exception as e:
            logger.error(f"Error in async clinical recommendations: {e}")
            return ["Error generating clinical recommendations"]

    def analyze_disease_probabilities_sync(
        self,
        predicate_strings: List[str],
        verification_results: List[Dict],
        patient_data: Dict
    ) -> Dict[str, float]:
        """Analyze disease probabilities based on verified predicates (synchronous version)"""

        # Extract verified findings from predicate strings
        verified_symptoms = []
        verified_labs = []
        verified_conditions = []
        verified_medications = []

        for predicate_str, result in zip(predicate_strings, verification_results):
            if result.get("verified", False):
                if predicate_str.startswith("has_symptom"):
                    # Extract symptom from predicate
                    parts = predicate_str.split(',')
                    if len(parts) >= 2:
                        symptom = parts[1].strip(' )').replace('_', ' ')
                        verified_symptoms.append(symptom)
                elif predicate_str.startswith("has_lab_value"):
                    # Extract lab from predicate
                    parts = predicate_str.split(',')
                    if len(parts) >= 2:
                        lab = parts[1].strip()
                        verified_labs.append(lab)
                elif predicate_str.startswith("has_condition"):
                    # Extract condition from predicate
                    parts = predicate_str.split(',')
                    if len(parts) >= 2:
                        condition = parts[1].strip(' )').replace('_', ' ')
                        verified_conditions.append(condition)
                elif predicate_str.startswith("takes_medication"):
                    # Extract medication from predicate
                    parts = predicate_str.split(',')
                    if len(parts) >= 2:
                        medication = parts[1].strip(' )').replace('_', ' ')
                        verified_medications.append(medication)

        # Enhanced disease probability calculation
        disease_probabilities = {}

        # Cardiovascular diseases
        if any("chest pain" in symptom or "chest_pain" in symptom for symptom in verified_symptoms):
            if any("troponin" in lab.lower() for lab in verified_labs):
                disease_probabilities["myocardial_infarction"] = 0.90
                disease_probabilities["acute_coronary_syndrome"] = 0.95
            elif any("hypertension" in condition for condition in verified_conditions):
                disease_probabilities["hypertensive_heart_disease"] = 0.75
            else:
                disease_probabilities["angina_pectoris"] = 0.65
                disease_probabilities["chest_wall_pain"] = 0.45

        # Respiratory diseases
        dyspnea_indicators = ["shortness of breath", "dyspnea", "difficulty breathing"]
        if any(any(indicator in symptom for indicator in dyspnea_indicators) for symptom in verified_symptoms):
            disease_probabilities["congestive_heart_failure"] = 0.55
            disease_probabilities["chronic_obstructive_pulmonary_disease"] = 0.45
            disease_probabilities["asthma"] = 0.40
            disease_probabilities["pneumonia"] = 0.35

        # Endocrine diseases
        if any("glucose" in lab.lower() for lab in verified_labs):
            disease_probabilities["diabetes_mellitus"] = 0.85
            if any("metformin" in med.lower() for med in verified_medications):
                disease_probabilities["diabetes_mellitus"] = 0.95  # Higher confidence if on treatment

        # Infectious diseases
        if any("fever" in symptom.lower() or "temperature" in symptom.lower() for symptom in verified_symptoms):
            if any("breath" in symptom for symptom in verified_symptoms):
                disease_probabilities["pneumonia"] = 0.70
                disease_probabilities["influenza"] = 0.50
            else:
                disease_probabilities["urinary_tract_infection"] = 0.40
                disease_probabilities["viral_illness"] = 0.35

        # Neurological diseases
        if any("headache" in symptom.lower() or "dizziness" in symptom.lower() for symptom in verified_symptoms):
            if any("hypertension" in condition for condition in verified_conditions):
                disease_probabilities["hypertensive_encephalopathy"] = 0.60
            else:
                disease_probabilities["migraine"] = 0.45
                disease_probabilities["tension_headache"] = 0.40

        # Renal diseases
        if any("creatinine" in lab.lower() or "bun" in lab.lower() for lab in verified_labs):
            disease_probabilities["chronic_kidney_disease"] = 0.75
            disease_probabilities["acute_kidney_injury"] = 0.60

        # Liver diseases
        if any("bilirubin" in lab.lower() or "albumin" in lab.lower() for lab in verified_labs):
            disease_probabilities["liver_disease"] = 0.70

        return disease_probabilities

    def generate_clinical_recommendations_sync(
        self,
        disease_probabilities: Dict[str, float],
        patient_data: Dict,
        verification_results: List[Dict]
    ) -> List[str]:
        """Generate clinical recommendations based on analysis (synchronous version)"""

        recommendations = []

        # High probability diseases (≥80% confidence)
        very_high_prob_diseases = {k: v for k, v in disease_probabilities.items() if v >= 0.8}

        if very_high_prob_diseases:
            for disease, prob in very_high_prob_diseases.items():
                if "myocardial_infarction" in disease:
                    recommendations.append("🚨 EMERGENCY: Immediate cardiology consultation required")
                    recommendations.append("Obtain serial ECGs and cardiac biomarkers")
                    recommendations.append("Consider emergent cardiac catheterization")
                    recommendations.append("Initiate aspirin 325mg and heparin anticoagulation")
                elif "acute_coronary_syndrome" in disease:
                    recommendations.append("Urgent cardiology evaluation within 24 hours")
                    recommendations.append("Serial troponins and ECG monitoring")
                    recommendations.append("Start dual antiplatelet therapy")
                elif "diabetes_mellitus" in disease and prob >= 0.9:
                    recommendations.append("Endocrinology consultation for diabetes management")
                    recommendations.append("Initiate metformin therapy if not contraindicated")
                    recommendations.append("Order HbA1c, lipid panel, and microalbumin")
                    recommendations.append("Diabetes education and lifestyle counseling")

        # Moderate to high probability diseases (60-80% confidence)
        high_prob_diseases = {k: v for k, v in disease_probabilities.items() if 0.6 <= v < 0.8}

        if high_prob_diseases:
            for disease, prob in high_prob_diseases.items():
                if "heart_failure" in disease:
                    recommendations.append("Cardiology consultation for heart failure evaluation")
                    recommendations.append("Obtain echocardiogram and BNP level")
                    recommendations.append("Initiate ACE inhibitor therapy")
                elif "chronic_kidney_disease" in disease:
                    recommendations.append("Nephrology consultation recommended")
                    recommendations.append("Renal ultrasound and urinalysis")
                    recommendations.append("ACE inhibitor for renal protection")
                elif "hypertension" in disease:
                    recommendations.append("Regular blood pressure monitoring")
                    recommendations.append("Consider ACE inhibitor or ARB therapy")
                    recommendations.append("Lifestyle modifications: low sodium diet, exercise")

        # General recommendations based on verification success
        verified_count = sum(1 for result in verification_results if result.get("verified", False))
        total_count = len(verification_results)

        if total_count > 0:
            success_rate = verified_count / total_count
            if success_rate < 0.4:
                recommendations.append("🔍 Additional diagnostic workup strongly recommended")
                recommendations.append("Consider comprehensive history and physical examination")
                recommendations.append("Order additional laboratory and imaging studies")
                recommendations.append("Consider specialist consultation based on symptoms")
            elif success_rate < 0.6:
                recommendations.append("Additional testing recommended to confirm diagnosis")
                recommendations.append("Clinical correlation with additional findings needed")
                recommendations.append("Consider follow-up evaluation in 1-2 weeks")

        # Age-specific considerations
        age = patient_data.get("age", 0)
        if age >= 65:
            recommendations.append("Geriatric assessment may be beneficial")
        elif age <= 18:
            recommendations.append("Pediatric specialist consultation may be appropriate")

        if not recommendations:
            recommendations.append("Continue routine clinical care and monitoring")
            recommendations.append("Follow up as indicated by clinical presentation")
            recommendations.append("Regular health maintenance and preventive care")

        return recommendations
