from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import re
import logging
from datetime import datetime, timedelta

from services.fol_logic_engine import DeterministicFOLVerifier

logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    predicate_id: str
    verified: bool
    confidence_score: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    verification_method: str
    reasoning: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "predicate_id": self.predicate_id,
            "verified": self.verified,
            "confidence_score": self.confidence_score,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "verification_method": self.verification_method,
            "reasoning": self.reasoning
        }

class PatientDataVerifier:
    def __init__(self):
        """Initialize patient data verification engine"""
        logger.info("Initializing Patient Data Verifier")
        
        self.verification_methods = {
            "has_symptom": self._verify_symptom,
            "has_condition": self._verify_condition,
            "takes_medication": self._verify_medication,
            "has_lab_value": self._verify_lab_value,
            "has_vital_sign": self._verify_vital_sign
        }
        
        # Normal ranges for common lab values and vitals
        self.normal_ranges = {
            "glucose": {"min": 70, "max": 100, "unit": "mg/dL"},
            "creatinine": {"min": 0.6, "max": 1.2, "unit": "mg/dL"},
            "troponin": {"min": 0.0, "max": 0.04, "unit": "ng/mL"},
            "hemoglobin": {"min": 12.0, "max": 16.0, "unit": "g/dL"},
            "white blood cells": {"min": 4.0, "max": 11.0, "unit": "K/uL"},
            "blood pressure": {"systolic": {"min": 90, "max": 120}, "diastolic": {"min": 60, "max": 80}},
            "heart rate": {"min": 60, "max": 100, "unit": "bpm"},
            "temperature": {"min": 97.0, "max": 99.5, "unit": "°F"},
            "respiratory rate": {"min": 12, "max": 20, "unit": "breaths/min"},
            "oxygen saturation": {"min": 95, "max": 100, "unit": "%"}
        }
    
    async def verify_predicates(
        self, 
        predicates: List[Dict], 
        patient_data: Dict,
        db_session=None
    ) -> List[VerificationResult]:
        """Verify all predicates against patient data"""
        verification_results = []
        
        logger.info(f"Verifying {len(predicates)} predicates against patient data")
        
        for predicate in predicates:
            try:
                result = await self._verify_single_predicate(predicate, patient_data, db_session)
                verification_results.append(result)
            except Exception as e:
                logger.error(f"Verification failed for predicate {predicate.get('fol_string', '')}: {str(e)}")
                # Create failed verification result
                result = VerificationResult(
                    predicate_id=predicate.get('fol_string', ''),
                    verified=False,
                    confidence_score=0.0,
                    supporting_evidence=[],
                    contradicting_evidence=[f"Verification error: {str(e)}"],
                    verification_method="error"
                )
                verification_results.append(result)
        
        logger.info(f"Completed verification of {len(verification_results)} predicates")
        return verification_results
    
    async def _verify_single_predicate(
        self, 
        predicate: Dict, 
        patient_data: Dict,
        db_session=None
    ) -> VerificationResult:
        """Verify a single FOL predicate"""
        predicate_type = predicate.get('type')
        predicate_id = predicate.get('fol_string', '')
        
        logger.debug(f"Verifying predicate: {predicate_id}")
        
        if predicate_type not in self.verification_methods:
            logger.warning(f"Unknown predicate type: {predicate_type}")
            return VerificationResult(
                predicate_id=predicate_id,
                verified=False,
                confidence_score=0.0,
                supporting_evidence=[],
                contradicting_evidence=["Unknown predicate type"],
                verification_method="unknown"
            )
        
        verification_func = self.verification_methods[predicate_type]
        return await verification_func(predicate, patient_data, db_session)
    
    async def _verify_symptom(
        self, 
        predicate: Dict, 
        patient_data: Dict,
        db_session=None
    ) -> VerificationResult:
        """Verify symptom predicate against patient data"""
        symptom = predicate['object'].lower()
        negation = predicate.get('negation', False)
        
        supporting_evidence = []
        contradicting_evidence = []
        
        logger.debug(f"Verifying symptom: {symptom} (negation: {negation})")
        
        # Check patient reported symptoms
        patient_symptoms = patient_data.get('symptoms', [])
        symptom_found = any(
            self._semantic_similarity(symptom, reported_symptom.lower()) > 0.7
            for reported_symptom in patient_symptoms
        )
        
        if symptom_found:
            supporting_evidence.append(f"Patient reported symptom matches: {symptom}")
        else:
            contradicting_evidence.append(f"No patient report of symptom: {symptom}")
        
        # Check clinical notes
        clinical_notes = patient_data.get('clinical_notes', '')
        if clinical_notes and symptom in clinical_notes.lower():
            supporting_evidence.append(f"Symptom mentioned in clinical notes")
        
        # Check chief complaint
        chief_complaint = patient_data.get('chief_complaint', '')
        if chief_complaint and self._semantic_similarity(symptom, chief_complaint.lower()) > 0.6:
            supporting_evidence.append(f"Symptom matches chief complaint")
        
        # Check vital signs correlation
        vital_correlation = self._check_vital_correlation(symptom, patient_data.get('vitals', {}))
        if vital_correlation:
            supporting_evidence.append(f"Vital signs consistent with symptom: {vital_correlation}")
        
        # Calculate confidence
        confidence = self._calculate_symptom_confidence(
            supporting_evidence, 
            contradicting_evidence, 
            negation
        )
        
        verified = (confidence > 0.6 and not negation) or (confidence < 0.4 and negation)
        
        return VerificationResult(
            predicate_id=predicate['fol_string'],
            verified=verified,
            confidence_score=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            verification_method="symptom_verification"
        )
    
    async def _verify_condition(
        self, 
        predicate: Dict, 
        patient_data: Dict,
        db_session=None
    ) -> VerificationResult:
        """Verify condition/diagnosis predicate"""
        condition = predicate['object'].lower()
        negation = predicate.get('negation', False)
        
        supporting_evidence = []
        contradicting_evidence = []
        
        logger.debug(f"Verifying condition: {condition} (negation: {negation})")
        
        # Check medical history
        medical_history = patient_data.get('medical_history', [])
        condition_in_history = any(
            self._semantic_similarity(condition, hist_condition.lower()) > 0.8
            for hist_condition in medical_history
        )
        
        if condition_in_history:
            supporting_evidence.append(f"Condition found in medical history")
        else:
            contradicting_evidence.append(f"Condition not found in medical history")
        
        # Check current conditions/diagnoses
        current_conditions = patient_data.get('current_conditions', [])
        if any(self._semantic_similarity(condition, curr_cond.lower()) > 0.8 for curr_cond in current_conditions):
            supporting_evidence.append(f"Condition found in current diagnoses")
        
        # Check ICD codes
        icd_codes = patient_data.get('icd_codes', [])
        if self._check_icd_match(condition, icd_codes):
            supporting_evidence.append(f"ICD code matches condition")
        
        # Check lab values supporting condition
        lab_support = self._check_lab_support_for_condition(
            condition, 
            patient_data.get('lab_results', {})
        )
        if lab_support:
            supporting_evidence.extend(lab_support)
        
        # Check medication support (conditions often have associated medications)
        med_support = self._check_medication_support_for_condition(
            condition,
            patient_data.get('current_medications', [])
        )
        if med_support:
            supporting_evidence.extend(med_support)
        
        # Check symptom support for condition
        symptom_support = self._check_symptom_support_for_condition(
            condition,
            patient_data.get('symptoms', [])
        )
        if symptom_support:
            supporting_evidence.extend(symptom_support)
        
        confidence = self._calculate_evidence_confidence(supporting_evidence, contradicting_evidence, negation)
        
        return VerificationResult(
            predicate_id=predicate['fol_string'],
            verified=confidence > 0.5,  # Lower threshold to account for lab/symptom support
            confidence_score=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            verification_method="condition_verification"
        )
    
    async def _verify_medication(
        self, 
        predicate: Dict, 
        patient_data: Dict,
        db_session=None
    ) -> VerificationResult:
        """Verify medication predicate"""
        medication = predicate['object'].lower()
        negation = predicate.get('negation', False)
        
        supporting_evidence = []
        contradicting_evidence = []
        
        logger.debug(f"Verifying medication: {medication} (negation: {negation})")
        
        # Check current medications
        current_meds = patient_data.get('current_medications', [])
        med_found = any(
            self._check_drug_name_similarity(medication, current_med.lower())
            for current_med in current_meds
        )
        
        if med_found:
            supporting_evidence.append(f"Medication found in current medications")
        else:
            contradicting_evidence.append(f"Medication not in current medication list")
        
        # Check prescription history
        prescription_history = patient_data.get('prescription_history', [])
        if any(self._check_drug_name_similarity(medication, rx.lower()) for rx in prescription_history):
            supporting_evidence.append(f"Medication found in prescription history")
        
        # Check medication reconciliation notes
        med_notes = patient_data.get('medication_notes', '')
        if med_notes and medication in med_notes.lower():
            supporting_evidence.append(f"Medication mentioned in medication notes")
        
        confidence = self._calculate_evidence_confidence(supporting_evidence, contradicting_evidence, negation)
        
        return VerificationResult(
            predicate_id=predicate['fol_string'],
            verified=confidence > 0.8,  # High threshold for medications
            confidence_score=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            verification_method="medication_verification"
        )
    
    async def _verify_lab_value(
        self, 
        predicate: Dict, 
        patient_data: Dict,
        db_session=None
    ) -> VerificationResult:
        """Verify laboratory value predicate"""
        lab_object = predicate['object']
        
        if ':' not in lab_object:
            return VerificationResult(
                predicate_id=predicate['fol_string'],
                verified=False,
                confidence_score=0.0,
                supporting_evidence=[],
                contradicting_evidence=["Invalid lab value format - no ':' separator"],
                verification_method="lab_verification"
            )
        
        parts = lab_object.split(':')
        lab_name = parts[0].strip().lower()
        expected_value = parts[1].strip() if len(parts) > 1 else ""
        
        supporting_evidence = []
        contradicting_evidence = []
        
        logger.debug(f"Verifying lab value: {lab_name} = {expected_value}")
        
        # Check recent lab results
        lab_results = patient_data.get('lab_results', {})
        
        # Find matching lab by name similarity
        matching_lab = None
        actual_value = None
        
        for lab_key, lab_val in lab_results.items():
            if self._semantic_similarity(lab_name, lab_key.lower()) > 0.8:
                matching_lab = lab_key
                actual_value = lab_val
                break
        
        if matching_lab and actual_value is not None:
            if self._compare_lab_values(expected_value, actual_value, lab_name):
                supporting_evidence.append(f"Lab value matches: {matching_lab} = {actual_value}")
            else:
                contradicting_evidence.append(f"Lab value mismatch: expected {expected_value}, got {actual_value}")
        else:
            contradicting_evidence.append(f"Lab value not found: {lab_name}")
        
        # Check if value is within normal range
        if matching_lab and actual_value:
            range_assessment = self._assess_lab_range(lab_name, actual_value)
            if range_assessment:
                supporting_evidence.append(range_assessment)
        
        confidence = len(supporting_evidence) / (len(supporting_evidence) + len(contradicting_evidence) + 1)
        
        return VerificationResult(
            predicate_id=predicate['fol_string'],
            verified=confidence > 0.5,  # Lower threshold for initial matching
            confidence_score=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            verification_method="lab_verification"
        )
    
    async def _verify_vital_sign(
        self, 
        predicate: Dict, 
        patient_data: Dict,
        db_session=None
    ) -> VerificationResult:
        """Verify vital sign predicate"""
        vital_object = predicate['object']
        
        if ':' not in vital_object:
            return VerificationResult(
                predicate_id=predicate['fol_string'],
                verified=False,
                confidence_score=0.0,
                supporting_evidence=[],
                contradicting_evidence=["Invalid vital sign format"],
                verification_method="vital_verification"
            )
        
        parts = vital_object.split(':')
        vital_name = parts[0].strip().lower()
        expected_value = parts[1].strip() if len(parts) > 1 else ""
        
        supporting_evidence = []
        contradicting_evidence = []
        
        logger.debug(f"Verifying vital sign: {vital_name} = {expected_value}")
        
        # Check vital signs
        vitals = patient_data.get('vitals', {})
        
        # Map vital names
        vital_mappings = {
            "blood_pressure": ["blood pressure", "bp", "blood_pressure"],
            "heart_rate": ["heart rate", "hr", "pulse", "heart_rate"],
            "temperature": ["temperature", "temp", "body temperature"],
            "respiratory_rate": ["respiratory rate", "rr", "resp rate"],
            "oxygen_saturation": ["oxygen saturation", "o2 sat", "spo2", "pulse ox"]
        }
        
        matching_vital = None
        actual_value = None
        
        # Find matching vital
        for vital_key, vital_val in vitals.items():
            if self._semantic_similarity(vital_name, vital_key.lower()) > 0.7:
                matching_vital = vital_key
                actual_value = vital_val
                break
        
        # Try mapping variations
        if not matching_vital:
            for mapped_name, variations in vital_mappings.items():
                if any(self._semantic_similarity(vital_name, var) > 0.8 for var in variations):
                    if mapped_name in vitals:
                        matching_vital = mapped_name
                        actual_value = vitals[mapped_name]
                        break
        
        if matching_vital and actual_value is not None:
            if self._compare_vital_values(expected_value, actual_value, vital_name):
                supporting_evidence.append(f"Vital sign matches: {matching_vital} = {actual_value}")
            else:
                contradicting_evidence.append(f"Vital sign mismatch: expected {expected_value}, got {actual_value}")
            
            # Check if value is within normal range
            range_assessment = self._assess_vital_range(vital_name, actual_value)
            if range_assessment:
                supporting_evidence.append(range_assessment)
        else:
            contradicting_evidence.append(f"Vital sign not found: {vital_name}")
        
        confidence = len(supporting_evidence) / (len(supporting_evidence) + len(contradicting_evidence) + 1)
        
        return VerificationResult(
            predicate_id=predicate['fol_string'],
            verified=confidence > 0.8,
            confidence_score=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            verification_method="vital_verification"
        )
    
    def _compare_lab_values(self, expected_value: str, actual_value, lab_name: str) -> bool:
        """Compare expected lab value with actual value"""
        try:
            # Handle different value descriptions
            expected_lower = expected_value.lower().strip()
            
            # If expected value contains descriptive terms
            if any(term in expected_lower for term in ['elevated', 'high', 'increased']):
                return self._is_value_above_normal(lab_name, actual_value)
            elif any(term in expected_lower for term in ['low', 'decreased', 'reduced']):
                return self._is_value_below_normal(lab_name, actual_value)
            elif any(term in expected_lower for term in ['normal', 'wnl', 'within']):
                return self._is_value_normal(lab_name, actual_value)
            else:
                # Try to compare numeric values
                expected_numeric = self._parse_numeric(expected_value)
                actual_numeric = self._parse_numeric(actual_value)
                
                if expected_numeric is not None and actual_numeric is not None:
                    # Allow for 10% tolerance in lab values
                    tolerance = 0.1 * expected_numeric
                    return abs(expected_numeric - actual_numeric) <= tolerance
                
                # If values contain the same text/description
                return self._semantic_similarity(expected_value, str(actual_value)) > 0.8
        
        except (ValueError, TypeError):
            return False
        
        return False


    def _semantic_similarity(self, term1: str, term2: str) -> float:
        """Calculate semantic similarity between medical terms"""
        if not term1 or not term2:
            return 0.0
        
        # Normalize terms by replacing underscores with spaces and removing extra whitespace
        term1 = re.sub(r'[_-]', ' ', term1.lower().strip())
        term2 = re.sub(r'[_-]', ' ', term2.lower().strip())
        term1 = re.sub(r'\s+', ' ', term1)
        term2 = re.sub(r'\s+', ' ', term2)
        
        # Check for exact match after normalization
        if term1 == term2:
            return 1.0
        
        # Simple word overlap similarity
        words1 = set(term1.split())
        words2 = set(term2.split())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Boost score for substring matches
        if term1 in term2 or term2 in term1:
            jaccard += 0.3
        
        return min(jaccard, 1.0)
    
    def _check_vital_correlation(self, symptom: str, vitals: Dict) -> Optional[str]:
        """Check if vital signs correlate with reported symptom"""
        correlations = {
            "fever": lambda v: self._parse_temperature(v.get('temperature', 98.6)) > 100.4,
            "tachycardia": lambda v: self._parse_numeric(v.get('heart_rate', 70)) > 100,
            "hypertension": lambda v: self._parse_bp(v.get('blood_pressure', '120/80'))[0] > 140,
            "shortness of breath": lambda v: self._parse_numeric(v.get('respiratory_rate', 16)) > 20,
            "hypoxia": lambda v: self._parse_numeric(v.get('oxygen_saturation', 98)) < 95
        }
        
        for symptom_key, check_func in correlations.items():
            if self._semantic_similarity(symptom, symptom_key) > 0.7:
                try:
                    if check_func(vitals):
                        return f"Vital signs support {symptom_key}"
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _parse_bp(self, bp_string: str) -> Tuple[int, int]:
        """Parse blood pressure string"""
        try:
            if isinstance(bp_string, str) and '/' in bp_string:
                systolic, diastolic = map(int, bp_string.split('/'))
                return systolic, diastolic
            elif isinstance(bp_string, (int, float)):
                return int(bp_string), 80  # Assume systolic only
            else:
                return 120, 80  # Default values
        except (ValueError, AttributeError):
            return 120, 80  # Default values
    
    def _parse_temperature(self, temp) -> float:
        """Parse temperature value"""
        try:
            return float(temp)
        except (ValueError, TypeError):
            return 98.6  # Default normal temperature
    
    def _parse_numeric(self, value) -> float:
        """Parse numeric value from various formats"""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                # Extract first number from string
                import re
                numbers = re.findall(r'\d+\.?\d*', value)
                return float(numbers[0]) if numbers else 0.0
            else:
                return 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _calculate_symptom_confidence(
        self, 
        supporting: List[str], 
        contradicting: List[str], 
        negation: bool
    ) -> float:
        """Calculate confidence score for symptom verification"""
        base_confidence = len(supporting) / (len(supporting) + len(contradicting) + 1)
        
        # Adjust for negation
        if negation:
            base_confidence = 1.0 - base_confidence
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _calculate_evidence_confidence(
        self,
        supporting: List[str],
        contradicting: List[str],
        negation: bool
    ) -> float:
        """Calculate confidence based on evidence"""
        if not supporting and not contradicting:
            return 0.5  # Neutral when no evidence
        
        total_evidence = len(supporting) + len(contradicting)
        support_ratio = len(supporting) / total_evidence if total_evidence > 0 else 0.5
        
        # Apply negation
        if negation:
            support_ratio = 1.0 - support_ratio
        
        return support_ratio
    
    def _check_icd_match(self, condition: str, icd_codes: List[str]) -> bool:
        """Check if condition matches any ICD codes"""
        # Simplified ICD-10 mapping
        condition_icd_map = {
            "diabetes": ["E10", "E11", "E12", "E13", "E14"],
            "hypertension": ["I10", "I11", "I12", "I13", "I15"],
            "pneumonia": ["J12", "J13", "J14", "J15", "J18"],
            "myocardial infarction": ["I21", "I22", "I25.2"],
            "heart failure": ["I50"],
            "chronic kidney disease": ["N18"],
            "atrial fibrillation": ["I48"]
        }
        
        for condition_key, expected_codes in condition_icd_map.items():
            if self._semantic_similarity(condition, condition_key) > 0.8:
                return any(any(code in icd for code in expected_codes) for icd in icd_codes)
        
        return False
    
    def _check_lab_support_for_condition(self, condition: str, lab_results: Dict) -> List[str]:
        """Check if lab results support the condition"""
        condition_lab_map = {
            "diabetes": {
                "glucose": lambda x: self._parse_numeric(x) > 126,
                "hemoglobin a1c": lambda x: self._parse_numeric(x) > 6.5,
                "hba1c": lambda x: self._parse_numeric(x) > 6.5
            },
            "kidney disease": {
                "creatinine": lambda x: self._parse_numeric(x) > 1.2,
                "blood urea nitrogen": lambda x: self._parse_numeric(x) > 20,
                "bun": lambda x: self._parse_numeric(x) > 20
            },
            "myocardial infarction": {
                "troponin": lambda x: self._parse_numeric(x) > 0.04,
                "creatine kinase": lambda x: self._parse_numeric(x) > 200,
                "ck-mb": lambda x: self._parse_numeric(x) > 5
            }
        }
        
        supporting_labs = []
        
        for condition_key, lab_checks in condition_lab_map.items():
            if self._semantic_similarity(condition, condition_key) > 0.8:
                for lab_name, check_func in lab_checks.items():
                    # Find matching lab in results
                    for result_key, result_value in lab_results.items():
                        if self._semantic_similarity(lab_name, result_key) > 0.8:
                            try:
                                if check_func(result_value):
                                    supporting_labs.append(f"{result_key} level ({result_value}) supports {condition}")
                            except (ValueError, TypeError):
                                pass
        
        return supporting_labs
    
    def _check_medication_support_for_condition(self, condition: str, medications: List[str]) -> List[str]:
        """Check if medications support the condition"""
        condition_med_map = {
            "diabetes": ["metformin", "insulin", "glipizide", "glyburide", "pioglitazone"],
            "hypertension": ["lisinopril", "amlodipine", "losartan", "hydrochlorothiazide", "atenolol"],
            "heart failure": ["furosemide", "carvedilol", "lisinopril", "spironolactone"],
            "atrial fibrillation": ["warfarin", "apixaban", "rivaroxaban", "diltiazem", "metoprolol"]
        }
        
        supporting_meds = []
        
        for condition_key, expected_meds in condition_med_map.items():
            if self._semantic_similarity(condition, condition_key) > 0.8:
                for expected_med in expected_meds:
                    for patient_med in medications:
                        if self._check_drug_name_similarity(expected_med, patient_med):
                            supporting_meds.append(f"Medication {patient_med} supports {condition}")
        
        return supporting_meds
    
    def _check_drug_name_similarity(self, drug1: str, drug2: str) -> bool:
        """Check if drug names are similar"""
        return self._semantic_similarity(drug1, drug2) > 0.8
    
    def _compare_lab_values(self, expected: str, actual, lab_name: str) -> bool:
        """Compare expected vs actual lab values"""
        try:
            expected_clean = expected.lower().strip()
            actual_str = str(actual).lower().strip()
            
            # Handle qualitative comparisons
            if expected_clean in ["high", "elevated", "increased"]:
                return any(term in actual_str for term in ["high", "elevated", "increased", "abnormal"]) or \
                       self._is_value_above_normal(lab_name, actual)
            
            elif expected_clean in ["low", "decreased", "reduced"]:
                return any(term in actual_str for term in ["low", "decreased", "reduced"]) or \
                       self._is_value_below_normal(lab_name, actual)
            

            
            elif expected_clean in ["normal", "wnl", "within normal limits"]:
                return any(term in actual_str for term in ["normal", "wnl", "within normal limits"]) or \
                       self._is_value_normal(lab_name, actual)
            
            # Handle numeric comparisons
            expected_num = self._parse_numeric(expected)
            actual_num = self._parse_numeric(actual)
            
            if expected_num > 0 and actual_num > 0:
                # Allow 15% variance for lab values
                return abs(expected_num - actual_num) / expected_num < 0.15
            
            # Fallback to string comparison
            return expected_clean == actual_str
            
        except Exception:
            return False
    
    def _compare_vital_values(self, expected: str, actual, vital_name: str) -> bool:
        """Compare expected vs actual vital sign values"""
        try:
            # Handle blood pressure specially
            if "blood" in vital_name or "bp" in vital_name:
                expected_bp = self._parse_bp(expected)
                actual_bp = self._parse_bp(str(actual))
                return abs(expected_bp[0] - actual_bp[0]) < 10 and abs(expected_bp[1] - actual_bp[1]) < 5
            
            # Handle numeric vitals
            expected_num = self._parse_numeric(expected)
            actual_num = self._parse_numeric(actual)
            
            if expected_num > 0 and actual_num > 0:
                # Allow 10% variance for vitals
                return abs(expected_num - actual_num) / expected_num < 0.10
            
            return False
            
        except Exception:
            return False
    
    def _assess_lab_range(self, lab_name: str, value) -> Optional[str]:
        """Assess if lab value is within normal range"""
        try:
            numeric_value = self._parse_numeric(value)
            
            for range_key, range_info in self.normal_ranges.items():
                if self._semantic_similarity(lab_name, range_key) > 0.8:
                    min_val = range_info.get("min", 0)
                    max_val = range_info.get("max", float('inf'))
                    
                    if numeric_value < min_val:
                        return f"{lab_name} value ({numeric_value}) is below normal range"
                    elif numeric_value > max_val:
                        return f"{lab_name} value ({numeric_value}) is above normal range"
                    else:
                        return f"{lab_name} value ({numeric_value}) is within normal range"
            
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _assess_vital_range(self, vital_name: str, value) -> Optional[str]:
        """Assess if vital sign is within normal range"""
        return self._assess_lab_range(vital_name, value)  # Same logic
    
    def _is_value_above_normal(self, lab_name: str, value) -> bool:
        """Check if value is above normal range"""
        try:
            numeric_value = self._parse_numeric(value)
            for range_key, range_info in self.normal_ranges.items():
                if self._semantic_similarity(lab_name, range_key) > 0.8:
                    return numeric_value > range_info.get("max", float('inf'))
        except:
            pass
        return False
    
    def _is_value_below_normal(self, lab_name: str, value) -> bool:
        """Check if value is below normal range"""
        try:
            numeric_value = self._parse_numeric(value)
            for range_key, range_info in self.normal_ranges.items():
                if self._semantic_similarity(lab_name, range_key) > 0.8:
                    return numeric_value < range_info.get("min", 0)
        except:
            pass
        return False
    
    def _is_value_normal(self, lab_name: str, value) -> bool:
        """Check if value is within normal range"""
        try:
            numeric_value = self._parse_numeric(value)
            for range_key, range_info in self.normal_ranges.items():
                if self._semantic_similarity(lab_name, range_key) > 0.8:
                    min_val = range_info.get("min", 0)
                    max_val = range_info.get("max", float('inf'))
                    return min_val <= numeric_value <= max_val
        except:
            pass
        return False

    def _check_symptom_support_for_condition(self, condition: str, symptoms: List[str]) -> List[str]:
        """Check if symptoms support a given condition"""
        supporting_symptoms = []

        # Common symptom-condition associations
        condition_symptoms = {
            "cholestasis": ["jaundice", "yellow skin", "scleral icterus", "itching", "pale stools", "dark urine"],
            "hypercholesterolemia": ["xanthomas", "corneal arcus", "chest pain"],
            "pseudohyponatremia": ["confusion", "altered mental status", "headache", "nausea"],
            "biliary obstruction": ["jaundice", "abdominal pain", "nausea", "vomiting", "pale stools"],
            "malignancy": ["weight loss", "fatigue", "night sweats", "loss of appetite"],
            "pancreatic adenocarcinoma": ["abdominal pain", "weight loss", "jaundice", "steatorrhea"],
            "cholangiocarcinoma": ["jaundice", "abdominal pain", "weight loss", "fever"],
            "diabetes": ["polyuria", "polydipsia", "polyphagia", "fatigue", "blurred vision"],
            "coronary artery disease": ["chest pain", "shortness of breath", "fatigue", "palpitations"],
            "hyponatremia": ["confusion", "headache", "nausea", "seizures", "muscle cramps"]
        }

        condition_lower = condition.lower()

        # Find the most relevant condition
        for known_condition, associated_symptoms in condition_symptoms.items():
            if known_condition in condition_lower or self._semantic_similarity(condition_lower, known_condition) > 0.7:
                # Check which symptoms from the patient match this condition
                for symptom in symptoms:
                    symptom_lower = symptom.lower()
                    for associated_symptom in associated_symptoms:
                        if (associated_symptom in symptom_lower or
                            self._semantic_similarity(symptom_lower, associated_symptom) > 0.8):
                            supporting_symptoms.append(f"Symptom '{symptom}' supports {condition}")
                            break
                break

        return supporting_symptoms

    def verify_string_predicate(self, predicate_str: str, patient_data: Dict) -> Dict:
        """
        Verify a string-based FOL predicate using the deterministic approach

        This method is designed to work with the deterministic FOL engine's string predicates
        """
        try:
            logger.debug(f"Verifying string predicate: {predicate_str}")

            # Parse the predicate string to extract components
            parsed = self._parse_predicate_string(predicate_str)
            if not parsed:
                return {
                    "predicate": predicate_str,
                    "verified": False,
                    "confidence_score": 0.0,
                    "reasoning": "Unable to parse predicate string",
                    "evaluation_method": "string_parsing_error"
                }

            predicate_type = parsed.get("type")
            object_value = parsed.get("object")
            expected_value = parsed.get("expected_value")

            # Route to appropriate verification method based on predicate type
            if predicate_type == "has_symptom":
                return self._verify_symptom_string(predicate_str, object_value, patient_data)
            elif predicate_type == "has_condition":
                return self._verify_condition_string(predicate_str, object_value, patient_data)
            elif predicate_type == "takes_medication":
                return self._verify_medication_string(predicate_str, object_value, patient_data)
            elif predicate_type == "has_lab_value":
                return self._verify_lab_value_string(predicate_str, object_value, expected_value, patient_data)
            elif predicate_type == "has_vital_sign":
                return self._verify_vital_sign_string(predicate_str, object_value, expected_value, patient_data)
            else:
                return {
                    "predicate": predicate_str,
                    "verified": False,
                    "confidence_score": 0.0,
                    "reasoning": f"Unknown predicate type: {predicate_type}",
                    "evaluation_method": "unknown_predicate_type"
                }

        except Exception as e:
            logger.error(f"Error verifying string predicate {predicate_str}: {str(e)}")
            return {
                "predicate": predicate_str,
                "verified": False,
                "confidence_score": 0.0,
                "reasoning": f"Verification error: {str(e)}",
                "evaluation_method": "error"
            }

    def _parse_predicate_string(self, predicate_str: str) -> Optional[Dict]:
        """Parse a FOL predicate string to extract components"""
        try:
            # Clean the predicate string
            clean_predicate = predicate_str.strip()

            # Handle different predicate types
            if clean_predicate.startswith("has_symptom("):
                object_part = clean_predicate.replace("has_symptom(patient,", "").replace(")", "").strip()
                return {
                    "type": "has_symptom",
                    "object": object_part.replace("_", " "),
                    "expected_value": None
                }

            elif clean_predicate.startswith("has_condition("):
                object_part = clean_predicate.replace("has_condition(patient,", "").replace(")", "").strip()
                return {
                    "type": "has_condition",
                    "object": object_part.replace("_", " "),
                    "expected_value": None
                }

            elif clean_predicate.startswith("takes_medication("):
                object_part = clean_predicate.replace("takes_medication(patient,", "").replace(")", "").strip()
                return {
                    "type": "takes_medication",
                    "object": object_part.replace("_", " "),
                    "expected_value": None
                }

            elif clean_predicate.startswith("has_lab_value("):
                # Extract lab name and expected value
                content = clean_predicate.replace("has_lab_value(patient,", "").replace(")", "").strip()
                if "," in content:
                    parts = content.split(",", 1)
                    lab_name = parts[0].strip().replace("_", " ")
                    expected_value = parts[1].strip()
                    return {
                        "type": "has_lab_value",
                        "object": lab_name,
                        "expected_value": expected_value
                    }

            elif clean_predicate.startswith("has_vital_sign("):
                # Extract vital sign name and expected value
                content = clean_predicate.replace("has_vital_sign(patient,", "").replace(")", "").strip()
                if "," in content:
                    parts = content.split(",", 1)
                    vital_name = parts[0].strip().replace("_", " ")
                    expected_value = parts[1].strip()
                    return {
                        "type": "has_vital_sign",
                        "object": vital_name,
                        "expected_value": expected_value
                    }

            return None

        except Exception as e:
            logger.error(f"Error parsing predicate string {predicate_str}: {str(e)}")
            return None

    def _verify_symptom_string(self, predicate_str: str, symptom: str, patient_data: Dict) -> Dict:
        """Verify a symptom predicate from string format"""
        supporting_evidence = []
        contradicting_evidence = []

        # Check patient reported symptoms
        patient_symptoms = patient_data.get('symptoms', [])
        symptom_found = any(
            self._semantic_similarity(symptom, reported_symptom.lower()) > 0.7
            for reported_symptom in patient_symptoms
        )

        if symptom_found:
            supporting_evidence.append(f"Patient reported symptom matches: {symptom}")
        else:
            contradicting_evidence.append(f"No patient report of symptom: {symptom}")

        # Check clinical notes
        clinical_notes = patient_data.get('clinical_notes', '')
        if clinical_notes and symptom in clinical_notes.lower():
            supporting_evidence.append(f"Symptom mentioned in clinical notes")

        # Check chief complaint
        chief_complaint = patient_data.get('chief_complaint', '')
        if chief_complaint and self._semantic_similarity(symptom, chief_complaint.lower()) > 0.6:
            supporting_evidence.append(f"Symptom matches chief complaint")

        confidence = len(supporting_evidence) / (len(supporting_evidence) + len(contradicting_evidence) + 1)

        return {
            "predicate": predicate_str,
            "verified": confidence > 0.6,
            "confidence_score": confidence,
            "reasoning": f"Symptom verification with {len(supporting_evidence)} supporting and {len(contradicting_evidence)} contradicting evidence",
            "evaluation_method": "string_symptom_verification"
        }

    def _verify_condition_string(self, predicate_str: str, condition: str, patient_data: Dict) -> Dict:
        """Verify a condition predicate from string format"""
        supporting_evidence = []
        contradicting_evidence = []

        # Check medical history
        medical_history = patient_data.get('medical_history', [])
        condition_in_history = any(
            self._semantic_similarity(condition, hist_condition.lower()) > 0.8
            for hist_condition in medical_history
        )

        if condition_in_history:
            supporting_evidence.append(f"Condition found in medical history")
        else:
            contradicting_evidence.append(f"Condition not found in medical history")

        # Check current conditions
        current_conditions = patient_data.get('current_conditions', [])
        if any(self._semantic_similarity(condition, curr_cond.lower()) > 0.8 for curr_cond in current_conditions):
            supporting_evidence.append(f"Condition found in current diagnoses")

        # Check ICD codes
        icd_codes = patient_data.get('icd_codes', [])
        if self._check_icd_match(condition, icd_codes):
            supporting_evidence.append(f"ICD code matches condition")

        confidence = len(supporting_evidence) / (len(supporting_evidence) + len(contradicting_evidence) + 1)

        return {
            "predicate": predicate_str,
            "verified": confidence > 0.5,
            "confidence_score": confidence,
            "reasoning": f"Condition verification with {len(supporting_evidence)} supporting evidence",
            "evaluation_method": "string_condition_verification"
        }

    def _verify_medication_string(self, predicate_str: str, medication: str, patient_data: Dict) -> Dict:
        """Verify a medication predicate from string format"""
        supporting_evidence = []
        contradicting_evidence = []

        # Check current medications
        current_meds = patient_data.get('current_medications', [])
        med_found = any(
            self._check_drug_name_similarity(medication, current_med.lower())
            for current_med in current_meds
        )

        if med_found:
            supporting_evidence.append(f"Medication found in current medications")
        else:
            contradicting_evidence.append(f"Medication not in current medication list")

        confidence = len(supporting_evidence) / (len(supporting_evidence) + len(contradicting_evidence) + 1)

        return {
            "predicate": predicate_str,
            "verified": confidence > 0.8,
            "confidence_score": confidence,
            "reasoning": f"Medication verification with {len(supporting_evidence)} supporting evidence",
            "evaluation_method": "string_medication_verification"
        }

    def _verify_lab_value_string(self, predicate_str: str, lab_name: str, expected_value: str, patient_data: Dict) -> Dict:
        """Verify a lab value predicate from string format"""
        supporting_evidence = []
        contradicting_evidence = []

        # Check lab results
        lab_results = patient_data.get('lab_results', {})

        # Find matching lab
        matching_lab = None
        actual_value = None

        for lab_key, lab_val in lab_results.items():
            if self._semantic_similarity(lab_name, lab_key.lower()) > 0.8:
                matching_lab = lab_key
                actual_value = lab_val
                break

        if matching_lab and actual_value is not None:
            if self._compare_lab_values(expected_value, actual_value, lab_name):
                supporting_evidence.append(f"Lab value matches: {matching_lab} = {actual_value}")
            else:
                contradicting_evidence.append(f"Lab value mismatch: expected {expected_value}, got {actual_value}")
        else:
            contradicting_evidence.append(f"Lab value not found: {lab_name}")

        confidence = len(supporting_evidence) / (len(supporting_evidence) + len(contradicting_evidence) + 1)

        return {
            "predicate": predicate_str,
            "verified": confidence > 0.5,
            "confidence_score": confidence,
            "reasoning": f"Lab value verification with {len(supporting_evidence)} supporting evidence",
            "evaluation_method": "string_lab_verification"
        }

    def _verify_vital_sign_string(self, predicate_str: str, vital_name: str, expected_value: str, patient_data: Dict) -> Dict:
        """Verify a vital sign predicate from string format"""
        supporting_evidence = []
        contradicting_evidence = []

        # Check vital signs
        vitals = patient_data.get('vitals', {})

        # Find matching vital
        matching_vital = None
        actual_value = None

        for vital_key, vital_val in vitals.items():
            if self._semantic_similarity(vital_name, vital_key.lower()) > 0.7:
                matching_vital = vital_key
                actual_value = vital_val
                break

        if matching_vital and actual_value is not None:
            if self._compare_vital_values(expected_value, actual_value, vital_name):
                supporting_evidence.append(f"Vital sign matches: {matching_vital} = {actual_value}")
            else:
                contradicting_evidence.append(f"Vital sign mismatch: expected {expected_value}, got {actual_value}")
        else:
            contradicting_evidence.append(f"Vital sign not found: {vital_name}")

        confidence = len(supporting_evidence) / (len(supporting_evidence) + len(contradicting_evidence) + 1)

        return {
            "predicate": predicate_str,
            "verified": confidence > 0.8,
            "confidence_score": confidence,
            "reasoning": f"Vital sign verification with {len(supporting_evidence)} supporting evidence",
            "evaluation_method": "string_vital_verification"
        }
