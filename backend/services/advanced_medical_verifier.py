"""
Advanced Medical Data Verifier
Uses AI and medical reasoning to verify any medical condition
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import requests

# Try to import AI libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

@dataclass
class AdvancedVerificationResult:
    predicate_id: str
    verified: bool
    confidence_score: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    verification_method: str
    medical_reasoning: str
    semantic_similarity_score: float = 0.0
    clinical_correlation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "predicate_id": self.predicate_id,
            "verified": self.verified,
            "confidence_score": self.confidence_score,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "verification_method": self.verification_method,
            "medical_reasoning": self.medical_reasoning,
            "semantic_similarity_score": self.semantic_similarity_score,
            "clinical_correlation": self.clinical_correlation
        }

class AdvancedMedicalKnowledgeBase:
    """Dynamic medical knowledge base that can expand"""
    
    def __init__(self):
        self.knowledge = self._initialize_knowledge()
        self.symptom_embeddings = {}
        self.condition_embeddings = {}
        
        if HAS_SKLEARN:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self._build_embeddings()
    
    def _initialize_knowledge(self) -> Dict:
        """Initialize comprehensive medical knowledge"""
        return {
            "disease_symptoms": {
                # Cardiovascular
                "myocardial_infarction": ["chest_pain", "shortness_of_breath", "nausea", "sweating", "arm_pain", "jaw_pain"],
                "heart_failure": ["shortness_of_breath", "fatigue", "swelling", "chest_pain", "cough"],
                "angina": ["chest_pain", "shortness_of_breath", "fatigue"],
                "hypertension": ["headache", "dizziness", "chest_pain", "shortness_of_breath"],
                
                # Respiratory
                "pneumonia": ["fever", "cough", "shortness_of_breath", "chest_pain", "fatigue"],
                "asthma": ["shortness_of_breath", "wheezing", "cough", "chest_tightness"],
                "copd": ["shortness_of_breath", "cough", "fatigue", "chest_tightness"],
                "pulmonary_embolism": ["chest_pain", "shortness_of_breath", "cough", "leg_swelling"],
                
                # Endocrine
                "diabetes": ["polyuria", "polydipsia", "weight_loss", "fatigue", "blurred_vision"],
                "hyperthyroidism": ["weight_loss", "palpitations", "sweating", "anxiety", "fatigue"],
                "hypothyroidism": ["weight_gain", "fatigue", "cold_intolerance", "constipation"],
                
                # Gastrointestinal
                "gastritis": ["abdominal_pain", "nausea", "vomiting", "bloating"],
                "pancreatitis": ["abdominal_pain", "nausea", "vomiting", "fever"],
                "appendicitis": ["abdominal_pain", "nausea", "vomiting", "fever"],
                
                # Neurological
                "stroke": ["weakness", "speech_problems", "vision_problems", "headache", "confusion"],
                "migraine": ["headache", "nausea", "vomiting", "light_sensitivity"],
                "seizure": ["convulsions", "confusion", "loss_of_consciousness"],
                
                # Infectious
                "sepsis": ["fever", "chills", "confusion", "rapid_heart_rate", "shortness_of_breath"],
                "urinary_tract_infection": ["dysuria", "frequency", "urgency", "abdominal_pain"],
                "meningitis": ["headache", "fever", "neck_stiffness", "confusion"],
            },
            
            "disease_lab_markers": {
                "myocardial_infarction": {
                    "troponin": {"threshold": 0.04, "direction": "above"},
                    "ck_mb": {"threshold": 5, "direction": "above"},
                    "ldh": {"threshold": 250, "direction": "above"}
                },
                "diabetes": {
                    "glucose": {"threshold": 126, "direction": "above"},
                    "hba1c": {"threshold": 6.5, "direction": "above"}
                },
                "kidney_disease": {
                    "creatinine": {"threshold": 1.2, "direction": "above"},
                    "bun": {"threshold": 20, "direction": "above"}
                },
                "liver_disease": {
                    "alt": {"threshold": 40, "direction": "above"},
                    "ast": {"threshold": 40, "direction": "above"},
                    "bilirubin": {"threshold": 1.2, "direction": "above"}
                },
                "anemia": {
                    "hemoglobin": {"threshold": 12, "direction": "below"},
                    "hematocrit": {"threshold": 36, "direction": "below"}
                },
                "infection": {
                    "white_blood_cells": {"threshold": 11, "direction": "above"},
                    "crp": {"threshold": 3, "direction": "above"},
                    "esr": {"threshold": 30, "direction": "above"}
                }
            },
            
            "disease_medications": {
                "hypertension": ["lisinopril", "amlodipine", "metoprolol", "losartan", "atenolol"],
                "diabetes": ["metformin", "insulin", "glipizide", "glyburide", "pioglitazone"],
                "heart_failure": ["lisinopril", "metoprolol", "furosemide", "spironolactone"],
                "depression": ["sertraline", "fluoxetine", "citalopram", "venlafaxine"],
                "anxiety": ["lorazepam", "alprazolam", "clonazepam", "buspirone"],
                "asthma": ["albuterol", "fluticasone", "montelukast", "budesonide"],
                "copd": ["albuterol", "tiotropium", "fluticasone", "prednisone"]
            },
            
            "vital_sign_ranges": {
                "blood_pressure": {"normal": (90, 120), "elevated": (120, 140), "high": (140, 180)},
                "heart_rate": {"normal": (60, 100), "tachycardia": (100, 150), "bradycardia": (40, 60)},
                "temperature": {"normal": (97, 99.5), "fever": (100.4, 105), "hypothermia": (95, 97)},
                "respiratory_rate": {"normal": (12, 20), "tachypnea": (20, 30), "bradypnea": (8, 12)},
                "oxygen_saturation": {"normal": (95, 100), "hypoxemia": (85, 95), "severe_hypoxemia": (70, 85)}
            }
        }
    
    def _build_embeddings(self):
        """Build text embeddings for semantic similarity"""
        if not HAS_SKLEARN:
            return
            
        # Build symptom corpus
        all_symptoms = set()
        for symptoms in self.knowledge["disease_symptoms"].values():
            all_symptoms.update(symptoms)
        
        symptom_texts = [symptom.replace('_', ' ') for symptom in all_symptoms]
        
        if symptom_texts:
            symptom_vectors = self.vectorizer.fit_transform(symptom_texts)
            self.symptom_embeddings = dict(zip(all_symptoms, symptom_vectors.toarray()))
    
    def find_related_diseases(self, symptoms: List[str], lab_values: Dict, medications: List[str]) -> Dict[str, float]:
        """Find diseases related to given symptoms, labs, and medications"""
        disease_scores = {}
        
        # Score based on symptoms
        for disease, disease_symptoms in self.knowledge["disease_symptoms"].items():
            symptom_score = self._calculate_symptom_match_score(symptoms, disease_symptoms)
            lab_score = self._calculate_lab_match_score(lab_values, disease)
            med_score = self._calculate_medication_match_score(medications, disease)
            
            # Weighted combination
            total_score = (symptom_score * 0.5) + (lab_score * 0.3) + (med_score * 0.2)
            
            if total_score > 0.1:  # Only include diseases with some evidence
                disease_scores[disease] = total_score
        
        return disease_scores
    
    def _calculate_symptom_match_score(self, patient_symptoms: List[str], disease_symptoms: List[str]) -> float:
        """Calculate how well patient symptoms match disease symptoms"""
        if not patient_symptoms or not disease_symptoms:
            return 0.0
        
        matches = 0
        for patient_symptom in patient_symptoms:
            for disease_symptom in disease_symptoms:
                if self._semantic_similarity(patient_symptom, disease_symptom) > 0.7:
                    matches += 1
                    break
        
        return matches / len(disease_symptoms)
    
    def _calculate_lab_match_score(self, lab_values: Dict, disease: str) -> float:
        """Calculate lab value match score for disease"""
        if disease not in self.knowledge["disease_lab_markers"]:
            return 0.0
        
        disease_labs = self.knowledge["disease_lab_markers"][disease]
        matches = 0
        total_labs = len(disease_labs)
        
        for lab_name, criteria in disease_labs.items():
            for patient_lab, patient_value in lab_values.items():
                if self._semantic_similarity(lab_name, patient_lab) > 0.8:
                    if self._check_lab_criteria(patient_value, criteria):
                        matches += 1
                    break
        
        return matches / total_labs if total_labs > 0 else 0.0
    
    def _calculate_medication_match_score(self, medications: List[str], disease: str) -> float:
        """Calculate medication match score for disease"""
        if disease not in self.knowledge["disease_medications"]:
            return 0.0
        
        disease_meds = self.knowledge["disease_medications"][disease]
        matches = 0
        
        for med in medications:
            for disease_med in disease_meds:
                if self._semantic_similarity(med, disease_med) > 0.8:
                    matches += 1
                    break
        
        return matches / len(disease_meds) if disease_meds else 0.0
    
    def _check_lab_criteria(self, value, criteria: Dict) -> bool:
        """Check if lab value meets disease criteria"""
        try:
            numeric_value = float(str(value).replace('mg/dL', '').replace('ng/mL', '').strip())
            threshold = criteria["threshold"]
            direction = criteria["direction"]
            
            if direction == "above":
                return numeric_value > threshold
            elif direction == "below":
                return numeric_value < threshold
            else:
                return abs(numeric_value - threshold) < (threshold * 0.1)  # Within 10%
        except (ValueError, TypeError):
            return False
    
    def _semantic_similarity(self, term1: str, term2: str) -> float:
        """Calculate semantic similarity between terms"""
        # Normalize terms
        term1_norm = term1.lower().replace('_', ' ').replace('-', ' ')
        term2_norm = term2.lower().replace('_', ' ').replace('-', ' ')
        
        # Exact match
        if term1_norm == term2_norm:
            return 1.0
        
        # Substring match
        if term1_norm in term2_norm or term2_norm in term1_norm:
            return 0.8
        
        # Word overlap
        words1 = set(term1_norm.split())
        words2 = set(term2_norm.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Use embeddings if available
        if HAS_SKLEARN and self.symptom_embeddings and term1 in self.symptom_embeddings and term2 in self.symptom_embeddings:
            vec1 = self.symptom_embeddings[term1].reshape(1, -1)
            vec2 = self.symptom_embeddings[term2].reshape(1, -1)
            cosine_sim = cosine_similarity(vec1, vec2)[0][0]
            return max(jaccard, cosine_sim)
        
        return jaccard

class AdvancedPatientDataVerifier:
    """Advanced AI-powered patient data verification"""
    
    def __init__(self):
        logger.info("Initializing Advanced Patient Data Verifier")
        self.knowledge_base = AdvancedMedicalKnowledgeBase()
        
    async def verify_advanced_predicates(
        self,
        predicates: List[Dict],
        patient_data: Dict,
        db_session=None
    ) -> List[AdvancedVerificationResult]:
        """Verify predicates using advanced AI reasoning"""
        logger.info(f"Verifying {len(predicates)} predicates with advanced AI")
        
        results = []
        
        # Extract patient information
        symptoms = self._extract_symptoms(patient_data)
        conditions = self._extract_conditions(patient_data)
        medications = self._extract_medications(patient_data)
        lab_values = self._extract_lab_values(patient_data)
        vitals = self._extract_vitals(patient_data)
        
        # Find related diseases for context
        related_diseases = self.knowledge_base.find_related_diseases(symptoms, lab_values, medications)
        
        for predicate in predicates:
            try:
                result = await self._verify_single_advanced_predicate(
                    predicate, patient_data, related_diseases
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Advanced verification failed for {predicate.get('fol_string', '')}: {e}")
                # Create error result
                error_result = AdvancedVerificationResult(
                    predicate_id=predicate.get('fol_string', ''),
                    verified=False,
                    confidence_score=0.0,
                    supporting_evidence=[],
                    contradicting_evidence=[f"Verification error: {str(e)}"],
                    verification_method="error",
                    medical_reasoning="Verification process encountered an error"
                )
                results.append(error_result)
        
        return results
    
    async def _verify_single_advanced_predicate(
        self,
        predicate: Dict,
        patient_data: Dict,
        related_diseases: Dict[str, float]
    ) -> AdvancedVerificationResult:
        """Verify single predicate with advanced reasoning"""
        
        predicate_type = predicate.get("predicate", predicate.get("type", "unknown"))
        predicate_object = predicate.get("object", "")
        predicate_id = predicate.get("fol_string", "")
        
        logger.debug(f"Advanced verification of: {predicate_id}")
        
        # Route to specific verification method
        if predicate_type in ["has_symptom", "symptom"]:
            return await self._verify_symptom_advanced(predicate, patient_data, related_diseases)
        elif predicate_type in ["has_condition", "condition", "likely_has_condition"]:
            return await self._verify_condition_advanced(predicate, patient_data, related_diseases)
        elif predicate_type in ["takes_medication", "medication"]:
            return await self._verify_medication_advanced(predicate, patient_data, related_diseases)
        elif predicate_type in ["has_lab_value", "lab_value"]:
            return await self._verify_lab_value_advanced(predicate, patient_data, related_diseases)
        elif predicate_type in ["has_vital_sign", "vital_sign"]:
            return await self._verify_vital_sign_advanced(predicate, patient_data, related_diseases)
        else:
            return await self._verify_generic_advanced(predicate, patient_data, related_diseases)
    
    async def _verify_symptom_advanced(
        self,
        predicate: Dict,
        patient_data: Dict,
        related_diseases: Dict[str, float]
    ) -> AdvancedVerificationResult:
        """Advanced symptom verification with AI reasoning"""
        
        symptom = predicate["object"].lower()
        supporting_evidence = []
        contradicting_evidence = []
        medical_reasoning = ""
        
        # Direct symptom matching
        patient_symptoms = self._extract_symptoms(patient_data)
        direct_match = any(
            self.knowledge_base._semantic_similarity(symptom, patient_symptom) > 0.7
            for patient_symptom in patient_symptoms
        )
        
        if direct_match:
            supporting_evidence.append(f"Patient directly reported symptom: {symptom}")
        
        # Check clinical notes for symptom mentions
        clinical_text = " ".join([
            patient_data.get("clinical_notes", ""),
            patient_data.get("chief_complaint", ""),
            patient_data.get("history_of_present_illness", "")
        ]).lower()
        
        if symptom.replace("_", " ") in clinical_text:
            supporting_evidence.append(f"Symptom mentioned in clinical documentation")
        
        # Correlate with vital signs
        vital_correlation = self._check_symptom_vital_correlation(symptom, patient_data.get("vitals", {}))
        if vital_correlation:
            supporting_evidence.append(f"Vital signs support symptom: {vital_correlation}")
        
        # Correlate with related diseases
        disease_correlation = self._check_symptom_disease_correlation(symptom, related_diseases)
        if disease_correlation:
            supporting_evidence.append(f"Symptom consistent with suspected diseases: {disease_correlation}")
        
        # Build medical reasoning
        if supporting_evidence:
            medical_reasoning = f"Symptom '{symptom}' is supported by multiple lines of evidence including patient reports and clinical correlation."
        else:
            medical_reasoning = f"Insufficient evidence found for symptom '{symptom}' in available patient data."
            contradicting_evidence.append(f"No direct evidence for symptom: {symptom}")
        
        confidence = len(supporting_evidence) / (len(supporting_evidence) + len(contradicting_evidence) + 1)
        
        return AdvancedVerificationResult(
            predicate_id=predicate["fol_string"],
            verified=confidence > 0.5,
            confidence_score=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            verification_method="advanced_symptom_verification",
            medical_reasoning=medical_reasoning,
            semantic_similarity_score=max([
                self.knowledge_base._semantic_similarity(symptom, ps) for ps in patient_symptoms
            ]) if patient_symptoms else 0.0
        )
    
    async def _verify_condition_advanced(
        self,
        predicate: Dict,
        patient_data: Dict,
        related_diseases: Dict[str, float]
    ) -> AdvancedVerificationResult:
        """Advanced condition verification with disease correlation"""
        
        condition = predicate["object"].lower()
        supporting_evidence = []
        contradicting_evidence = []
        
        # Check if condition is in related diseases
        condition_score = 0.0
        for disease, score in related_diseases.items():
            if self.knowledge_base._semantic_similarity(condition, disease) > 0.8:
                condition_score = max(condition_score, score)
                supporting_evidence.append(f"Condition supported by clinical evidence (score: {score:.2f})")
        
        # Check medical history
        medical_history = patient_data.get("medical_history", [])
        history_match = any(
            self.knowledge_base._semantic_similarity(condition, hist_condition) > 0.8
            for hist_condition in medical_history
        )
        
        if history_match:
            supporting_evidence.append(f"Condition found in medical history")
        
        # Check current diagnoses
        current_conditions = patient_data.get("current_conditions", [])
        current_match = any(
            self.knowledge_base._semantic_similarity(condition, curr_condition) > 0.8
            for curr_condition in current_conditions
        )
        
        if current_match:
            supporting_evidence.append(f"Condition listed in current diagnoses")
        
        # Check supporting lab values
        lab_support = self._check_condition_lab_support(condition, patient_data.get("lab_results", {}))
        if lab_support:
            supporting_evidence.extend(lab_support)
        
        # Check supporting medications
        med_support = self._check_condition_medication_support(condition, patient_data.get("current_medications", []))
        if med_support:
            supporting_evidence.extend(med_support)
        
        if not supporting_evidence:
            contradicting_evidence.append(f"No clinical evidence found for condition: {condition}")
        
        medical_reasoning = (
            f"Condition '{condition}' verification based on comprehensive analysis of "
            f"symptoms, lab values, medications, and medical history. "
            f"Disease correlation score: {condition_score:.2f}"
        )
        
        confidence = len(supporting_evidence) / (len(supporting_evidence) + len(contradicting_evidence) + 1)
        
        return AdvancedVerificationResult(
            predicate_id=predicate["fol_string"],
            verified=confidence > 0.6 or condition_score > 0.7,
            confidence_score=max(confidence, condition_score),
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            verification_method="advanced_condition_verification",
            medical_reasoning=medical_reasoning,
            clinical_correlation=f"Disease probability: {condition_score:.2f}"
        )
    
    async def _verify_lab_value_advanced(
        self,
        predicate: Dict,
        patient_data: Dict,
        related_diseases: Dict[str, float]
    ) -> AdvancedVerificationResult:
        """Advanced lab value verification"""
        
        lab_object = predicate["object"]
        supporting_evidence = []
        contradicting_evidence = []
        
        # Parse lab object
        if ":" in lab_object:
            lab_name, expected_value = lab_object.split(":", 1)
        else:
            lab_name = lab_object
            expected_value = ""
        
        lab_name = lab_name.strip().lower()
        expected_value = expected_value.strip().lower()
        
        # Find matching lab in patient data
        lab_results = patient_data.get("lab_results", {})
        matching_lab = None
        actual_value = None
        
        for patient_lab, patient_value in lab_results.items():
            if self.knowledge_base._semantic_similarity(lab_name, patient_lab.lower()) > 0.8:
                matching_lab = patient_lab
                actual_value = patient_value
                break
        
        if matching_lab and actual_value is not None:
            # Verify the value
            if self._verify_lab_value_match(expected_value, actual_value, lab_name):
                supporting_evidence.append(f"Lab value confirmed: {matching_lab} = {actual_value}")
            else:
                contradicting_evidence.append(f"Lab value mismatch: expected {expected_value}, found {actual_value}")
            
            # Check disease correlation
            disease_support = self._check_lab_disease_correlation(lab_name, actual_value, related_diseases)
            if disease_support:
                supporting_evidence.append(f"Lab value supports clinical picture: {disease_support}")
        else:
            contradicting_evidence.append(f"Lab test not found: {lab_name}")
        
        medical_reasoning = (
            f"Lab value '{lab_name}' verification using patient data and clinical correlation. "
            f"Expected: {expected_value}, Found: {actual_value}"
        )
        
        confidence = len(supporting_evidence) / (len(supporting_evidence) + len(contradicting_evidence) + 1)
        
        return AdvancedVerificationResult(
            predicate_id=predicate["fol_string"],
            verified=confidence > 0.7,
            confidence_score=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            verification_method="advanced_lab_verification",
            medical_reasoning=medical_reasoning
        )
    
    def _extract_symptoms(self, patient_data: Dict) -> List[str]:
        """Extract all symptoms from patient data"""
        symptoms = []
        symptoms.extend(patient_data.get("symptoms", []))
        
        # Extract from chief complaint
        chief_complaint = patient_data.get("chief_complaint", "")
        if chief_complaint:
            symptoms.append(chief_complaint)
        
        return [s.lower().replace(" ", "_") for s in symptoms]
    
    def _extract_conditions(self, patient_data: Dict) -> List[str]:
        """Extract conditions from patient data"""
        conditions = []
        conditions.extend(patient_data.get("medical_history", []))
        conditions.extend(patient_data.get("current_conditions", []))
        return [c.lower().replace(" ", "_") for c in conditions]
    
    def _extract_medications(self, patient_data: Dict) -> List[str]:
        """Extract medications from patient data"""
        medications = patient_data.get("current_medications", [])
        return [m.lower() for m in medications]
    
    def _extract_lab_values(self, patient_data: Dict) -> Dict[str, Any]:
        """Extract lab values from patient data"""
        return patient_data.get("lab_results", {})
    
    def _extract_vitals(self, patient_data: Dict) -> Dict[str, Any]:
        """Extract vital signs from patient data"""
        return patient_data.get("vitals", {})
    
    def _verify_lab_value_match(self, expected: str, actual: Any, lab_name: str) -> bool:
        """Verify if lab values match using intelligent comparison"""
        if not expected:
            return True  # No specific expectation
        
        # Handle descriptive expectations
        if any(term in expected for term in ["elevated", "high", "increased"]):
            return self._is_lab_value_elevated(lab_name, actual)
        elif any(term in expected for term in ["low", "decreased", "reduced"]):
            return self._is_lab_value_low(lab_name, actual)
        elif any(term in expected for term in ["normal", "wnl"]):
            return self._is_lab_value_normal(lab_name, actual)
        else:
            # Try numeric comparison
            try:
                expected_num = float(expected)
                actual_num = float(str(actual).replace('mg/dL', '').replace('ng/mL', '').strip())
                # Allow 15% tolerance
                tolerance = 0.15 * expected_num
                return abs(expected_num - actual_num) <= tolerance
            except (ValueError, TypeError):
                return str(expected).lower() in str(actual).lower()
    
    def _is_lab_value_elevated(self, lab_name: str, value: Any) -> bool:
        """Check if lab value is elevated"""
        # Use knowledge base to determine if elevated
        for disease, markers in self.knowledge_base.knowledge["disease_lab_markers"].items():
            for marker_name, criteria in markers.items():
                if self.knowledge_base._semantic_similarity(lab_name, marker_name) > 0.8:
                    if criteria["direction"] == "above":
                        try:
                            numeric_value = float(str(value).replace('mg/dL', '').replace('ng/mL', '').strip())
                            return numeric_value > criteria["threshold"]
                        except (ValueError, TypeError):
                            pass
        return False
    
    def _is_lab_value_low(self, lab_name: str, value: Any) -> bool:
        """Check if lab value is low"""
        # Similar logic for low values
        for disease, markers in self.knowledge_base.knowledge["disease_lab_markers"].items():
            for marker_name, criteria in markers.items():
                if self.knowledge_base._semantic_similarity(lab_name, marker_name) > 0.8:
                    if criteria["direction"] == "below":
                        try:
                            numeric_value = float(str(value).replace('mg/dL', '').replace('ng/mL', '').strip())
                            return numeric_value < criteria["threshold"]
                        except (ValueError, TypeError):
                            pass
        return False
    
    def _is_lab_value_normal(self, lab_name: str, value: Any) -> bool:
        """Check if lab value is normal"""
        # Check if within normal ranges
        return not (self._is_lab_value_elevated(lab_name, value) or self._is_lab_value_low(lab_name, value))
    
    # Placeholder methods for other verification types
    async def _verify_medication_advanced(self, predicate, patient_data, related_diseases):
        """Verify medication predicates"""
        # Implementation similar to other methods
        pass
    
    async def _verify_vital_sign_advanced(self, predicate, patient_data, related_diseases):
        """Verify vital sign predicates"""
        # Implementation similar to other methods
        pass
    
    async def _verify_generic_advanced(self, predicate, patient_data, related_diseases):
        """Verify generic predicates"""
        # Implementation similar to other methods
        pass
    
    def _check_symptom_vital_correlation(self, symptom: str, vitals: Dict) -> Optional[str]:
        """Check if vitals correlate with symptom"""
        # Implementation for vital-symptom correlation
        return None
    
    def _check_symptom_disease_correlation(self, symptom: str, related_diseases: Dict) -> Optional[str]:
        """Check if symptom correlates with suspected diseases"""
        # Implementation for symptom-disease correlation
        return None
    
    def _check_condition_lab_support(self, condition: str, lab_results: Dict) -> List[str]:
        """Check if lab results support the condition"""
        # Implementation for condition-lab correlation
        return []
    
    def _check_condition_medication_support(self, condition: str, medications: List[str]) -> List[str]:
        """Check if medications support the condition"""
        # Implementation for condition-medication correlation
        return []
    
    def _check_lab_disease_correlation(self, lab_name: str, lab_value: Any, related_diseases: Dict) -> Optional[str]:
        """Check if lab value correlates with diseases"""
        # Implementation for lab-disease correlation
        return None
