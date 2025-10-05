"""
Enhanced FHIR Parser for CortexMD
Supports FHIR R4 standard with medical terminology normalization
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, validator
import re
import json

class FHIRPatient(BaseModel):
    """FHIR Patient resource model"""
    id: Optional[str] = None
    name: Optional[str] = None
    gender: Optional[str] = None
    birth_date: Optional[str] = None
    age: Optional[int] = None
    address: Optional[Dict[str, Any]] = None
    telecom: Optional[List[Dict[str, str]]] = None
    
class FHIRObservation(BaseModel):
    """FHIR Observation resource model"""
    id: Optional[str] = None
    code: Dict[str, Any]  # LOINC/SNOMED codes
    value: Union[str, float, int, Dict[str, Any]]
    unit: Optional[str] = None
    reference_range: Optional[Dict[str, Any]] = None
    effective_datetime: Optional[str] = None
    status: str = "final"
    
class FHIRCondition(BaseModel):
    """FHIR Condition resource model"""
    id: Optional[str] = None
    code: Dict[str, Any]  # ICD-10/SNOMED codes
    clinical_status: str = "active"
    verification_status: str = "confirmed"
    category: Optional[str] = None
    severity: Optional[str] = None
    onset_datetime: Optional[str] = None

class EnhancedFHIRParser:
    """Enhanced FHIR parser with medical terminology support"""
    
    def __init__(self):
        self.vital_signs_mapping = {
            "8480-6": "systolic_bp",
            "8462-4": "diastolic_bp", 
            "8867-4": "heart_rate",
            "9279-1": "respiratory_rate",
            "8310-5": "body_temperature",
            "2708-6": "oxygen_saturation",
            "29463-7": "body_weight",
            "8302-2": "body_height",
            "39156-5": "bmi"
        }
        
        self.symptom_normalizations = {
            "chest pain": ["chest discomfort", "thoracic pain", "cardiac pain"],
            "shortness of breath": ["dyspnea", "breathlessness", "sob"],
            "headache": ["cephalgia", "head pain"],
            "abdominal pain": ["stomach pain", "belly pain", "abdominal discomfort"],
            "nausea": ["feeling sick", "queasiness"],
            "fatigue": ["tiredness", "exhaustion", "weakness"]
        }
        
    def parse_fhir_bundle(self, fhir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a complete FHIR bundle into structured format"""
        
        parsed_data = {
            "patient": {},
            "vital_signs": {},
            "symptoms": [],
            "conditions": [],
            "medications": [],
            "allergies": [],
            "procedures": [],
            "lab_results": {},
            "imaging_results": [],
            "family_history": [],
            "social_history": {}
        }
        
        if "entry" in fhir_data:
            for entry in fhir_data["entry"]:
                resource = entry.get("resource", {})
                resource_type = resource.get("resourceType")
                
                if resource_type == "Patient":
                    parsed_data["patient"] = self._parse_patient(resource)
                elif resource_type == "Observation":
                    self._parse_observation(resource, parsed_data)
                elif resource_type == "Condition":
                    parsed_data["conditions"].append(self._parse_condition(resource))
                elif resource_type == "MedicationStatement":
                    parsed_data["medications"].append(self._parse_medication(resource))
                elif resource_type == "AllergyIntolerance":
                    parsed_data["allergies"].append(self._parse_allergy(resource))
                elif resource_type == "Procedure":
                    parsed_data["procedures"].append(self._parse_procedure(resource))
        
        # Handle simple FHIR format (our current format)
        else:
            if "patient" in fhir_data:
                parsed_data["patient"] = fhir_data["patient"]
            if "symptoms" in fhir_data:
                parsed_data["symptoms"] = self._normalize_symptoms(fhir_data["symptoms"])
            if "vital_signs" in fhir_data:
                parsed_data["vital_signs"] = fhir_data["vital_signs"]
            if "medical_history" in fhir_data:
                parsed_data["conditions"] = [{"name": condition} for condition in fhir_data["medical_history"]]
        
        return parsed_data
    
    def _parse_patient(self, patient_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FHIR Patient resource"""
        patient_data = {
            "id": patient_resource.get("id"),
            "gender": patient_resource.get("gender"),
        }
        
        # Parse name
        if "name" in patient_resource and patient_resource["name"]:
            name = patient_resource["name"][0]
            patient_data["name"] = f"{name.get('given', [''])[0]} {name.get('family', '')}"
        
        # Parse birth date and calculate age
        if "birthDate" in patient_resource:
            patient_data["birth_date"] = patient_resource["birthDate"]
            patient_data["age"] = self._calculate_age(patient_resource["birthDate"])
        
        return patient_data
    
    def _parse_observation(self, observation: Dict[str, Any], parsed_data: Dict[str, Any]):
        """Parse FHIR Observation resource"""
        
        # Extract LOINC code
        code_system = observation.get("code", {})
        loinc_code = None
        
        if "coding" in code_system:
            for coding in code_system["coding"]:
                if coding.get("system") == "http://loinc.org":
                    loinc_code = coding.get("code")
                    break
        
        # Determine observation type
        if loinc_code in self.vital_signs_mapping:
            # It's a vital sign
            vital_name = self.vital_signs_mapping[loinc_code]
            value = self._extract_observation_value(observation)
            parsed_data["vital_signs"][vital_name] = value
            
        elif "category" in observation:
            # Check if it's a lab result
            for category in observation.get("category", []):
                if any("laboratory" in coding.get("code", "").lower() 
                       for coding in category.get("coding", [])):
                    lab_name = self._get_display_name(observation.get("code", {}))
                    lab_value = self._extract_observation_value(observation)
                    parsed_data["lab_results"][lab_name] = lab_value
                    break
    
    def _parse_condition(self, condition: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FHIR Condition resource"""
        return {
            "id": condition.get("id"),
            "name": self._get_display_name(condition.get("code", {})),
            "clinical_status": condition.get("clinicalStatus", {}).get("coding", [{}])[0].get("code"),
            "severity": self._get_display_name(condition.get("severity", {})),
            "onset": condition.get("onsetDateTime"),
            "category": self._get_display_name(condition.get("category", [{}])[0])
        }
    
    def _parse_medication(self, medication: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FHIR MedicationStatement resource"""
        return {
            "id": medication.get("id"),
            "medication": self._get_display_name(medication.get("medicationCodeableConcept", {})),
            "status": medication.get("status"),
            "dosage": medication.get("dosage", [{}])[0].get("text"),
            "effective_period": medication.get("effectivePeriod")
        }
    
    def _parse_allergy(self, allergy: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FHIR AllergyIntolerance resource"""
        return {
            "id": allergy.get("id"),
            "substance": self._get_display_name(allergy.get("code", {})),
            "criticality": allergy.get("criticality"),
            "type": allergy.get("type"),
            "category": allergy.get("category", [None])[0] if allergy.get("category") else None
        }
    
    def _parse_procedure(self, procedure: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FHIR Procedure resource"""
        return {
            "id": procedure.get("id"),
            "name": self._get_display_name(procedure.get("code", {})),
            "status": procedure.get("status"),
            "performed": procedure.get("performedDateTime"),
            "category": self._get_display_name(procedure.get("category", {}))
        }
    
    def _extract_observation_value(self, observation: Dict[str, Any]) -> Any:
        """Extract value from FHIR Observation"""
        
        if "valueQuantity" in observation:
            value_qty = observation["valueQuantity"]
            return {
                "value": value_qty.get("value"),
                "unit": value_qty.get("unit"),
                "code": value_qty.get("code")
            }
        elif "valueString" in observation:
            return observation["valueString"]
        elif "valueBoolean" in observation:
            return observation["valueBoolean"]
        elif "valueCodeableConcept" in observation:
            return self._get_display_name(observation["valueCodeableConcept"])
        
        return None
    
    def _get_display_name(self, codeable_concept: Dict[str, Any]) -> str:
        """Extract display name from CodeableConcept"""
        
        if "text" in codeable_concept:
            return codeable_concept["text"]
        
        if "coding" in codeable_concept and codeable_concept["coding"]:
            coding = codeable_concept["coding"][0]
            return coding.get("display", coding.get("code", "Unknown"))
        
        return "Unknown"
    
    def _calculate_age(self, birth_date: str) -> Optional[int]:
        """Calculate age from birth date"""
        try:
            birth = datetime.strptime(birth_date, "%Y-%m-%d")
            today = datetime.now()
            return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        except:
            return None
    
    def _normalize_symptoms(self, symptoms: List[str]) -> List[str]:
        """Normalize symptom terminology"""
        normalized = []
        
        for symptom in symptoms:
            symptom_lower = symptom.lower().strip()
            
            # Check for exact matches or normalizations
            found_match = False
            for standard_term, variations in self.symptom_normalizations.items():
                if symptom_lower == standard_term or symptom_lower in variations:
                    if standard_term not in normalized:
                        normalized.append(standard_term)
                    found_match = True
                    break
            
            # If no normalization found, add original
            if not found_match:
                normalized.append(symptom)
        
        return normalized
    
    def to_clinical_text(self, parsed_data: Dict[str, Any]) -> str:
        """Convert parsed FHIR data to clinical text narrative"""
        
        text_parts = []
        
        # Patient demographics
        patient = parsed_data.get("patient", {})
        if patient:
            demo_parts = []
            if patient.get("age"):
                demo_parts.append(f"{patient['age']}-year-old")
            if patient.get("gender"):
                demo_parts.append(patient["gender"])
            if demo_parts:
                text_parts.append(" ".join(demo_parts))
        
        # Chief complaint / symptoms
        symptoms = parsed_data.get("symptoms", [])
        if symptoms:
            text_parts.append(f"Presenting with: {', '.join(symptoms)}")
        
        # Vital signs
        vitals = parsed_data.get("vital_signs", {})
        if vitals:
            vital_strs = []
            for key, value in vitals.items():
                if isinstance(value, dict) and "value" in value:
                    vital_strs.append(f"{key}: {value['value']} {value.get('unit', '')}")
                else:
                    vital_strs.append(f"{key}: {value}")
            if vital_strs:
                text_parts.append(f"Vital signs: {', '.join(vital_strs)}")
        
        # Medical history / conditions
        conditions = parsed_data.get("conditions", [])
        if conditions:
            condition_names = [c.get("name", str(c)) for c in conditions if c]
            text_parts.append(f"Medical history: {', '.join(condition_names)}")
        
        # Current medications
        medications = parsed_data.get("medications", [])
        if medications:
            med_names = [m.get("medication", str(m)) for m in medications if m]
            text_parts.append(f"Current medications: {', '.join(med_names)}")
        
        # Lab results
        labs = parsed_data.get("lab_results", {})
        if labs:
            lab_strs = []
            for test, result in labs.items():
                if isinstance(result, dict) and "value" in result:
                    lab_strs.append(f"{test}: {result['value']} {result.get('unit', '')}")
                else:
                    lab_strs.append(f"{test}: {result}")
            if lab_strs:
                text_parts.append(f"Laboratory results: {', '.join(lab_strs)}")
        
        return ". ".join(text_parts) + "."

    def validate_fhir_data(self, fhir_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate FHIR data quality and return issues"""
        
        issues = {
            "errors": [],
            "warnings": [],
            "missing_data": []
        }
        
        # Check for required patient data
        if "patient" not in fhir_data or not fhir_data["patient"]:
            issues["errors"].append("Missing patient information")
        else:
            patient = fhir_data["patient"]
            if not patient.get("age") and not patient.get("birth_date"):
                issues["missing_data"].append("Patient age or birth date")
            if not patient.get("gender"):
                issues["missing_data"].append("Patient gender")
        
        # Check for clinical data
        has_symptoms = bool(fhir_data.get("symptoms"))
        has_vitals = bool(fhir_data.get("vital_signs"))
        has_conditions = bool(fhir_data.get("conditions") or fhir_data.get("medical_history"))
        
        if not any([has_symptoms, has_vitals, has_conditions]):
            issues["warnings"].append("No clinical data (symptoms, vitals, or conditions) provided")
        
        # Validate vital signs ranges
        vitals = fhir_data.get("vital_signs", {})
        if vitals:
            self._validate_vital_signs(vitals, issues)
        
        return issues
    
    def _validate_vital_signs(self, vitals: Dict[str, Any], issues: Dict[str, List[str]]):
        """Validate vital signs are within reasonable ranges"""
        
        ranges = {
            "heart_rate": (30, 200),
            "systolic_bp": (60, 250),
            "diastolic_bp": (30, 150),
            "respiratory_rate": (8, 60),
            "body_temperature": (90, 110),  # Fahrenheit
            "oxygen_saturation": (70, 100)
        }
        
        for vital, value in vitals.items():
            if vital in ranges:
                min_val, max_val = ranges[vital]
                vital_value = value
                
                # Extract numeric value if it's a dict
                if isinstance(value, dict):
                    vital_value = value.get("value", value)
                
                try:
                    numeric_value = float(vital_value)
                    if not (min_val <= numeric_value <= max_val):
                        issues["warnings"].append(f"{vital} ({numeric_value}) outside normal range ({min_val}-{max_val})")
                except (ValueError, TypeError):
                    issues["warnings"].append(f"Invalid {vital} value: {vital_value}")
