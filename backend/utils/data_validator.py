"""
Data Validation and Cleaning Module for CortexMD
Implements data quality checks, anonymization, and standardization
"""

import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum
import json

class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationResult:
    """Result of data validation"""
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None

class DataValidator:
    """Comprehensive data validation for medical data"""
    
    def __init__(self):
        self.phi_patterns = self._load_phi_patterns()
        self.medical_value_ranges = self._load_medical_ranges()
        self.required_fields = self._load_required_fields()
        
    def _load_phi_patterns(self) -> Dict[str, str]:
        """Load patterns for detecting PHI (Protected Health Information)"""
        return {
            "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "mrn": r"\b(?:mrn|medical record)\s*:?\s*([A-Z0-9]{6,})\b",
            "date_birth": r"\b(?:dob|born|birth)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
            "full_date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            "name_patterns": r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+)*",
            "address": r"\b\d+\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd)\b"
        }
    
    def _load_medical_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Load normal ranges for medical values"""
        return {
            "vital_signs": {
                "systolic_bp": {"min": 80, "max": 200, "unit": "mmHg"},
                "diastolic_bp": {"min": 40, "max": 120, "unit": "mmHg"},
                "heart_rate": {"min": 30, "max": 180, "unit": "bpm"},
                "respiratory_rate": {"min": 8, "max": 40, "unit": "breaths/min"},
                "temperature_f": {"min": 95.0, "max": 106.0, "unit": "°F"},
                "temperature_c": {"min": 35.0, "max": 41.0, "unit": "°C"},
                "oxygen_saturation": {"min": 80, "max": 100, "unit": "%"},
                "body_weight": {"min": 50, "max": 500, "unit": "lbs"},
                "height_inches": {"min": 36, "max": 84, "unit": "inches"}
            },
            "lab_values": {
                "glucose": {"min": 50, "max": 400, "unit": "mg/dL"},
                "hemoglobin": {"min": 8.0, "max": 18.0, "unit": "g/dL"},
                "white_blood_cell": {"min": 3.0, "max": 15.0, "unit": "K/uL"},
                "creatinine": {"min": 0.5, "max": 5.0, "unit": "mg/dL"},
                "sodium": {"min": 130, "max": 150, "unit": "mEq/L"},
                "potassium": {"min": 3.0, "max": 6.0, "unit": "mEq/L"}
            }
        }
    
    def _load_required_fields(self) -> Dict[str, List[str]]:
        """Load required fields for different data types"""
        return {
            "patient": ["age", "gender"],
            "vital_signs": [],  # No required vitals, but at least one recommended
            "diagnosis_request": ["text_data"],  # At least some clinical information
            "fhir_patient": ["id", "gender"]
        }
    
    def validate_patient_data(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Comprehensive validation of patient data"""
        results = []
        
        # Check for PHI
        results.extend(self._check_phi(data))
        
        # Validate required fields
        results.extend(self._validate_required_fields(data))
        
        # Validate vital signs
        if "vital_signs" in data:
            results.extend(self._validate_vital_signs(data["vital_signs"]))
        
        # Validate demographic data
        if "patient" in data:
            results.extend(self._validate_demographics(data["patient"]))
        
        # Validate medical history
        if "medical_history" in data or "conditions" in data:
            results.extend(self._validate_medical_history(data))
        
        # Check data completeness
        results.extend(self._assess_data_completeness(data))
        
        return results
    
    def _check_phi(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Check for Protected Health Information"""
        results = []
        data_str = json.dumps(data, default=str)
        
        for phi_type, pattern in self.phi_patterns.items():
            matches = re.findall(pattern, data_str, re.IGNORECASE)
            if matches:
                results.append(ValidationResult(
                    severity=ValidationSeverity.ERROR,
                    message=f"Potential {phi_type.upper()} detected in data",
                    field=phi_type,
                    suggestion="Remove or anonymize personal health information"
                ))
        
        return results
    
    def _validate_required_fields(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate presence of required fields"""
        results = []
        
        # Check patient data
        if "patient" in data:
            patient_data = data["patient"]
            for field in self.required_fields["patient"]:
                if field not in patient_data or not patient_data[field]:
                    results.append(ValidationResult(
                        severity=ValidationSeverity.WARNING,
                        message=f"Missing required patient field: {field}",
                        field=f"patient.{field}",
                        suggestion=f"Provide {field} for better diagnostic accuracy"
                    ))
        
        # Check for any clinical data
        clinical_fields = ["text_data", "symptoms", "vital_signs", "medical_history"]
        has_clinical_data = any(field in data and data[field] for field in clinical_fields)
        
        if not has_clinical_data:
            results.append(ValidationResult(
                severity=ValidationSeverity.ERROR,
                message="No clinical data provided",
                suggestion="Provide clinical symptoms, history, or examination findings"
            ))
        
        return results
    
    def _validate_vital_signs(self, vitals: Dict[str, Any]) -> List[ValidationResult]:
        """Validate vital signs against normal ranges"""
        results = []
        ranges = self.medical_value_ranges["vital_signs"]
        
        for vital, value in vitals.items():
            if vital in ranges:
                range_info = ranges[vital]
                
                # Extract numeric value
                numeric_value = self._extract_numeric_value(value)
                
                if numeric_value is not None:
                    if numeric_value < range_info["min"]:
                        results.append(ValidationResult(
                            severity=ValidationSeverity.WARNING,
                            message=f"{vital} ({numeric_value} {range_info['unit']}) below normal range",
                            field=f"vital_signs.{vital}",
                            suggestion=f"Normal range: {range_info['min']}-{range_info['max']} {range_info['unit']}"
                        ))
                    elif numeric_value > range_info["max"]:
                        results.append(ValidationResult(
                            severity=ValidationSeverity.WARNING,
                            message=f"{vital} ({numeric_value} {range_info['unit']}) above normal range",
                            field=f"vital_signs.{vital}",
                            suggestion=f"Normal range: {range_info['min']}-{range_info['max']} {range_info['unit']}"
                        ))
                else:
                    results.append(ValidationResult(
                        severity=ValidationSeverity.WARNING,
                        message=f"Invalid {vital} value: {value}",
                        field=f"vital_signs.{vital}",
                        suggestion="Provide numeric value with appropriate unit"
                    ))
        
        return results
    
    def _validate_demographics(self, patient: Dict[str, Any]) -> List[ValidationResult]:
        """Validate patient demographic data"""
        results = []
        
        # Age validation
        if "age" in patient:
            age = patient["age"]
            try:
                age_num = int(age)
                if age_num < 0 or age_num > 150:
                    results.append(ValidationResult(
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid age: {age}",
                        field="patient.age",
                        suggestion="Age should be between 0 and 150"
                    ))
            except (ValueError, TypeError):
                results.append(ValidationResult(
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid age format: {age}",
                    field="patient.age",
                    suggestion="Age should be a number"
                ))
        
        # Gender validation
        if "gender" in patient:
            gender = str(patient["gender"]).lower()
            valid_genders = ["male", "female", "m", "f", "other", "unknown"]
            if gender not in valid_genders:
                results.append(ValidationResult(
                    severity=ValidationSeverity.WARNING,
                    message=f"Unusual gender value: {patient['gender']}",
                    field="patient.gender",
                    suggestion="Use standard gender values (male, female, other, unknown)"
                ))
        
        return results
    
    def _validate_medical_history(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate medical history data"""
        results = []
        
        medical_history = data.get("medical_history", []) or data.get("conditions", [])
        
        if medical_history:
            # Check for proper formatting
            for i, condition in enumerate(medical_history):
                if isinstance(condition, str):
                    if len(condition.strip()) < 3:
                        results.append(ValidationResult(
                            severity=ValidationSeverity.WARNING,
                            message=f"Very short medical history entry: '{condition}'",
                            field=f"medical_history[{i}]",
                            suggestion="Provide more detailed medical history"
                        ))
                elif isinstance(condition, dict):
                    if "name" not in condition:
                        results.append(ValidationResult(
                            severity=ValidationSeverity.WARNING,
                            message=f"Medical condition missing name field",
                            field=f"medical_history[{i}]",
                            suggestion="Include condition name"
                        ))
        
        return results
    
    def _assess_data_completeness(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Assess overall data completeness for diagnosis"""
        results = []
        completeness_score = 0
        max_score = 10
        
        # Check presence of key data types
        if "patient" in data and data["patient"]:
            completeness_score += 2
        if ("text_data" in data and data["text_data"]) or ("symptoms" in data and data["symptoms"]):
            completeness_score += 3
        if "vital_signs" in data and data["vital_signs"]:
            completeness_score += 2
        if "medical_history" in data and data["medical_history"]:
            completeness_score += 2
        if "medications" in data and data["medications"]:
            completeness_score += 1
        
        completeness_percentage = (completeness_score / max_score) * 100
        
        if completeness_percentage < 40:
            results.append(ValidationResult(
                severity=ValidationSeverity.WARNING,
                message=f"Low data completeness ({completeness_percentage:.0f}%)",
                suggestion="Provide more clinical information for better diagnosis accuracy"
            ))
        elif completeness_percentage < 70:
            results.append(ValidationResult(
                severity=ValidationSeverity.INFO,
                message=f"Moderate data completeness ({completeness_percentage:.0f}%)",
                suggestion="Additional clinical data could improve diagnosis accuracy"
            ))
        
        return results
    
    def _extract_numeric_value(self, value: Any) -> Optional[float]:
        """Extract numeric value from various formats"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Try to extract number from string
            numbers = re.findall(r'\d+\.?\d*', value)
            if numbers:
                try:
                    return float(numbers[0])
                except ValueError:
                    pass
        elif isinstance(value, dict) and "value" in value:
            return self._extract_numeric_value(value["value"])
        
        return None

class DataCleaner:
    """Data cleaning and standardization utilities"""
    
    def __init__(self):
        self.standardizations = self._load_standardizations()
    
    def _load_standardizations(self) -> Dict[str, Any]:
        """Load standardization mappings"""
        return {
            "units": {
                "temperature": {"f": "°F", "fahrenheit": "°F", "c": "°C", "celsius": "°C"},
                "weight": {"lbs": "pounds", "lb": "pounds", "pounds": "pounds", "kg": "kg"},
                "height": {"ft": "feet", "in": "inches", "'": "feet", '"': "inches"}
            },
            "gender": {"m": "male", "f": "female", "male": "male", "female": "female"},
            "symptoms": {
                "sob": "shortness of breath",
                "cp": "chest pain", 
                "ha": "headache",
                "n/v": "nausea and vomiting",
                "diaphoresis": "sweating"
            }
        }
    
    def clean_patient_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize patient data"""
        cleaned_data = data.copy()
        
        # Clean patient demographics
        if "patient" in cleaned_data:
            cleaned_data["patient"] = self._clean_demographics(cleaned_data["patient"])
        
        # Clean vital signs
        if "vital_signs" in cleaned_data:
            cleaned_data["vital_signs"] = self._clean_vital_signs(cleaned_data["vital_signs"])
        
        # Clean symptoms
        if "symptoms" in cleaned_data:
            cleaned_data["symptoms"] = self._clean_symptoms(cleaned_data["symptoms"])
        
        # Clean text data
        if "text_data" in cleaned_data:
            cleaned_data["text_data"] = self._clean_text_data(cleaned_data["text_data"])
        
        return cleaned_data
    
    def _clean_demographics(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """Clean patient demographic data"""
        cleaned = patient.copy()
        
        # Standardize gender
        if "gender" in cleaned:
            gender = str(cleaned["gender"]).lower().strip()
            cleaned["gender"] = self.standardizations["gender"].get(gender, gender)
        
        # Clean age
        if "age" in cleaned:
            age_str = str(cleaned["age"]).strip()
            # Extract number from age string
            age_match = re.search(r'\d+', age_str)
            if age_match:
                cleaned["age"] = int(age_match.group())
        
        return cleaned
    
    def _clean_vital_signs(self, vitals: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize vital signs"""
        cleaned = {}
        
        for vital, value in vitals.items():
            # Standardize vital sign names
            standard_name = self._standardize_vital_name(vital)
            
            # Clean the value
            cleaned_value = self._clean_vital_value(value, standard_name)
            
            if cleaned_value is not None:
                cleaned[standard_name] = cleaned_value
        
        return cleaned
    
    def _clean_symptoms(self, symptoms: List[str]) -> List[str]:
        """Clean and standardize symptom list"""
        cleaned_symptoms = []
        
        for symptom in symptoms:
            if isinstance(symptom, str):
                # Normalize to lowercase and strip
                symptom_clean = symptom.lower().strip()
                
                # Apply standardizations
                standardized = self.standardizations["symptoms"].get(symptom_clean, symptom_clean)
                
                if standardized not in cleaned_symptoms:
                    cleaned_symptoms.append(standardized)
        
        return cleaned_symptoms
    
    def _clean_text_data(self, text: str) -> str:
        """Clean clinical text data"""
        if not isinstance(text, str):
            return str(text)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common filler words that don't add clinical value
        filler_patterns = [
            r'\buh+\b', r'\bum+\b', r'\ber+\b',  # Hesitation sounds
            r'\byou know\b', r'\blike\b(?=\s)',  # Filler phrases
        ]
        
        for pattern in filler_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra spaces created by removals
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _standardize_vital_name(self, vital_name: str) -> str:
        """Standardize vital sign names"""
        name_mappings = {
            "bp": "blood_pressure",
            "hr": "heart_rate",
            "pulse": "heart_rate",
            "temp": "temperature",
            "rr": "respiratory_rate",
            "resp": "respiratory_rate",
            "o2sat": "oxygen_saturation",
            "spo2": "oxygen_saturation",
            "wt": "weight",
            "ht": "height"
        }
        
        clean_name = vital_name.lower().strip().replace(" ", "_")
        return name_mappings.get(clean_name, clean_name)
    
    def _clean_vital_value(self, value: Any, vital_type: str) -> Any:
        """Clean vital sign values"""
        if isinstance(value, dict):
            # Already structured - clean the value field
            if "value" in value:
                cleaned_value = self._extract_numeric_value(value["value"])
                if cleaned_value is not None:
                    value["value"] = cleaned_value
            return value
        
        # Extract numeric value
        numeric_value = self._extract_numeric_value(value)
        if numeric_value is not None:
            # Apply unit conversions if needed
            if vital_type == "temperature":
                # Convert Celsius to Fahrenheit if value seems to be in Celsius
                if 30 <= numeric_value <= 45:  # Likely Celsius
                    numeric_value = (numeric_value * 9/5) + 32
            
            return {"value": numeric_value, "raw_text": str(value)}
        
        return None
    
    def _extract_numeric_value(self, value: Any) -> Optional[float]:
        """Extract numeric value from various formats"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Try to extract number from string
            numbers = re.findall(r'\d+\.?\d*', value)
            if numbers:
                try:
                    return float(numbers[0])
                except ValueError:
                    pass
        
        return None

class DataAnonymizer:
    """Anonymize patient data to protect privacy"""
    
    def __init__(self, salt: str = "cortexmd_salt"):
        self.salt = salt
        self.phi_patterns = {
            "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "name": r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",
            "address": r"\b\d+\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|drive|dr)\b"
        }
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize patient data"""
        anonymized = self._deep_copy_dict(data)
        
        # Convert to string for pattern matching
        data_str = json.dumps(anonymized, default=str)
        
        # Apply anonymization patterns
        for phi_type, pattern in self.phi_patterns.items():
            data_str = re.sub(pattern, f"[{phi_type.upper()}_REMOVED]", data_str, flags=re.IGNORECASE)
        
        # Parse back to dict
        try:
            anonymized = json.loads(data_str)
        except json.JSONDecodeError:
            # If parsing fails, apply anonymization field by field
            anonymized = self._anonymize_fields(anonymized)
        
        # Generate pseudonymous patient ID
        if "patient" in anonymized and "id" in anonymized["patient"]:
            original_id = anonymized["patient"]["id"]
            anonymized["patient"]["id"] = self._generate_pseudonym(original_id)
        
        return anonymized
    
    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy dictionary"""
        return json.loads(json.dumps(d, default=str))
    
    def _anonymize_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize specific fields"""
        sensitive_fields = ["name", "email", "phone", "ssn", "address"]
        
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = self._anonymize_fields(value)
            elif isinstance(value, str) and key.lower() in sensitive_fields:
                data[key] = f"[{key.upper()}_REMOVED]"
        
        return data
    
    def _generate_pseudonym(self, original_value: str) -> str:
        """Generate pseudonymous identifier"""
        combined = f"{original_value}{self.salt}"
        hash_obj = hashlib.sha256(combined.encode())
        return f"ANON_{hash_obj.hexdigest()[:8].upper()}"
