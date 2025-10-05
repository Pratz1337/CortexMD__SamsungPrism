"""
Medical Text Preprocessing Module for CortexMD
Handles clinical text normalization, entity extraction, and standardization
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import string

@dataclass
class MedicalEntity:
    """Represents an extracted medical entity"""
    text: str
    category: str
    confidence: float
    start_pos: int
    end_pos: int
    normalized_form: Optional[str] = None
    
class MedicalTextPreprocessor:
    """Advanced medical text preprocessing with entity recognition"""
    
    def __init__(self):
        self.symptom_patterns = self._load_symptom_patterns()
        self.medication_patterns = self._load_medication_patterns()
        self.measurement_patterns = self._load_measurement_patterns()
        self.temporal_patterns = self._load_temporal_patterns()
        self.negation_patterns = self._load_negation_patterns()
        
    def _load_symptom_patterns(self) -> Dict[str, List[str]]:
        """Load symptom recognition patterns"""
        return {
            "pain": [
                r"\b(?:chest|abdominal|back|head|neck|joint|muscle)\s+pain\b",
                r"\bpain\s+in\s+(?:chest|abdomen|back|head|neck)\b",
                r"\b(?:aching|throbbing|stabbing|burning|sharp|dull)\s+pain\b",
                r"\bpainful\s+(?:breathing|swallowing|urination)\b"
            ],
            "respiratory": [
                r"\b(?:shortness of breath|dyspnea|breathlessness)\b",
                r"\b(?:wheezing|coughing|cough)\b",
                r"\b(?:difficulty breathing|trouble breathing)\b",
                r"\b(?:chest tightness|chest congestion)\b"
            ],
            "cardiovascular": [
                r"\b(?:palpitations|heart racing|irregular heartbeat)\b",
                r"\b(?:chest discomfort|chest pressure)\b",
                r"\b(?:dizziness|lightheadedness|fainting|syncope)\b"
            ],
            "gastrointestinal": [
                r"\b(?:nausea|vomiting|diarrhea|constipation)\b",
                r"\b(?:abdominal distension|bloating)\b",
                r"\b(?:loss of appetite|decreased appetite)\b"
            ],
            "neurological": [
                r"\b(?:headache|migraine|cephalgia)\b",
                r"\b(?:confusion|disorientation)\b",
                r"\b(?:weakness|numbness|tingling)\b",
                r"\b(?:seizure|convulsion)\b"
            ],
            "constitutional": [
                r"\b(?:fever|pyrexia|temperature)\b",
                r"\b(?:fatigue|tiredness|exhaustion)\b",
                r"\b(?:weight loss|weight gain)\b",
                r"\b(?:night sweats|diaphoresis)\b"
            ]
        }
    
    def _load_medication_patterns(self) -> Dict[str, str]:
        """Load medication recognition patterns"""
        return {
            "dosage": r"\b\d+\s*(?:mg|mcg|g|ml|cc|units?|iu)\b",
            "frequency": r"\b(?:once|twice|three times|four times|q\d+h|bid|tid|qid|daily|weekly)\b",
            "route": r"\b(?:oral|po|iv|im|sc|sublingual|topical|inhaled)\b",
            "common_meds": r"\b(?:aspirin|ibuprofen|acetaminophen|metformin|lisinopril|amlodipine|atorvastatin|metoprolol)\b"
        }
    
    def _load_measurement_patterns(self) -> Dict[str, str]:
        """Load vital signs and measurement patterns"""
        return {
            "blood_pressure": r"\b(?:bp|blood pressure)\s*:?\s*(\d{2,3})/(\d{2,3})\b",
            "heart_rate": r"\b(?:hr|heart rate|pulse)\s*:?\s*(\d{2,3})\s*(?:bpm)?\b",
            "temperature": r"\b(?:temp|temperature)\s*:?\s*(\d{2,3}(?:\.\d)?)\s*(?:°?[fc])?\b",
            "respiratory_rate": r"\b(?:rr|respiratory rate|resp)\s*:?\s*(\d{1,2})\b",
            "oxygen_saturation": r"\b(?:o2 sat|oxygen saturation|spo2)\s*:?\s*(\d{2,3})%?\b",
            "weight": r"\b(?:weight|wt)\s*:?\s*(\d{2,3}(?:\.\d)?)\s*(?:lbs?|kg|pounds?)\b",
            "height": r"\b(?:height|ht)\s*:?\s*(\d+)(?:'(\d+)\"?|\s*(?:ft|feet)\s*(\d+)\s*(?:in|inches)?)?\b"
        }
    
    def _load_temporal_patterns(self) -> Dict[str, str]:
        """Load temporal expression patterns"""
        return {
            "duration": r"\b(?:for|lasting|over|during)\s+(\d+)\s*(days?|weeks?|months?|years?|hours?|minutes?)\b",
            "onset": r"\b(?:started|began|onset)\s+(\d+)\s*(days?|weeks?|months?|years?|hours?)\s+ago\b",
            "frequency_temporal": r"\b(\d+)\s*(?:times?)\s+per\s+(day|week|month|hour)\b",
            "relative_time": r"\b(?:yesterday|today|this morning|last night|last week|recently)\b"
        }
    
    def _load_negation_patterns(self) -> List[str]:
        """Load negation patterns for clinical text"""
        return [
            r"\bno\s+(?:signs?|symptoms?|evidence)\s+of\b",
            r"\bdenies?\b",
            r"\bnegative\s+for\b",
            r"\babsent\b",
            r"\bwithout\b",
            r"\bnot?\s+(?:present|noted|observed|reported)\b",
            r"\bunlikely\b",
            r"\bruled?\s+out\b"
        ]
    
    def preprocess_clinical_text(self, text: str) -> Dict[str, Any]:
        """Main preprocessing function for clinical text"""
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Extract medical entities
        entities = self._extract_medical_entities(cleaned_text)
        
        # Process negations
        negated_entities = self._process_negations(cleaned_text, entities)
        
        # Extract measurements and vital signs
        measurements = self._extract_measurements(cleaned_text)
        
        # Extract temporal information
        temporal_info = self._extract_temporal_info(cleaned_text)
        
        # Normalize medical terminology
        normalized_entities = self._normalize_entities(entities)
        
        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "entities": normalized_entities,
            "negated_entities": negated_entities,
            "measurements": measurements,
            "temporal_info": temporal_info,
            "entity_summary": self._summarize_entities(normalized_entities),
            "clinical_impression": self._generate_clinical_impression(normalized_entities, measurements)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize clinical text"""
        
        # Convert to lowercase for processing (preserve original for display)
        text = text.strip()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Standardize common abbreviations
        abbreviations = {
            r'\bpt\b': 'patient',
            r'\bc/o\b': 'complains of',
            r'\bh/o\b': 'history of',
            r'\bs/p\b': 'status post',
            r'\bw/\b': 'with',
            r'\bw/o\b': 'without',
            r'\by/o\b': 'year old',
            r'\byo\b': 'year old',
            r'\bm\b(?=\s|$)': 'male',
            r'\bf\b(?=\s|$)': 'female'
        }
        
        for abbrev, expansion in abbreviations.items():
            text = re.sub(abbrev, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _extract_medical_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities from text"""
        entities = []
        
        # Extract symptoms
        for category, patterns in self.symptom_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append(MedicalEntity(
                        text=match.group(),
                        category=f"symptom_{category}",
                        confidence=0.8,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        
        # Extract medications
        med_pattern = self.medication_patterns["common_meds"]
        matches = re.finditer(med_pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append(MedicalEntity(
                text=match.group(),
                category="medication",
                confidence=0.9,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        return entities
    
    def _extract_measurements(self, text: str) -> Dict[str, Any]:
        """Extract vital signs and measurements"""
        measurements = {}
        
        for measurement_type, pattern in self.measurement_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if measurement_type == "blood_pressure":
                    systolic, diastolic = match.groups()
                    measurements["blood_pressure"] = {
                        "systolic": int(systolic),
                        "diastolic": int(diastolic),
                        "raw_text": match.group()
                    }
                elif measurement_type == "height" and len(match.groups()) > 1:
                    # Handle feet/inches format
                    feet = match.group(1)
                    inches = match.group(2) or match.group(3) or "0"
                    total_inches = int(feet) * 12 + int(inches)
                    measurements["height"] = {
                        "inches": total_inches,
                        "feet_inches": f"{feet}'{inches}\"",
                        "raw_text": match.group()
                    }
                else:
                    # Single value measurements
                    value = match.group(1)
                    try:
                        measurements[measurement_type] = {
                            "value": float(value),
                            "raw_text": match.group()
                        }
                    except ValueError:
                        pass
        
        return measurements
    
    def _extract_temporal_info(self, text: str) -> List[Dict[str, Any]]:
        """Extract temporal information from text"""
        temporal_info = []
        
        for temporal_type, pattern in self.temporal_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                temporal_info.append({
                    "type": temporal_type,
                    "text": match.group(),
                    "value": match.group(1) if len(match.groups()) > 0 else None,
                    "unit": match.group(2) if len(match.groups()) > 1 else None,
                    "start_pos": match.start(),
                    "end_pos": match.end()
                })
        
        return temporal_info
    
    def _process_negations(self, text: str, entities: List[MedicalEntity]) -> List[str]:
        """Identify negated medical entities"""
        negated_entities = []
        
        for pattern in self.negation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Look for entities within 10 words after negation
                negation_end = match.end()
                scope_text = text[negation_end:negation_end + 100]
                
                for entity in entities:
                    if (entity.start_pos >= negation_end and 
                        entity.start_pos <= negation_end + 100):
                        negated_entities.append(entity.text)
        
        return list(set(negated_entities))
    
    def _normalize_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Normalize medical entities to standard forms"""
        
        normalization_map = {
            "sob": "shortness of breath",
            "dyspnea": "shortness of breath", 
            "cp": "chest pain",
            "ha": "headache",
            "n/v": "nausea and vomiting",
            "diaphoresis": "sweating",
            "pyrexia": "fever"
        }
        
        for entity in entities:
            normalized = normalization_map.get(entity.text.lower())
            if normalized:
                entity.normalized_form = normalized
        
        return entities
    
    def _summarize_entities(self, entities: List[MedicalEntity]) -> Dict[str, List[str]]:
        """Summarize extracted entities by category"""
        summary = {}
        
        for entity in entities:
            category = entity.category
            if category not in summary:
                summary[category] = []
            
            display_text = entity.normalized_form or entity.text
            if display_text not in summary[category]:
                summary[category].append(display_text)
        
        return summary
    
    def _generate_clinical_impression(self, entities: List[MedicalEntity], measurements: Dict[str, Any]) -> str:
        """Generate a clinical impression from extracted data"""
        
        impression_parts = []
        
        # Summarize symptoms
        symptoms = [e for e in entities if e.category.startswith("symptom")]
        if symptoms:
            symptom_text = ", ".join([e.normalized_form or e.text for e in symptoms[:5]])
            impression_parts.append(f"Patient presents with {symptom_text}")
        
        # Summarize vital signs
        if measurements:
            vital_parts = []
            if "blood_pressure" in measurements:
                bp = measurements["blood_pressure"]
                vital_parts.append(f"BP {bp['systolic']}/{bp['diastolic']}")
            if "heart_rate" in measurements:
                hr = measurements["heart_rate"]
                vital_parts.append(f"HR {hr['value']}")
            if "temperature" in measurements:
                temp = measurements["temperature"]
                vital_parts.append(f"Temp {temp['value']}")
            
            if vital_parts:
                impression_parts.append(f"Vital signs: {', '.join(vital_parts)}")
        
        return ". ".join(impression_parts) + "." if impression_parts else "Limited clinical information available."
    
    def extract_chief_complaint(self, text: str) -> str:
        """Extract the chief complaint from clinical text"""
        
        # Patterns for chief complaint identification
        cc_patterns = [
            r"chief complaint:?\s*(.+?)(?:\n|\.)",
            r"cc:?\s*(.+?)(?:\n|\.)",
            r"patient (?:complains of|presents with|reports)\s*(.+?)(?:\n|\.)",
            r"c/o\s*(.+?)(?:\n|\.)"
        ]
        
        for pattern in cc_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                cc = match.group(1).strip()
                return cc[:200]  # Limit length
        
        # Fallback: look for first mentioned symptom
        entities = self._extract_medical_entities(text)
        symptoms = [e for e in entities if e.category.startswith("symptom")]
        if symptoms:
            return symptoms[0].normalized_form or symptoms[0].text
        
        return "Not clearly specified"
    
    def standardize_medical_terminology(self, text: str) -> str:
        """Standardize medical terminology in text"""
        
        # Common medical term standardizations
        standardizations = {
            r"\bmi\b": "myocardial infarction",
            r"\bcopd\b": "chronic obstructive pulmonary disease",
            r"\bhtn\b": "hypertension",
            r"\bdm\b": "diabetes mellitus",
            r"\bcad\b": "coronary artery disease",
            r"\bchf\b": "congestive heart failure",
            r"\bafib\b": "atrial fibrillation",
            r"\bpe\b": "pulmonary embolism",
            r"\bdvt\b": "deep vein thrombosis",
            r"\buri\b": "upper respiratory infection",
            r"\buti\b": "urinary tract infection"
        }
        
        standardized_text = text
        for abbrev, full_form in standardizations.items():
            standardized_text = re.sub(abbrev, full_form, standardized_text, flags=re.IGNORECASE)
        
        return standardized_text
