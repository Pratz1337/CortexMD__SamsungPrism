import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PredicateType(Enum):
    HAS_SYMPTOM = "has_symptom"
    HAS_CONDITION = "has_condition"
    TAKES_MEDICATION = "takes_medication"
    HAS_LAB_VALUE = "has_lab_value"
    HAS_VITAL_SIGN = "has_vital_sign"
    HAS_DEMOGRAPHIC = "has_demographic"
    TEMPORAL_RELATION = "temporal_relation"
    CAUSAL_RELATION = "causal_relation"

@dataclass
class FOLPredicate:
    predicate_type: PredicateType
    subject: str
    relation: str
    object: str
    confidence: float
    temporal_modifier: Optional[str] = None
    negation: bool = False
    original_text: str = ""
    
    def to_fol_string(self) -> str:
        """Convert to First-Order Logic string representation"""
        negation_prefix = "¬" if self.negation else ""
        temporal_suffix = f"@{self.temporal_modifier}" if self.temporal_modifier else ""
        
        return f"{negation_prefix}{self.relation}({self.subject}, {self.object}){temporal_suffix}"
    
    def to_dict(self) -> Dict:
        return {
            "type": self.predicate_type.value,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
            "temporal_modifier": self.temporal_modifier,
            "negation": self.negation,
            "fol_string": self.to_fol_string(),
            "original_text": self.original_text
        }

class FOLExtractor:
    def __init__(self):
        """Initialize medical patterns for predicate extraction"""
        logger.info("Initializing FOL Extractor")
        
        # Medical relationship patterns
        self.medical_patterns = {
            "symptom_patterns": [
                r"(?:patient|he|she|they)\s+(?:presents with|has|shows|exhibits|reports)\s+([^,\.;]+?)(?:,|\.|\s+and|\s+with|\s*$)",
                r"(?:experiencing|suffering from|complaining of)\s+([^,\.;]+?)(?:,|\.|\s+and|\s*$)",
                r"(?:symptoms include|symptoms are)\s+([^,\.;]+?)(?:,|\.|\s+and|\s*$)",
                r"(?:chief complaint|cc):\s*([^,\.;]+?)(?:,|\.|\s+and|\s*$)",
                r"(?:patient reports|reports)\s+([^,\.;]+?)(?:,|\.|\s+and|\s*$)",
                r"([a-zA-Z\s]+?)\s*(?:pain|ache|discomfort|difficulty|shortness|nausea|vomiting|fever|headache|dizziness)",
                r"(?:acute|chronic|severe|mild|moderate)\s+([a-zA-Z\s]+?)(?:\s+pain|\s+discomfort|\s+symptoms?)"
            ],
            "condition_patterns": [
                r"(?:diagnosed with|diagnosis of|suffers from|has)\s+([^,\.;]+?)(?:,|\.|\s+and|\s*$)",
                r"(?:patient has|history of)\s+([^,\.;]+?)(?:,|\.|\s+and|\s*$)",
                r"(?:consistent with|suggestive of)\s+([^,\.;]+?)(?:,|\.|\s+and|\s*$)",
                r"(?:condition|disease|disorder):\s*([^,\.;]+?)(?:,|\.|\s+and|\s*$)",
                r"(?:medical history includes|past medical history)\s+([^,\.;]+?)(?:,|\.|\s+and|\s*$)",
                r"\b(diabetes|hypertension|asthma|copd|pneumonia|myocardial infarction|heart attack|stroke|cancer)\b"
            ],
            "medication_patterns": [
                r"(?:taking|prescribed|on|given)\s+([a-zA-Z0-9\s\-]+?)(?:\s+for|\s+to treat|,|\.|\s+daily|\s+bid|\s+tid|\s+mg|\s*$)",
                r"(?:medication|drug|therapy):\s*([a-zA-Z0-9\s\-]+?)(?:,|\.|\s+for|\s*$)",
                r"(?:treatment with|treated with)\s+([a-zA-Z0-9\s\-]+?)(?:,|\.|\s+for|\s*$)",
                r"(?:current medications|meds):\s*([a-zA-Z0-9\s\-,]+?)(?:\.|\s*$)",
                r"\b(metformin|lisinopril|atorvastatin|aspirin|insulin|warfarin|digoxin|furosemide|amlodipine)\b"
            ],
            "lab_patterns": [
                r"(troponin|glucose|creatinine|hemoglobin|hba1c|cholesterol|bun|sodium|potassium)\s+(?:is|was|level|value|result)?\s*(elevated|high|low|normal|abnormal|\d+[\.\d]*)",
                r"(?:lab shows|results show|bloodwork shows)\s+([^,\.;]+?)(?:,|\.|\s*$)",
                r"(troponin|glucose|creatinine|hemoglobin|hba1c)\s*:?\s*(\d+[\.\d]*\s*[A-Za-z/%]*)",
                r"(?:laboratory|lab)\s+(?:results|values|findings):\s*([^,\.;]+?)(?:,|\.|\s*$)",
                r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:of|at|=)\s+(\d+[\.\d]*)"
            ],
            "vital_patterns": [
                r"(?:blood pressure|BP)\s*:?\s*(\d+/\d+)",
                r"(?:heart rate|HR|pulse)\s*:?\s*(\d+)",
                r"(?:temperature|temp)\s*:?\s*(\d+[\.\d]*)",
                r"(?:respiratory rate|RR)\s*:?\s*(\d+)",
                r"(?:oxygen saturation|o2 sat|spo2)\s*:?\s*(\d+)",
                r"(?:weight)\s*:?\s*(\d+[\.\d]*\s*(?:kg|lbs|pounds)?)"
            ],
            "temporal_patterns": [
                r"(?:for the past|over the last|in the last|since)\s+(\d+\s+(?:days?|weeks?|months?|years?))",
                r"(?:acute|chronic|recent|longstanding|persistent)",
                r"(?:suddenly|gradually|progressively)",
                r"(?:onset|started|began)\s+(\d+\s+(?:hours?|days?|weeks?)\s+ago)",
                r"(?:duration|lasting)\s+(\d+\s+(?:minutes?|hours?|days?))"
            ],
            "negation_patterns": [
                r"(?:no|not|without|absent|denies|negative for|ruled out)",
                r"(?:doesn't have|does not have|hasn't|has not)",
                r"(?:unable to|cannot|can't)",
                r"(?:unremarkable|within normal limits|wnl)"
            ]
        }
        
        # Compile regex patterns
        self.compiled_patterns = {}
        for category, patterns in self.medical_patterns.items():
            self.compiled_patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def extract_predicates(self, explanation_text: str) -> List[FOLPredicate]:
        """Extract FOL predicates from medical explanation text"""
        predicates = []
        
        logger.info(f"Extracting predicates from text: {explanation_text[:100]}...")
        
        # Extract different types of predicates
        predicates.extend(self._extract_symptom_predicates(explanation_text))
        predicates.extend(self._extract_condition_predicates(explanation_text))
        predicates.extend(self._extract_medication_predicates(explanation_text))
        predicates.extend(self._extract_lab_predicates(explanation_text))
        predicates.extend(self._extract_vital_predicates(explanation_text))
        
        # Filter and deduplicate
        predicates = self._filter_predicates(predicates)
        
        logger.info(f"Extracted {len(predicates)} FOL predicates from explanation")
        return predicates
    
    def _extract_symptom_predicates(self, text: str) -> List[FOLPredicate]:
        """Extract symptom-related predicates"""
        predicates = []
        
        for pattern in self.compiled_patterns["symptom_patterns"]:
            matches = pattern.finditer(text)
            for match in matches:
                symptom_text = match.group(1).strip()
                
                # Split compound symptoms
                symptoms = self._split_compound_terms(symptom_text)
                
                for symptom in symptoms:
                    # Clean symptom text
                    symptom = self._clean_medical_term(symptom)
                    if not symptom:
                        continue
                    
                    # Check for negation
                    negation = self._check_negation(match.start(), text)
                    
                    # Extract temporal information
                    temporal = self._extract_temporal_info(symptom)
                    
                    predicate = FOLPredicate(
                        predicate_type=PredicateType.HAS_SYMPTOM,
                        subject="patient",
                        relation="has_symptom",
                        object=symptom.lower(),
                        confidence=0.8,
                        negation=negation,
                        temporal_modifier=temporal,
                        original_text=match.group(0)
                    )
                    predicates.append(predicate)
        
        return predicates
    
    def _split_compound_terms(self, term: str) -> List[str]:
        """Split compound medical terms into individual terms"""
        # Split on common conjunctions
        connectors = ['and', 'with', 'plus', 'along with', 'as well as', ',']
        
        terms = [term.strip()]
        
        for connector in connectors:
            new_terms = []
            for t in terms:
                if connector in t:
                    parts = t.split(connector)
                    new_terms.extend([p.strip() for p in parts if p.strip()])
                else:
                    new_terms.append(t)
            terms = new_terms
        
        # Filter out empty terms and single words that aren't medical terms
        medical_keywords = [
            'pain', 'ache', 'discomfort', 'difficulty', 'shortness', 'breath', 
            'nausea', 'vomiting', 'fever', 'headache', 'dizziness', 'fatigue',
            'troponin', 'glucose', 'pressure', 'rate', 'temperature'
        ]
        
        valid_terms = []
        for term in terms:
            if len(term) > 2:  # At least 3 characters
                words = term.split()
                if len(words) > 1 or any(keyword in term.lower() for keyword in medical_keywords):
                    valid_terms.append(term)
        
        return valid_terms if valid_terms else [term]
    
    def _extract_condition_predicates(self, text: str) -> List[FOLPredicate]:
        """Extract condition/diagnosis predicates"""
        predicates = []
        
        for pattern in self.compiled_patterns["condition_patterns"]:
            matches = pattern.finditer(text)
            for match in matches:
                condition_text = match.group(1).strip()
                
                # Split compound conditions
                conditions = self._split_compound_terms(condition_text)
                
                for condition in conditions:
                    condition = self._clean_medical_term(condition)
                    
                    if not condition:
                        continue
                    
                    negation = self._check_negation(match.start(), text)
                    
                    predicate = FOLPredicate(
                        predicate_type=PredicateType.HAS_CONDITION,
                        subject="patient",
                        relation="has_condition",
                        object=condition.lower(),
                        confidence=0.9,
                        negation=negation,
                        original_text=match.group(0)
                    )
                    predicates.append(predicate)
        
        return predicates
    
    def _extract_medication_predicates(self, text: str) -> List[FOLPredicate]:
        """Extract medication predicates"""
        predicates = []
        
        for pattern in self.compiled_patterns["medication_patterns"]:
            matches = pattern.finditer(text)
            for match in matches:
                medication = match.group(1).strip()
                medication = self._clean_medication_name(medication)
                
                if not medication:
                    continue
                
                negation = self._check_negation(match.start(), text)
                
                predicate = FOLPredicate(
                    predicate_type=PredicateType.TAKES_MEDICATION,
                    subject="patient",
                    relation="takes_medication",
                    object=medication.lower(),
                    confidence=0.85,
                    negation=negation,
                    original_text=match.group(0)
                )
                predicates.append(predicate)
        
        return predicates
    
    def _extract_lab_predicates(self, text: str) -> List[FOLPredicate]:
        """Extract laboratory value predicates"""
        predicates = []
        
        for pattern in self.compiled_patterns["lab_patterns"]:
            matches = pattern.finditer(text)
            for match in matches:
                if len(match.groups()) >= 2:
                    lab_name = match.group(1).strip()
                    lab_value = match.group(2).strip()
                    
                    # Clean lab name
                    lab_name = self._clean_medical_term(lab_name)
                    if not lab_name:
                        continue
                    
                    # Simplify lab name to core component
                    if 'troponin' in lab_name.lower():
                        lab_name = 'troponin'
                    elif 'glucose' in lab_name.lower():
                        lab_name = 'glucose'
                    elif 'creatinine' in lab_name.lower():
                        lab_name = 'creatinine'
                    
                    predicate = FOLPredicate(
                        predicate_type=PredicateType.HAS_LAB_VALUE,
                        subject="patient",
                        relation="has_lab_value",
                        object=f"{lab_name}:{lab_value}".lower(),
                        confidence=0.95,
                        original_text=match.group(0)
                    )
                    predicates.append(predicate)
        
        return predicates
    
    def _extract_vital_predicates(self, text: str) -> List[FOLPredicate]:
        """Extract vital sign predicates"""
        predicates = []
        
        vital_mappings = {
            r"blood pressure|BP": "blood_pressure",
            r"heart rate|HR|pulse": "heart_rate",
            r"temperature|temp": "temperature",
            r"respiratory rate|RR": "respiratory_rate",
            r"oxygen saturation|o2 sat|spo2": "oxygen_saturation",
            r"weight": "weight"
        }
        
        for pattern in self.compiled_patterns["vital_patterns"]:
            matches = pattern.finditer(text)
            for match in matches:
                vital_value = match.group(1).strip()
                
                # Determine vital type from pattern
                pattern_text = pattern.pattern
                vital_type = "vital_sign"
                
                for mapping_pattern, mapped_name in vital_mappings.items():
                    if re.search(mapping_pattern, pattern_text, re.IGNORECASE):
                        vital_type = mapped_name
                        break
                
                predicate = FOLPredicate(
                    predicate_type=PredicateType.HAS_VITAL_SIGN,
                    subject="patient",
                    relation="has_vital_sign",
                    object=f"{vital_type}:{vital_value}".lower(),
                    confidence=0.9,
                    original_text=match.group(0)
                )
                predicates.append(predicate)
        
        return predicates
    
    def _check_negation(self, position: int, text: str) -> bool:
        """Check if the predicate is negated"""
        # Look for negation within 50 characters before the match
        context = text[max(0, position-50):position]
        
        for pattern in self.compiled_patterns["negation_patterns"]:
            if pattern.search(context):
                return True
        return False
    
    def _extract_temporal_info(self, text: str) -> Optional[str]:
        """Extract temporal information from text"""
        for pattern in self.compiled_patterns["temporal_patterns"]:
            match = pattern.search(text)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        return None
    
    def _clean_medical_term(self, term: str) -> str:
        """Clean and standardize medical term"""
        if not term:
            return ""
        
        # Remove extra whitespace and punctuation
        cleaned = ' '.join(term.strip().split())
        
        # Remove common non-medical words
        stop_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = cleaned.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        # Remove trailing punctuation and dosages for symptoms/conditions
        result = ' '.join(filtered_words).strip()
        result = re.sub(r'[,;\.!?\s]+$', '', result)
        
        return result if len(result) > 2 else ""
    
    def _clean_medication_name(self, medication: str) -> str:
        """Clean medication name specifically"""
        if not medication:
            return ""
        
        # Remove dosage information for medication name extraction
        cleaned = re.sub(r'\d+\s*(?:mg|mcg|g|ml|units?|iu)\b', '', medication, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b(?:daily|bid|tid|qid|prn|as needed|once|twice|three times)\b', '', cleaned, flags=re.IGNORECASE)
        
        return self._clean_medical_term(cleaned)
    
    def _filter_predicates(self, predicates: List[FOLPredicate]) -> List[FOLPredicate]:
        """Filter and deduplicate predicates"""
        # Remove duplicates based on FOL string
        seen = set()
        filtered = []
        
        # Quality filters for objects
        invalid_objects = {
            'patient', 'patient presents', 'acute chest', 'chest', 
            'patient has', 'he', 'she', 'they', 'and', 'with'
        }
        
        for predicate in predicates:
            fol_key = predicate.to_fol_string()
            object_clean = predicate.object.lower().strip()
            
            # Skip if already seen
            if fol_key in seen:
                continue
                
            # Skip low-quality objects
            if object_clean in invalid_objects:
                continue
                
            # Skip objects that are too short or too generic
            if len(object_clean) < 3:
                continue
                
            # Skip objects that are mostly non-alphabetic
            if not any(c.isalpha() for c in object_clean):
                continue
                
            # Skip if object doesn't contain meaningful medical content
            if not self._is_meaningful_medical_term(object_clean):
                continue
            
            seen.add(fol_key)
            filtered.append(predicate)
        
        # Sort by confidence
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered
    
    def _is_meaningful_medical_term(self, term: str) -> bool:
        """Check if term represents meaningful medical content"""
        medical_keywords = {
            'pain', 'ache', 'discomfort', 'difficulty', 'shortness', 'breath', 
            'nausea', 'vomiting', 'fever', 'headache', 'dizziness', 'fatigue',
            'troponin', 'glucose', 'pressure', 'rate', 'temperature', 'elevated',
            'high', 'low', 'normal', 'syndrome', 'disease', 'condition', 'infection',
            'diabetes', 'hypertension', 'asthma', 'pneumonia', 'heart', 'cardiac',
            'pulmonary', 'renal', 'hepatic', 'acute', 'chronic', 'severe', 'mild'
        }
        
        # Check if term contains medical keywords
        term_words = set(term.lower().split())
        if term_words.intersection(medical_keywords):
            return True
            
        # Check for medical patterns
        medical_patterns = [
            r'\b\w+itis\b',  # inflammation conditions
            r'\b\w+osis\b',  # pathological conditions
            r'\b\w+emia\b',  # blood conditions
            r'\b\w+uria\b',  # urine conditions
            r'\bmg/dl\b', r'\bng/ml\b', r'\bmcg\b'  # units
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, term, re.IGNORECASE):
                return True
        
        # Must have at least 2 meaningful words or be a compound medical term
        meaningful_words = [w for w in term_words if len(w) > 2 and w not in {'the', 'and', 'or', 'but', 'for', 'with'}]
        return len(meaningful_words) >= 1
