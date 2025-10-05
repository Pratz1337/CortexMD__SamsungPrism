"""
Advanced FOL Extractor using AI and Medical NLP
Supports dynamic extraction for any medical condition
"""

import re
import json
import asyncio
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import requests
from datetime import datetime

# Try to import medical NLP libraries
try:
    import spacy
    from spacy import displacy
    HAS_SPACY = True
except ImportError:
    spacy = None
    HAS_SPACY = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
    from transformers import pipeline  # type: ignore
    HAS_TRANSFORMERS = True
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModel = None
    AutoModelForTokenClassification = None
    pipeline = None
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)

class MedicalEntityType(Enum):
    SYMPTOM = "symptom"
    CONDITION = "condition"
    MEDICATION = "medication"
    LAB_TEST = "lab_test"
    LAB_VALUE = "lab_value"
    VITAL_SIGN = "vital_sign"
    PROCEDURE = "procedure"
    ANATOMY = "anatomy"
    TEMPORAL = "temporal"
    SEVERITY = "severity"
    NEGATION = "negation"

@dataclass
class MedicalEntity:
    text: str
    entity_type: MedicalEntityType
    start_pos: int
    end_pos: int
    confidence: float
    normalized_form: Optional[str] = None
    umls_cui: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

@dataclass
class AdvancedFOLPredicate:
    subject: str
    predicate: str
    object: str
    confidence: float
    evidence_entities: List[MedicalEntity]
    temporal_context: Optional[str] = None
    negation: bool = False
    semantic_embedding: Optional[List[float]] = None
    medical_reasoning: str = ""

    def to_fol_string(self) -> str:
        negation_prefix = "¬" if self.negation else ""
        temporal_suffix = f"@{self.temporal_context}" if self.temporal_context else ""
        return f"{negation_prefix}{self.predicate}({self.subject}, {self.object}){temporal_suffix}"

    def to_dict(self) -> Dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "negation": self.negation,
            "temporal_context": self.temporal_context,
            "medical_reasoning": self.medical_reasoning,
            "fol_string": self.to_fol_string(),
            "evidence_entities": [asdict(entity) for entity in self.evidence_entities]
        }

class EnhancedMedicalNER:
    """Enhanced Medical Named Entity Recognition with advanced NLP"""

    def __init__(self):
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize available medical NLP models with enhanced capabilities"""

        # Try to load medical models
        if HAS_SPACY and spacy is not None:
            try:
                # Try to load medical spaCy models with dependency parsing
                model_names = ["en_core_sci_sm", "en_core_sci_md", "en_ner_bc5cdr_md"]
                for model_name in model_names:
                    try:
                        self.models[model_name] = spacy.load(model_name)
                        logger.info(f"Loaded spaCy model: {model_name}")
                        break
                    except OSError:
                        continue

                # Fallback to general English model
                if not self.models:
                    self.models["en_core_web_sm"] = spacy.load("en_core_web_sm")
                    logger.info("Loaded general spaCy model")

            except Exception as e:
                logger.warning(f"Could not load spaCy models: {e}")

        if HAS_TRANSFORMERS and pipeline is not None:
            try:
                # Load medical BERT models with enhanced capabilities
                self.models["medical_ner"] = pipeline(
                    "ner",
                    model="d4data/biomedical-ner-all",
                    aggregation_strategy="simple"
                )
                logger.info("Loaded medical BERT NER model")

                # Load clinical BERT for better medical understanding
                self.models["clinical_bert"] = pipeline(
                    "fill-mask",
                    model="emilyalsentzer/Bio_ClinicalBERT"
                )
                logger.info("Loaded clinical BERT model")

            except Exception as e:
                logger.warning(f"Could not load medical BERT: {e}")

                # Fallback to general NER
                try:
                    self.models["general_ner"] = pipeline("ner", aggregation_strategy="simple")
                    logger.info("Loaded general NER model")
                except Exception as e:
                    logger.warning(f"Could not load any NER model: {e}")

    def extract_medical_entities_advanced(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities using advanced NLP techniques"""
        entities = []

        # Use multiple extraction methods for comprehensive coverage
        entities.extend(self._extract_with_dependency_parsing(text))
        entities.extend(self._extract_with_transformer_models(text))
        entities.extend(self._extract_with_contextual_patterns(text))

        return self._deduplicate_and_rank_entities(entities)

    def _extract_with_dependency_parsing(self, text: str) -> List[MedicalEntity]:
        """Extract entities using dependency parsing for better context understanding"""
        entities = []

        if not self.models:
            return entities

        # Use spaCy for dependency parsing
        model_name = next((name for name in self.models.keys() if "sci" in name or "web" in name), None)
        if not model_name:
            return entities

        try:
            nlp = self.models[model_name]
            doc = nlp(text)

            for token in doc:
                # Extract symptoms based on dependency relations
                if self._is_symptom_token(token, doc):
                    entity = MedicalEntity(
                        text=token.text,
                        entity_type=MedicalEntityType.SYMPTOM,
                        start_pos=token.idx,
                        end_pos=token.idx + len(token.text),
                        confidence=0.85
                    )
                    entities.append(entity)

                # Extract conditions with contextual understanding
                elif self._is_condition_token(token, doc):
                    entity = MedicalEntity(
                        text=token.text,
                        entity_type=MedicalEntityType.CONDITION,
                        start_pos=token.idx,
                        end_pos=token.idx + len(token.text),
                        confidence=0.9
                    )
                    entities.append(entity)

                # Extract medications with dosage information
                elif self._is_medication_token(token, doc):
                    # Include dosage if available
                    medication_text = self._extract_medication_with_dosage(token, doc)
                    entity = MedicalEntity(
                        text=medication_text,
                        entity_type=MedicalEntityType.MEDICATION,
                        start_pos=token.idx,
                        end_pos=token.idx + len(medication_text),
                        confidence=0.9
                    )
                    entities.append(entity)

            return entities
        except Exception as e:
            logger.error(f"Dependency parsing extraction failed: {e}")
            return []

    def _is_symptom_token(self, token, doc) -> bool:
        """Determine if token represents a symptom using dependency parsing"""
        symptom_indicators = [
            "pain", "ache", "discomfort", "difficulty", "shortness", "nausea",
            "vomiting", "fever", "headache", "dizziness", "fatigue", "weakness"
        ]

        # Check direct symptom words
        if token.lemma_.lower() in symptom_indicators:
            return True

        # Check dependency relations for symptom indicators
        for child in token.children:
            if child.dep_ in ["amod", "compound", "nmod"] and child.lemma_.lower() in symptom_indicators:
                return True

        # Check parent relations
        if token.dep_ in ["amod", "compound", "nmod"] and token.head.lemma_.lower() in symptom_indicators:
            return True

        return False

    def _is_condition_token(self, token, doc) -> bool:
        """Determine if token represents a condition using dependency parsing"""
        condition_indicators = [
            "diabetes", "hypertension", "asthma", "pneumonia", "infection",
            "cancer", "disease", "disorder", "syndrome", "inflammation"
        ]

        if token.lemma_.lower() in condition_indicators:
            return True

        # Check for medical suffixes
        if token.text.lower().endswith(("itis", "osis", "emia", "opathy", "algia", "oma")):
            return True

        return False

    def _is_medication_token(self, token, doc) -> bool:
        """Determine if token represents a medication using dependency parsing"""
        medication_indicators = [
            "aspirin", "metformin", "lisinopril", "insulin", "amoxicillin",
            "prednisone", "warfarin", "heparin", "atorvastatin"
        ]

        if token.lemma_.lower() in medication_indicators:
            return True

        # Check for medication suffixes
        if token.text.lower().endswith(("cillin", "mycin", "floxacin", "cycline")):
            return True

        return False

    def _extract_medication_with_dosage(self, token, doc) -> str:
        """Extract medication name with dosage information"""
        medication_text = token.text

        # Look for dosage information in nearby tokens
        for child in token.children:
            if child.dep_ in ["nummod", "quantmod"] or child.like_num:
                medication_text += f" {child.text}"

        # Look for units (mg, ml, etc.)
        for sibling in token.head.children:
            if sibling.text.lower() in ["mg", "ml", "g", "mcg", "units", "iu"]:
                medication_text += f" {sibling.text}"

        return medication_text

    def _extract_with_transformer_models(self, text: str) -> List[MedicalEntity]:
        """Extract entities using transformer-based models"""
        entities = []

        if "medical_ner" in self.models:
            try:
                results = self.models["medical_ner"](text)
                for result in results:
                    entity_type = self._map_bert_label_to_medical_type(result["entity_group"])
                    entity = MedicalEntity(
                        text=result["word"],
                        entity_type=entity_type,
                        start_pos=result["start"],
                        end_pos=result["end"],
                        confidence=result["score"]
                    )
                    entities.append(entity)
            except Exception as e:
                logger.error(f"Medical BERT extraction failed: {e}")

        return entities

    def _extract_with_contextual_patterns(self, text: str) -> List[MedicalEntity]:
        """Extract entities using enhanced contextual patterns"""
        entities = []

        # Enhanced medical patterns with context awareness
        contextual_patterns = {
            MedicalEntityType.SYMPTOM: [
                r'\b(?:severe|mild|moderate|acute|chronic)\s+(\w+)\s+(?:pain|ache|discomfort)\b',
                r'\b(?:experiencing|complaining of|suffering from)\s+([^,\.;]+?)(?:\s+for|\s+since|,|\.|\s*$)',
                r'\b(?:reports|states)\s+(?:having|experiencing)\s+([^,\.;]+?)(?:\s+for|\s+since|,|\.|\s*$)'
            ],
            MedicalEntityType.CONDITION: [
                r'\b(?:diagnosed with|diagnosis of|history of)\s+([^,\.;]+?)(?:\s+for|\s+since|,|\.|\s*$)',
                r'\b(?:suffers from|has been diagnosed with)\s+([^,\.;]+?)(?:\s+for|\s+since|,|\.|\s*$)'
            ],
            MedicalEntityType.MEDICATION: [
                r'\b(?:taking|prescribed|started on)\s+([a-zA-Z0-9\s\-]+?)(?:\s+(\d+(?:\.\d+)?)\s*(?:mg|ml|g|mcg))?(?:\s+for|\s+to treat|,|\.|\s+daily|\s+bid|\s+tid|\s*$)',
                r'\b(?:medication|drug|therapy):\s*([a-zA-Z0-9\s\-]+?)(?:\s+(\d+(?:\.\d+)?)\s*(?:mg|ml|g|mcg))?(?:,|\.|\s+for|\s*$)'
            ],
            MedicalEntityType.LAB_VALUE: [
                r'\b(troponin|glucose|creatinine|hemoglobin|hba1c)\s+(?:level|value)?\s*(?:is|was|measured|=)?\s*(\d+(?:\.\d+)?)\s*(?:ng/ml|mg/dl|g/dl|%|mmol/l)?\b',
                r'\b(?:lab|blood)\s+(?:shows|reveals|indicates)\s+([^,\.;]+?)(?:\s+for|\s+since|,|\.|\s*$)'
            ],
            MedicalEntityType.VITAL_SIGN: [
                r'\b(?:blood pressure|BP)\s*(?:is|was|=)?\s*(\d+/\d+)\s*(?:mmHg)?\b',
                r'\b(?:heart rate|HR|pulse)\s*(?:is|was|=)?\s*(\d+)\s*(?:bpm)?\b',
                r'\b(?:temperature|temp)\s*(?:is|was|=)?\s*(\d+(?:\.\d+)?)\s*(?:°F|°C)?\b'
            ]
        }

        for entity_type, patterns in contextual_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 1:
                        entity_text = match.group(1).strip()
                        if len(entity_text) > 2:  # Filter out very short matches
                            entity = MedicalEntity(
                                text=entity_text,
                                entity_type=entity_type,
                                start_pos=match.start(1),
                                end_pos=match.end(1),
                                confidence=0.8
                            )
                            entities.append(entity)

        return entities

    def _map_bert_label_to_medical_type(self, label: str) -> MedicalEntityType:
        """Map BERT entity labels to our medical types"""
        label_mapping = {
            "DISEASE": MedicalEntityType.CONDITION,
            "CHEMICAL": MedicalEntityType.MEDICATION,
            "SYMPTOM": MedicalEntityType.SYMPTOM,
            "ANATOMY": MedicalEntityType.ANATOMY,
            "PROCEDURE": MedicalEntityType.PROCEDURE
        }
        return label_mapping.get(label.upper(), MedicalEntityType.CONDITION)

    def _deduplicate_and_rank_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove duplicate entities and rank by confidence"""
        seen = set()
        unique_entities = []

        for entity in entities:
            key = (entity.text.lower(), entity.entity_type, entity.start_pos)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        # Sort by confidence
        unique_entities.sort(key=lambda x: x.confidence, reverse=True)

        return unique_entities

class ContextualPredicateGenerator:
    """Enhanced predicate generation with contextual understanding"""

    def __init__(self):
        self.medical_knowledge_base = self._load_medical_knowledge()
        self.negation_detector = AdvancedNegationDetector()
        self.temporal_extractor = TemporalContextExtractor()
        self.ambiguity_resolver = MedicalAmbiguityResolver()

    def _load_medical_knowledge(self) -> Dict:
        """Load comprehensive medical knowledge for enhanced predicate generation"""
        return {
            "symptom_condition_relations": {
                "chest_pain": ["myocardial_infarction", "angina", "pericarditis", "pulmonary_embolism"],
                "shortness_of_breath": ["heart_failure", "asthma", "copd", "pneumonia", "pulmonary_embolism"],
                "fever": ["infection", "pneumonia", "sepsis", "viral_illness", "bacterial_infection"],
                "headache": ["migraine", "tension_headache", "cluster_headache", "medication_overuse"],
                "abdominal_pain": ["appendicitis", "cholecystitis", "pancreatitis", "bowel_obstruction"],
                "joint_pain": ["rheumatoid_arthritis", "osteoarthritis", "gout", "systemic_lupus"],
                "back_pain": ["musculoskeletal_pain", "herniated_disc", "kidney_stones", "osteoporosis"]
            },
            "lab_condition_indicators": {
                "elevated_troponin": ["myocardial_infarction", "myocarditis", "cardiac_contusion"],
                "elevated_ck_mb": ["myocardial_infarction", "myocarditis", "rhabdomyolysis"],
                "elevated_glucose": ["diabetes_mellitus", "hyperglycemia", "stress_response"],
                "low_hemoglobin": ["anemia", "blood_loss", "bone_marrow_failure", "chronic_disease"],
                "elevated_creatinine": ["acute_kidney_injury", "chronic_kidney_disease", "rhabdomyolysis"],
                "elevated_bilirubin": ["liver_disease", "hemolysis", "biliary_obstruction"],
                "elevated_crp": ["inflammation", "infection", "autoimmune_disease"],
                "elevated_esr": ["inflammation", "infection", "malignancy", "autoimmune_disease"]
            },
            "medication_condition_relations": {
                "aspirin": ["coronary_artery_disease", "stroke_prevention", "pain_relief"],
                "metformin": ["diabetes_mellitus", "polycystic_ovary_syndrome"],
                "lisinopril": ["hypertension", "heart_failure", "diabetic_nephropathy"],
                "atorvastatin": ["hyperlipidemia", "coronary_artery_disease", "stroke_prevention"],
                "warfarin": ["atrial_fibrillation", "deep_vein_thrombosis", "pulmonary_embolism"],
                "insulin": ["diabetes_mellitus", "diabetic_ketoacidosis", "hyperglycemia"],
                "prednisone": ["inflammation", "autoimmune_disease", "asthma_exacerbation"],
                "amoxicillin": ["bacterial_infection", "otitis_media", "urinary_tract_infection"]
            },
            "temporal_patterns": {
                "acute": ["sudden_onset", "recent_development", "immediate_attention"],
                "chronic": ["long_standing", "persistent", "stable_management"],
                "intermittent": ["episodic", "recurrent", "variable_presentation"],
                "progressive": ["worsening", "deteriorating", "escalating_severity"]
            }
        }

    def convert_term_to_predicate_enhanced(
        self,
        term: str,
        entity_type: MedicalEntityType,
        context: str,
        surrounding_text: str
    ) -> AdvancedFOLPredicate:
        """Enhanced term-to-predicate conversion with comprehensive context awareness"""

        # Step 1: Resolve ambiguity in medical terms
        resolved_term = self.ambiguity_resolver.resolve_ambiguity(term, context, surrounding_text)

        # Step 2: Detect negation with advanced context
        negation = self.negation_detector.detect_negation_enhanced(term, surrounding_text)

        # Step 3: Extract temporal context
        temporal_context = self.temporal_extractor.extract_temporal_context(surrounding_text)

        # Step 4: Generate predicate based on entity type and context
        predicate_data = self._generate_predicate_with_context(
            resolved_term, entity_type, context, surrounding_text
        )

        # Step 5: Create evidence entities
        evidence_entities = self._create_evidence_entities(term, entity_type, surrounding_text)

        # Step 6: Generate medical reasoning
        medical_reasoning = self._generate_medical_reasoning(
            resolved_term, entity_type, context, negation, temporal_context
        )

        return AdvancedFOLPredicate(
            subject="patient",
            predicate=predicate_data["predicate"],
            object=predicate_data["object"],
            confidence=predicate_data["confidence"],
            evidence_entities=evidence_entities,
            temporal_context=temporal_context,
            negation=negation,
            medical_reasoning=medical_reasoning
        )

    def _generate_predicate_with_context(
        self,
        term: str,
        entity_type: MedicalEntityType,
        context: str,
        surrounding_text: str
    ) -> Dict[str, Any]:
        """Generate predicate with rich contextual understanding"""

        if entity_type == MedicalEntityType.SYMPTOM:
            return self._generate_symptom_predicate(term, context, surrounding_text)
        elif entity_type == MedicalEntityType.CONDITION:
            return self._generate_condition_predicate(term, context, surrounding_text)
        elif entity_type == MedicalEntityType.MEDICATION:
            return self._generate_medication_predicate(term, context, surrounding_text)
        elif entity_type == MedicalEntityType.LAB_VALUE:
            return self._generate_lab_predicate(term, context, surrounding_text)
        elif entity_type == MedicalEntityType.VITAL_SIGN:
            return self._generate_vital_sign_predicate(term, context, surrounding_text)
        else:
            return self._generate_generic_predicate(term, entity_type, context)

    def _generate_symptom_predicate(self, symptom: str, context: str, surrounding_text: str) -> Dict[str, Any]:
        """Generate symptom predicate with clinical context"""
        symptom_lower = symptom.lower()

        # Determine severity from context
        severity = self._extract_severity_from_context(surrounding_text)

        # Determine symptom type and implications
        symptom_type = self._classify_symptom_type(symptom_lower)

        # Generate contextual object
        if severity:
            object_text = f"{severity}_{symptom_lower}"
        else:
            object_text = symptom_lower

        # Add anatomical location if available
        location = self._extract_anatomical_location(surrounding_text, symptom_lower)
        if location:
            object_text = f"{object_text}_{location}"

        # Determine confidence based on context clarity
        confidence = self._calculate_contextual_confidence(surrounding_text, symptom_lower)

        return {
            "predicate": "has_symptom",
            "object": object_text,
            "confidence": confidence
        }

    def _generate_condition_predicate(self, condition: str, context: str, surrounding_text: str) -> Dict[str, Any]:
        """Generate condition predicate with diagnostic context"""
        condition_lower = condition.lower()

        # Check if this is a primary or secondary condition
        condition_type = self._determine_condition_type(condition_lower, surrounding_text)

        # Add temporal qualifier if available
        temporal_qualifier = self._extract_condition_temporal_qualifier(surrounding_text)

        object_text = condition_lower
        if temporal_qualifier:
            object_text = f"{temporal_qualifier}_{condition_lower}"

        # Higher confidence for explicitly diagnosed conditions
        if any(word in surrounding_text.lower() for word in ["diagnosed", "diagnosis", "confirmed"]):
            confidence = 0.95
        else:
            confidence = 0.85

        return {
            "predicate": "has_condition",
            "object": object_text,
            "confidence": confidence
        }

    def _generate_medication_predicate(self, medication: str, context: str, surrounding_text: str) -> Dict[str, Any]:
        """Generate medication predicate with dosage and purpose context"""
        medication_lower = medication.lower()

        # Extract dosage information
        dosage = self._extract_dosage_from_context(surrounding_text, medication_lower)

        # Extract purpose/indication
        purpose = self._extract_medication_purpose(surrounding_text, medication_lower)

        # Build object text
        object_text = medication_lower
        if dosage:
            object_text = f"{medication_lower}_{dosage}"

        # Add purpose context to reasoning rather than object for cleaner predicates
        confidence = 0.9 if dosage else 0.8  # Higher confidence with dosage info

        return {
            "predicate": "takes_medication",
            "object": object_text,
            "confidence": confidence
        }

    def _generate_lab_predicate(self, lab_term: str, context: str, surrounding_text: str) -> Dict[str, Any]:
        """Generate lab value predicate with enhanced interpretation"""
        lab_lower = lab_term.lower()

        # Enhanced lab value interpretation
        lab_interpretation = self._infer_lab_value_from_context_enhanced(lab_term, surrounding_text)

        # Determine if this is a specific value or a general result
        if lab_interpretation["is_numeric"]:
            object_text = f"{lab_lower}:{lab_interpretation['value']}"
            confidence = 0.95
        else:
            object_text = f"{lab_lower}:{lab_interpretation['interpretation']}"
            confidence = 0.9

        return {
            "predicate": "has_lab_value",
            "object": object_text,
            "confidence": confidence
        }

    def _generate_vital_sign_predicate(self, vital_term: str, context: str, surrounding_text: str) -> Dict[str, Any]:
        """Generate vital sign predicate with clinical interpretation"""
        vital_lower = vital_term.lower()

        # Extract the actual value
        vital_value = self._extract_vital_value_from_context(surrounding_text, vital_lower)

        # Determine vital sign type
        vital_type = self._map_vital_term_to_type(vital_lower)

        if vital_value:
            object_text = f"{vital_type}:{vital_value}"
            confidence = 0.95
        else:
            object_text = vital_type
            confidence = 0.8

        return {
            "predicate": "has_vital_sign",
            "object": object_text,
            "confidence": confidence
        }

    def _generate_generic_predicate(self, term: str, entity_type: MedicalEntityType, context: str) -> Dict[str, Any]:
        """Generate generic predicate for other entity types"""
        term_lower = term.lower()

        predicate_mapping = {
            MedicalEntityType.PROCEDURE: "has_procedure",
            MedicalEntityType.ANATOMY: "has_anatomical_finding",
            MedicalEntityType.SEVERITY: "has_severity_level"
        }

        predicate = predicate_mapping.get(entity_type, "has_medical_entity")

        return {
            "predicate": predicate,
            "object": term_lower,
            "confidence": 0.7
        }

    def _extract_severity_from_context(self, text: str) -> Optional[str]:
        """Extract severity modifiers from surrounding text"""
        severity_indicators = {
            "severe": "severe",
            "mild": "mild",
            "moderate": "moderate",
            "extreme": "extreme",
            "intense": "intense",
            "slight": "slight",
            "minimal": "minimal"
        }

        text_lower = text.lower()
        for indicator, severity in severity_indicators.items():
            if indicator in text_lower:
                return severity

        return None

    def _extract_anatomical_location(self, text: str, symptom: str) -> Optional[str]:
        """Extract anatomical location for symptoms"""
        location_patterns = [
            r'\b(left|right)\s+(\w+)\b',
            r'\b(\w+)\s+(pain|discomfort|ache)\s+in\s+(\w+)\b',
            r'\b(\w+)\s+(region|area|side)\b'
        ]

        for pattern in location_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                location = match.group(1) if len(match.groups()) >= 1 else match.group(0)
                return location.lower()

        return None

    def _calculate_contextual_confidence(self, text: str, term: str) -> float:
        """Calculate confidence based on contextual clarity"""
        base_confidence = 0.8

        # Increase confidence for explicit medical statements
        explicit_indicators = ["reports", "complains", "presents with", "experiencing", "diagnosed"]
        if any(indicator in text.lower() for indicator in explicit_indicators):
            base_confidence += 0.1

        # Decrease confidence for uncertain language
        uncertain_indicators = ["possible", "probable", "likely", "suspected", "may have"]
        if any(indicator in text.lower() for indicator in uncertain_indicators):
            base_confidence -= 0.1

        result = max(0.5, min(0.95, base_confidence))
        # Ensure we return a float, not a tuple
        if isinstance(result, tuple):
            result = float(result[0]) if result else 0.8
        return float(result)

    def _infer_lab_value_from_context_enhanced(self, lab_term: str, context: str) -> Dict[str, Any]:
        """Enhanced lab value interpretation with comprehensive context analysis"""
        context_lower = context.lower()
        lab_lower = lab_term.lower()

        # Extract numeric values with units
        numeric_pattern = r'(\d+(?:\.\d+)?)\s*(?:ng/ml|mg/dl|g/dl|%|mmol/l|iu|units|mmHg|bpm|°F|°C)?'
        numeric_match = re.search(numeric_pattern, context)

        if numeric_match:
            value = numeric_match.group(1)
            unit = numeric_match.group(2) if numeric_match.groups() > 1 and numeric_match.group(2) else ""

            return {
                "is_numeric": True,
                "value": f"{value}{unit}".strip(),
                "interpretation": self._interpret_lab_value(lab_lower, float(value), unit)
            }

        # Interpret qualitative descriptions
        qualitative_interpretations = {
            "elevated": "elevated",
            "high": "elevated",
            "increased": "elevated",
            "raised": "elevated",
            "low": "low",
            "decreased": "low",
            "reduced": "low",
            "normal": "normal",
            "within normal limits": "normal",
            "wnl": "normal",
            "abnormal": "abnormal",
            "positive": "positive",
            "negative": "negative"
        }

        for desc, interp in qualitative_interpretations.items():
            if desc in context_lower:
                return {
                    "is_numeric": False,
                    "value": None,
                    "interpretation": interp
                }

        # Default interpretation based on lab type
        default_interp = self._get_default_lab_interpretation(lab_lower)
        return {
            "is_numeric": False,
            "value": None,
            "interpretation": default_interp
        }

    def _interpret_lab_value(self, lab_name: str, value: float, unit: str) -> str:
        """Interpret numeric lab value based on normal ranges"""
        # This would be expanded with actual medical reference ranges
        lab_ranges = {
            "troponin": (0, 0.04),  # ng/ml
            "glucose": (70, 140),   # mg/dl
            "creatinine": (0.6, 1.2),  # mg/dl
            "hemoglobin": (12, 16),   # g/dl for women
            "wbc": (4.5, 11),         # K/uL
            "platelets": (150, 450)   # K/uL
        }

        if lab_name in lab_ranges:
            min_val, max_val = lab_ranges[lab_name]
            if value < min_val:
                return "low"
            elif value > max_val:
                return "elevated"
            else:
                return "normal"

        return "normal"  # Default assumption

    def _get_default_lab_interpretation(self, lab_name: str) -> str:
        """Get default interpretation for lab tests"""
        # Based on common clinical expectations
        if any(word in lab_name for word in ["troponin", "ck", "ldh"]):
            return "elevated"  # Cardiac markers often elevated in disease
        elif any(word in lab_name for word in ["glucose", "hba1c"]):
            return "elevated"  # Diabetes markers
        elif any(word in lab_name for word in ["hemoglobin", "hematocrit"]):
            return "normal"  # Often normal unless specified
        else:
            return "normal"

    def _create_evidence_entities(self, term: str, entity_type: MedicalEntityType, text: str) -> List[MedicalEntity]:
        """Create evidence entities for the predicate"""
        return [MedicalEntity(
            text=term,
            entity_type=entity_type,
            start_pos=text.find(term),
            end_pos=text.find(term) + len(term),
            confidence=0.9
        )]

    def _generate_medical_reasoning(self, term: str, entity_type: MedicalEntityType, context: str,
                                 negation: bool, temporal_context: Optional[str]) -> str:
        """Generate medical reasoning for the predicate"""
        reasoning_parts = []

        if negation:
            reasoning_parts.append(f"Patient does not exhibit {term}")
        else:
            reasoning_parts.append(f"Patient exhibits {term}")

        if temporal_context:
            reasoning_parts.append(f"with {temporal_context} temporal context")

        if entity_type == MedicalEntityType.SYMPTOM:
            reasoning_parts.append("based on reported symptoms")
        elif entity_type == MedicalEntityType.CONDITION:
            reasoning_parts.append("based on clinical assessment")
        elif entity_type == MedicalEntityType.MEDICATION:
            reasoning_parts.append("based on medication history")
        elif entity_type == MedicalEntityType.LAB_VALUE:
            reasoning_parts.append("based on laboratory results")

        return ". ".join(reasoning_parts)

    def _classify_symptom_type(self, symptom: str) -> Optional[str]:
        """Classify the type of symptom for clinical reasoning"""
        symptom_categories = {
            "cardiac": ["chest", "heart", "cardiac", "angina"],
            "respiratory": ["breath", "cough", "wheeze", "sob", "dyspnea"],
            "gastrointestinal": ["nausea", "vomiting", "abdominal", "diarrhea"],
            "neurological": ["headache", "dizziness", "confusion", "weakness"],
            "musculoskeletal": ["pain", "joint", "muscle", "back"]
        }

        for category, keywords in symptom_categories.items():
            if any(keyword in symptom for keyword in keywords):
                return category

        return None

    def _determine_condition_type(self, condition: str, text: str) -> str:
        """Determine if condition is primary or secondary"""
        if any(word in text.lower() for word in ["primary", "main", "principal"]):
            return "primary"
        elif any(word in text.lower() for word in ["secondary", "complication", "due to"]):
            return "secondary"
        else:
            return "unspecified"

    def _extract_condition_temporal_qualifier(self, text: str) -> Optional[str]:
        """Extract temporal qualifier for conditions"""
        temporal_qualifiers = ["acute", "chronic", "subacute", "recurrent", "persistent"]
        text_lower = text.lower()

        for qualifier in temporal_qualifiers:
            if qualifier in text_lower:
                return qualifier

        return None

    def _extract_dosage_from_context(self, text: str, medication: str) -> Optional[str]:
        """Extract dosage information from context"""
        dosage_patterns = [
            r'(\d+(?:\.\d+)?)\s*(mg|ml|g|mcg|units?|iu)\s*(?:daily|bid|tid|qid)?',
            r'(\d+(?:\.\d+)?)\s*(?:mg|ml|g|mcg)\s+per\s+day',
            r'(\d+(?:\.\d+)?)\s*(?:tablets?|capsules?)\s+(?:daily|bid|tid)'
        ]

        for pattern in dosage_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{match.group(1)} {match.group(2)}".strip()

        return None

    def _extract_medication_purpose(self, text: str, medication: str) -> Optional[str]:
        """Extract the purpose/indication for medication use"""
        purpose_patterns = [
            r'for\s+([^,\.;]+?)(?:\s+and|\s+with|,|\.|\s*$)',
            r'to\s+treat\s+([^,\.;]+?)(?:\s+and|\s+with|,|\.|\s*$)',
            r'indicated\s+for\s+([^,\.;]+?)(?:\s+and|\s+with|,|\.|\s*$)'
        ]

        for pattern in purpose_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                purpose = match.group(1).strip()
                if len(purpose) > 3:  # Filter out very short purposes
                    return purpose

        return None

    def _extract_vital_value_from_context(self, text: str, vital_term: str) -> Optional[str]:
        """Extract actual vital sign value from context"""
        if "blood pressure" in vital_term.lower() or "bp" in vital_term.lower():
            bp_pattern = r'(\d+/\d+)\s*(?:mmHg)?'
            match = re.search(bp_pattern, text)
            if match:
                return match.group(1)
        elif "heart rate" in vital_term.lower() or "hr" in vital_term.lower() or "pulse" in vital_term.lower():
            hr_pattern = r'(\d+)\s*(?:bpm|beats per minute)?'
            match = re.search(hr_pattern, text)
            if match:
                return f"{match.group(1)} bpm"
        elif "temperature" in vital_term.lower() or "temp" in vital_term.lower():
            temp_pattern = r'(\d+(?:\.\d+)?)\s*(?:°F|°C|degrees?)?'
            match = re.search(temp_pattern, text)
            if match:
                unit = match.group(2) if match.groups() > 1 and match.group(2) else "°F"
                return f"{match.group(1)} {unit}"

        return None

    def _map_vital_term_to_type(self, vital_term: str) -> str:
        """Map vital sign term to standard type"""
        vital_mapping = {
            "blood pressure": "blood_pressure",
            "bp": "blood_pressure",
            "heart rate": "heart_rate",
            "hr": "heart_rate",
            "pulse": "pulse_rate",
            "temperature": "temperature",
            "temp": "temperature",
            "respiratory rate": "respiratory_rate",
            "rr": "respiratory_rate",
            "oxygen saturation": "oxygen_saturation",
            "spo2": "oxygen_saturation",
            "weight": "weight",
            "height": "height"
        }

        vital_lower = vital_term.lower()
        for key, value in vital_mapping.items():
            if key in vital_lower:
                return value

        return "vital_sign"  # Default

class AdvancedNegationDetector:
    """Advanced negation detection with contextual understanding"""

    def __init__(self):
        self.negation_patterns = self._load_negation_patterns()

    def _load_negation_patterns(self) -> Dict[str, List[str]]:
        """Load comprehensive negation patterns"""
        return {
            "direct_negations": [
                "no", "not", "never", "none", "without", "absent", "denies",
                "negative for", "ruled out", "rules out", "doesn't have",
                "does not have", "hasn't", "has not", "unable to", "cannot",
                "can't", "unremarkable", "within normal limits", "wnl"
            ],
            "prepositional_negations": [
                "no evidence of", "no signs of", "no symptoms of",
                "no history of", "no family history of", "no complaints of"
            ],
            "pseudo_negations": [
                "stopped", "discontinued", "ceased", "resolved", "cleared"
            ]
        }

    def detect_negation_enhanced(self, term: str, context: str) -> bool:
        """Enhanced negation detection with positional and contextual analysis"""
        context_lower = context.lower()
        term_lower = term.lower()

        # Step 1: Check for direct negations within reasonable distance
        negation_positions = []
        for negation in self.negation_patterns["direct_negations"]:
            pos = context_lower.find(negation)
            while pos != -1:
                negation_positions.append((pos, len(negation)))
                pos = context_lower.find(negation, pos + 1)

        # Step 2: Check for prepositional negations
        for prep_neg in self.negation_patterns["prepositional_negations"]:
            if prep_neg in context_lower:
                # Check if the term appears after the prepositional negation
                prep_pos = context_lower.find(prep_neg)
                term_pos = context_lower.find(term_lower)
                if term_pos > prep_pos and term_pos < prep_pos + len(prep_neg) + 50:
                    return True

        # Step 3: Check positional relationship with direct negations
        term_pos = context_lower.find(term_lower)
        if term_pos == -1:
            return False

        for neg_pos, neg_len in negation_positions:
            distance = abs(term_pos - neg_pos)
            # Negation affects terms within ~30 characters
            if distance <= 30:
                # Check if negation comes before the term
                if neg_pos < term_pos:
                    return True

        # Step 4: Check for pseudo-negations (terms that imply absence)
        for pseudo_neg in self.negation_patterns["pseudo_negations"]:
            if pseudo_neg in context_lower:
                pseudo_pos = context_lower.find(pseudo_neg)
                if abs(term_pos - pseudo_pos) <= 20:
                    return True

        return False

class TemporalContextExtractor:
    """Extract temporal context from medical text"""

    def __init__(self):
        self.temporal_patterns = self._load_temporal_patterns()

    def _load_temporal_patterns(self) -> Dict[str, List[str]]:
        """Load temporal expression patterns"""
        return {
            "acute": [
                r'\bacute\b', r'\bsudden\b', r'\bsuddenly\b', r'\babrupt\b',
                r'\bnew onset\b', r'\brecent onset\b', r'\bstarted suddenly\b'
            ],
            "chronic": [
                r'\bchronic\b', r'\blong-standing\b', r'\blongstanding\b',
                r'\bpersistent\b', r'\bongoing\b', r'\bcontinuing\b'
            ],
            "duration": [
                r'\bfor\s+(\d+)\s+(day|week|month|year)s?\b',
                r'\bover\s+(\d+)\s+(day|week|month|year)s?\b',
                r'\bsince\s+(\d+)\s+(day|week|month|year)s?\s+ago\b',
                r'\blast\s+(\d+)\s+(day|week|month|year)s?\b'
            ],
            "frequency": [
                r'\bintermittent\b', r'\bepisodic\b', r'\brecurrent\b',
                r'\boccasional\b', r'\bfrequent\b', r'\bconstant\b'
            ]
        }

    def extract_temporal_context(self, text: str) -> Optional[str]:
        """Extract temporal context from text"""
        text_lower = text.lower()

        # Check for acute/chronic indicators
        for temporal_type, patterns in self.temporal_patterns.items():
            if temporal_type in ["acute", "chronic", "frequency"]:
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        return temporal_type

        # Check for duration patterns
        for pattern in self.temporal_patterns["duration"]:
            match = re.search(pattern, text_lower)
            if match:
                amount = match.group(1)
                unit = match.group(2)
                return f"{amount}_{unit}_duration"

        return None

class MedicalAmbiguityResolver:
    """Resolve ambiguity in medical terms using context"""

    def __init__(self):
        self.ambiguity_map = self._load_ambiguity_map()

    def _load_ambiguity_map(self) -> Dict[str, Dict[str, str]]:
        """Load mapping of ambiguous terms to contextual resolutions"""
        return {
            "pain": {
                "chest": "chest_pain",
                "abdominal": "abdominal_pain",
                "back": "back_pain",
                "head": "headache",
                "joint": "joint_pain",
                "muscle": "muscle_pain"
            },
            "pressure": {
                "blood": "blood_pressure",
                "intraocular": "intraocular_pressure",
                "intracranial": "intracranial_pressure"
            },
            "failure": {
                "heart": "heart_failure",
                "renal": "kidney_failure",
                "liver": "liver_failure",
                "respiratory": "respiratory_failure"
            }
        }

    def resolve_ambiguity(self, term: str, context: str, surrounding_text: str) -> str:
        """Resolve ambiguous medical terms using context"""
        term_lower = term.lower()
        full_text = f"{context} {surrounding_text}".lower()

        # Check if term has ambiguity mappings
        if term_lower in self.ambiguity_map:
            ambiguities = self.ambiguity_map[term_lower]

            # Look for contextual clues in the text
            for clue, resolved_term in ambiguities.items():
                if clue in full_text:
                    return resolved_term

        # If no specific resolution found, return original term
        return term_lower

class EnhancedFOLExtractor:
    """Enhanced FOL Extractor with advanced NLP capabilities and parallel processing"""

    def __init__(self):
        logger.info("Initializing Enhanced FOL Extractor with Advanced NLP")
        self.ner = EnhancedMedicalNER()
        self.predicate_generator = ContextualPredicateGenerator()
        self.max_workers = min(4, os.cpu_count() or 2)  # Adaptive worker count
        
        # Initialize ontology mapper for term normalization
        from services.ontology_mapper import OntologyMapper
        self.ontology_mapper = OntologyMapper()
        logger.info("✅ OntologyMapper initialized for term normalization")
        
        # Initialize patient data verifier for predicate verification
        from services.patient_data_verifier import PatientDataVerifier
        self.patient_verifier = PatientDataVerifier()
        logger.info("✅ PatientDataVerifier initialized for predicate verification")
        
        # Initialize Gemini 2.5 Flash for enhanced predicate extraction
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            logger.info("✅ Gemini 2.5 Flash initialized for enhanced FOL extraction")
        else:
            self.gemini_model = None
            logger.warning("⚠️ Gemini API key not found - using basic extraction only")
    
    def extract_medical_predicates(self, clinical_text: str) -> Dict[str, Any]:
        """
        Extract medical predicates from clinical text (SYNCHRONOUS VERSION for app.py)
        
        Args:
            clinical_text: The clinical text to analyze
            
        Returns:
            Dictionary containing predicates, entities, logic rules, and confidence scores
        """
        logger.info(f"🔬 Extracting medical predicates from clinical text ({len(clinical_text)} chars)")
        
        try:
            # Use enhanced extraction with Gemini 2.5 Flash if available
            if self.gemini_model:
                return self._extract_with_gemini_flash(clinical_text)
            else:
                return self._extract_with_basic_nlp(clinical_text)
                
        except Exception as e:
            logger.error(f"❌ Predicate extraction failed: {e}")
            return {
                'predicates': [],
                'entities': [],
                'logic_rules': [],
                'confidence_scores': {},
                'extraction_method': 'error',
                'error': str(e)
            }
    
    def _extract_with_gemini_flash(self, clinical_text: str) -> Dict[str, Any]:
        """Extract predicates using Gemini 2.5 Flash for enhanced accuracy"""
        
        prompt = f"""
You are a medical AI specialist tasked with extracting First-Order Logic (FOL) predicates from clinical text.

Clinical Text:
{clinical_text}

Please extract medical predicates in the following format. Each predicate should be a logical statement about the patient:

PREDICATES (return as valid FOL predicates):
- has_symptom(patient, symptom_name)
- has_condition(patient, condition_name)  
- has_vital_sign(patient, vital_name, value_range)
- takes_medication(patient, medication_name)
- has_lab_value(patient, lab_name, value_range)
- has_finding(patient, finding_name)

MEDICAL ENTITIES (extract key medical terms):
- Symptoms: [list symptoms]
- Conditions: [list conditions]
- Medications: [list medications]
- Lab Values: [list lab values]
- Vital Signs: [list vital signs]

LOGIC RULES (medical reasoning rules):
- If-then medical logic statements based on the clinical information

Please be precise and only extract information that is explicitly stated or strongly implied in the text.
Respond in JSON format:
{{
    "predicates": ["predicate1", "predicate2", ...],
    "entities": {{
        "symptoms": [...],
        "conditions": [...], 
        "medications": [...],
        "lab_values": [...],
        "vital_signs": [...]
    }},
    "logic_rules": ["rule1", "rule2", ...],
    "confidence_scores": {{
        "overall": 0.0-1.0,
        "predicate_accuracy": 0.0-1.0,
        "entity_extraction": 0.0-1.0
    }}
}}
"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Try to parse JSON response
            import json
            import re
            
            # Extract JSON from response (handle markdown formatting)
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Validate and clean the result
                predicates = result.get('predicates', [])
                entities = result.get('entities', {})
                logic_rules = result.get('logic_rules', [])
                confidence_scores = result.get('confidence_scores', {})
                
                # Flatten entities list for compatibility
                all_entities = []
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        all_entities.append(f"{entity_type}:{entity}")
                
                logger.info(f"✅ Gemini Flash extraction: {len(predicates)} predicates, {len(all_entities)} entities")
                
                return {
                    'predicates': predicates[:20],  # Limit to prevent overwhelming
                    'entities': all_entities[:30],  # Limit entities
                    'logic_rules': logic_rules[:10],  # Limit rules
                    'confidence_scores': confidence_scores,
                    'extraction_method': 'gemini_2.5_flash',
                    'raw_entities': entities
                }
            else:
                logger.warning("⚠️ Could not parse JSON from Gemini response, falling back to basic extraction")
                return self._extract_with_basic_nlp(clinical_text)
                
        except Exception as e:
            logger.error(f"❌ Gemini Flash extraction failed: {e}")
            return self._extract_with_basic_nlp(clinical_text)
    
    def _extract_with_basic_nlp(self, clinical_text: str) -> Dict[str, Any]:
        """Fallback basic extraction using pattern matching"""
        
        predicates = []
        entities = []
        logic_rules = []
        
        text_lower = clinical_text.lower()
        
        # Basic pattern-based extraction
        # Symptoms
        symptom_patterns = [
            r'(?:patient\s+)?(?:has|reports|complains\s+of|presents\s+with|experiencing)\s+([^,.;]+?)(?:\s+and|\s*[,.;]|\s*$)',
            r'(?:symptoms?\s+include|chief\s+complaint)\s*:?\s*([^,.;]+?)(?:\s+and|\s*[,.;]|\s*$)',
            r'(chest\s+pain|shortness\s+of\s+breath|nausea|vomiting|dizziness|headache|fever|fatigue|cough)'
        ]
        
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                symptom = match.group(1).strip() if len(match.groups()) > 0 else match.group().strip()
                if len(symptom) > 2:
                    predicates.append(f"has_symptom(patient, {symptom.replace(' ', '_')})")
                    entities.append(f"symptom:{symptom}")
        
        # Conditions
        condition_patterns = [
            r'(?:history\s+of|diagnosed\s+with|has\s+a?\s*(?:diagnosis\s+of)?)\s+([^,.;]+?)(?:\s+and|\s*[,.;]|\s*$)',
            r'(hypertension|diabetes|asthma|copd|heart\s+failure|coronary\s+artery\s+disease|myocardial\s+infarction)'
        ]
        
        for pattern in condition_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                condition = match.group(1).strip() if len(match.groups()) > 0 else match.group().strip()
                if len(condition) > 2:
                    predicates.append(f"has_condition(patient, {condition.replace(' ', '_')})")
                    entities.append(f"condition:{condition}")
        
        # Medications
        medication_patterns = [
            r'(?:taking|on|prescribed)\s+([a-zA-Z0-9\s\-]+?)(?:\s+(?:mg|mcg|daily|bid|tid)|\s*[,.;]|\s*$)',
            r'(lisinopril|metformin|aspirin|insulin|atorvastatin|amlodipine|warfarin|furosemide)'
        ]
        
        for pattern in medication_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                medication = match.group(1).strip() if len(match.groups()) > 0 else match.group().strip()
                if len(medication) > 2:
                    predicates.append(f"takes_medication(patient, {medication.replace(' ', '_')})")
                    entities.append(f"medication:{medication}")
        
        # Basic logic rules
        if 'chest pain' in text_lower and 'shortness of breath' in text_lower:
            logic_rules.append("if has_symptom(patient, chest_pain) and has_symptom(patient, shortness_of_breath) then consider_acute_coronary_syndrome")
        
        # Remove duplicates
        predicates = list(set(predicates))
        entities = list(set(entities))
        
        confidence_scores = {
            'overall': 0.6,  # Lower confidence for basic extraction
            'predicate_accuracy': 0.5,
            'entity_extraction': 0.7
        }
        
        logger.info(f"✅ Basic NLP extraction: {len(predicates)} predicates, {len(entities)} entities")
        
        return {
            'predicates': predicates,
            'entities': entities,
            'logic_rules': logic_rules,
            'confidence_scores': confidence_scores,
            'extraction_method': 'basic_nlp_patterns'
        }
    
    async def extract_and_verify_predicates(
        self, 
        clinical_text: str, 
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract FOL predicates and verify them against patient data with detailed reporting
        
        Args:
            clinical_text: The clinical explanation text
            patient_data: Patient data for verification
            
        Returns:
            Comprehensive verification report with predicate-level details
        """
        logger.info(f"🔬 Starting comprehensive FOL extraction and verification")
        
        try:
            # Step 1: Extract predicates from clinical text
            extraction_results = self.extract_medical_predicates(clinical_text)
            logger.info(f"✅ Extracted {len(extraction_results.get('predicates', []))} predicates")
            
            # Step 2: Convert predicates to structured format for verification
            structured_predicates = await self._structure_predicates_for_verification(
                extraction_results.get('predicates', []),
                clinical_text
            )
            logger.info(f"✅ Structured {len(structured_predicates)} predicates for verification")
            
            # Step 3: Verify each predicate against patient data
            verification_results = await self._verify_predicates_against_patient_data(
                structured_predicates,
                patient_data
            )
            logger.info(f"✅ Verified {len(verification_results)} predicates")
            
            # Step 4: Generate comprehensive confidence scores
            confidence_analysis = self._calculate_comprehensive_confidence(verification_results)
            
            # Step 5: Generate detailed report
            detailed_report = self._generate_detailed_verification_report(
                extraction_results,
                structured_predicates,
                verification_results,
                confidence_analysis
            )
            
            logger.info(f"🎯 FOL extraction and verification completed successfully")
            return detailed_report
            
        except Exception as e:
            logger.error(f"❌ FOL extraction and verification failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'predicates': [],
                'verification_results': [],
                'overall_confidence': 0.0,
                'success_rate': 0.0,
                'extraction_method': 'error'
            }
    
    async def _structure_predicates_for_verification(
        self,
        raw_predicates: List[str],
        clinical_text: str
    ) -> List[Dict[str, Any]]:
        """
        Convert raw predicate strings to structured format for verification
        """
        structured_predicates = []
        
        for i, predicate_str in enumerate(raw_predicates):
            try:
                # Parse predicate components
                parsed = self._parse_predicate_string(predicate_str)
                if not parsed:
                    continue
                
                # Normalize terms using ontology mapper
                normalized_object = await self._normalize_predicate_object(parsed['object'])
                
                structured_predicate = {
                    'id': f"predicate_{i+1}",
                    'original_string': predicate_str,
                    'predicate_type': parsed['predicate'],
                    'subject': parsed['subject'],
                    'object': normalized_object,
                    'original_object': parsed['object'],
                    'negation': parsed.get('negation', False),
                    'temporal_context': parsed.get('temporal_context'),
                    'confidence': parsed.get('confidence', 0.8),
                    'source_text': clinical_text[:200] + "..." if len(clinical_text) > 200 else clinical_text
                }
                
                structured_predicates.append(structured_predicate)
                
            except Exception as e:
                logger.warning(f"Failed to structure predicate '{predicate_str}': {e}")
                continue
        
        return structured_predicates
    
    def _parse_predicate_string(self, predicate_str: str) -> Optional[Dict[str, Any]]:
        """
        Parse a predicate string into components
        """
        import re
        
        # Handle negation
        negation = False
        if predicate_str.startswith('¬') or 'not ' in predicate_str.lower():
            negation = True
            predicate_str = predicate_str.lstrip('¬').replace('not ', '').strip()
        
        # Parse predicate(subject, object) format
        predicate_pattern = r'(\w+)\(([^,]+),\s*([^)]+)\)'
        match = re.match(predicate_pattern, predicate_str.strip())
        
        if match:
            predicate = match.group(1).strip()
            subject = match.group(2).strip()
            obj = match.group(3).strip()
            
            # Handle temporal context (@temporal_context)
            temporal_context = None
            if '@' in predicate_str:
                temporal_match = re.search(r'@(\w+)', predicate_str)
                if temporal_match:
                    temporal_context = temporal_match.group(1)
            
            return {
                'predicate': predicate,
                'subject': subject,
                'object': obj,
                'negation': negation,
                'temporal_context': temporal_context
            }
        
        return None
    
    async def _normalize_predicate_object(self, object_term: str) -> str:
        """
        Normalize predicate object using ontology mapper
        """
        try:
            # Clean the term
            cleaned_term = object_term.strip().lower()
            
            # Normalize using ontology mapper
            normalized_concept = await self.ontology_mapper.normalize_medical_term(cleaned_term)
            
            if normalized_concept and normalized_concept.preferred_name:
                return normalized_concept.preferred_name.lower().replace(' ', '_')
            else:
                return cleaned_term.replace(' ', '_')
                
        except Exception as e:
            logger.warning(f"Failed to normalize term '{object_term}': {e}")
            return object_term.strip().lower().replace(' ', '_')
    
    async def _verify_predicates_against_patient_data(
        self,
        structured_predicates: List[Dict[str, Any]],
        patient_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Verify each predicate against patient data with detailed analysis
        """
        verification_results = []
        
        for predicate in structured_predicates:
            try:
                # Verify based on predicate type
                result = await self._verify_single_predicate_enhanced(
                    predicate,
                    patient_data
                )
                verification_results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to verify predicate {predicate.get('id', 'unknown')}: {e}")
                # Create error result
                error_result = {
                    'predicate_id': predicate.get('id', 'unknown'),
                    'predicate_string': predicate.get('original_string', ''),
                    'verified': False,
                    'confidence_score': 0.0,
                    'supporting_evidence': [],
                    'contradicting_evidence': [f"Verification error: {str(e)}"],
                    'verification_method': 'error',
                    'reasoning': f"Failed to verify due to error: {str(e)}"
                }
                verification_results.append(error_result)
        
        return verification_results
    
    async def _verify_single_predicate_enhanced(
        self,
        predicate: Dict[str, Any],
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhanced verification of a single predicate with detailed evidence collection
        """
        predicate_type = predicate.get('predicate_type', '')
        predicate_object = predicate.get('object', '')
        original_object = predicate.get('original_object', '')
        negation = predicate.get('negation', False)
        
        supporting_evidence = []
        contradicting_evidence = []
        confidence_score = 0.0
        reasoning = ""
        
        try:
            if predicate_type == 'has_symptom':
                result = await self._verify_symptom_predicate(
                    predicate_object, original_object, patient_data, negation
                )
            elif predicate_type == 'has_condition':
                result = await self._verify_condition_predicate(
                    predicate_object, original_object, patient_data, negation
                )
            elif predicate_type == 'takes_medication':
                result = await self._verify_medication_predicate(
                    predicate_object, original_object, patient_data, negation
                )
            elif predicate_type == 'has_lab_value':
                result = await self._verify_lab_value_predicate(
                    predicate_object, original_object, patient_data, negation
                )
            elif predicate_type == 'has_vital_sign':
                result = await self._verify_vital_sign_predicate(
                    predicate_object, original_object, patient_data, negation
                )
            else:
                result = await self._verify_generic_predicate(
                    predicate_type, predicate_object, original_object, patient_data, negation
                )
            
            supporting_evidence = result.get('supporting_evidence', [])
            contradicting_evidence = result.get('contradicting_evidence', [])
            confidence_score = result.get('confidence_score', 0.0)
            reasoning = result.get('reasoning', '')
            
        except Exception as e:
            logger.error(f"Error in predicate verification: {e}")
            contradicting_evidence.append(f"Verification error: {str(e)}")
            reasoning = f"Failed to verify predicate: {str(e)}"
        
        # Determine if verified based on confidence and evidence
        verified = (confidence_score > 0.6 and len(supporting_evidence) > 0) and not (
            len(contradicting_evidence) > len(supporting_evidence) and confidence_score < 0.4
        )
        
        # Handle negation logic
        if negation:
            verified = not verified  # Flip verification for negated predicates
            if verified:
                reasoning = f"Negated predicate verified: {reasoning}"
            else:
                reasoning = f"Negated predicate not verified: {reasoning}"
        
        return {
            'predicate_id': predicate.get('id', 'unknown'),
            'predicate_string': predicate.get('original_string', ''),
            'predicate_type': predicate_type,
            'normalized_object': predicate_object,
            'original_object': original_object,
            'verified': verified,
            'confidence_score': float(confidence_score),
            'supporting_evidence': supporting_evidence,
            'contradicting_evidence': contradicting_evidence,
            'verification_method': 'enhanced_deterministic',
            'reasoning': reasoning,
            'negation': negation
        }
    
    async def _verify_condition_predicate(
        self,
        normalized_condition: str,
        original_condition: str,
        patient_data: Dict[str, Any],
        negation: bool = False
    ) -> Dict[str, Any]:
        """
        Verify condition predicate against patient data
        """
        supporting_evidence = []
        contradicting_evidence = []
        
        # Check medical history
        medical_history = patient_data.get('medical_history', [])
        condition_found = False
        
        for condition in medical_history:
            if self._semantic_similarity(normalized_condition, condition.lower().replace(' ', '_')) > 0.8:
                supporting_evidence.append(f"Condition found in medical history: {condition}")
                condition_found = True
            elif self._semantic_similarity(original_condition, condition.lower()) > 0.8:
                supporting_evidence.append(f"Original condition term matches medical history: {condition}")
                condition_found = True
        
        # Check current conditions
        current_conditions = patient_data.get('current_conditions', [])
        for condition in current_conditions:
            if self._semantic_similarity(normalized_condition, condition.lower().replace(' ', '_')) > 0.8:
                supporting_evidence.append(f"Condition found in current conditions: {condition}")
                condition_found = True
        
        if not condition_found:
            contradicting_evidence.append(f"No record of condition: {original_condition}")
        
        # Check clinical notes for diagnostic keywords
        clinical_notes = patient_data.get('clinical_notes', '')
        if clinical_notes:
            diagnostic_keywords = ['diagnosed with', 'diagnosis of', 'confirmed', 'presents with']
            for keyword in diagnostic_keywords:
                if keyword in clinical_notes.lower() and original_condition.lower() in clinical_notes.lower():
                    supporting_evidence.append(f"Clinical notes mention diagnosis: '{keyword} {original_condition}'")
                    break
        
        # Check lab correlations for the condition
        lab_results = patient_data.get('lab_results', {})
        lab_correlation = self._check_condition_lab_correlation(normalized_condition, lab_results)
        if lab_correlation:
            supporting_evidence.append(f"Lab results support condition: {lab_correlation}")
        
        confidence = self._calculate_evidence_confidence(supporting_evidence, contradicting_evidence)
        reasoning = f"Condition '{original_condition}' verification based on medical history, current conditions, and clinical evidence."
        
        return {
            'supporting_evidence': supporting_evidence,
            'contradicting_evidence': contradicting_evidence,
            'confidence_score': confidence,
            'reasoning': reasoning
        }
    
    async def _verify_generic_predicate(
        self,
        predicate_type: str,
        normalized_object: str,
        original_object: str,
        patient_data: Dict[str, Any],
        negation: bool = False
    ) -> Dict[str, Any]:
        """
        Generic verification for unknown predicate types
        """
        supporting_evidence = []
        contradicting_evidence = []
        
        # Basic text search in clinical notes
        clinical_notes = patient_data.get('clinical_notes', '')
        if clinical_notes and original_object.lower() in clinical_notes.lower():
            supporting_evidence.append(f"Term '{original_object}' found in clinical notes")
        else:
            contradicting_evidence.append(f"Term '{original_object}' not found in available data")
        
        # Check if it matches any data fields based on predicate type
        if predicate_type in ['has_finding', 'has_anatomical_finding']:
            # Check symptoms and clinical findings
            symptoms = patient_data.get('symptoms', [])
            for symptom in symptoms:
                if self._semantic_similarity(normalized_object, symptom.lower().replace(' ', '_')) > 0.6:
                    supporting_evidence.append(f"Finding matches reported symptom: {symptom}")
        
        elif predicate_type in ['has_procedure']:
            # Check for procedure mentions in notes
            if 'procedure' in clinical_notes.lower() or 'surgery' in clinical_notes.lower():
                supporting_evidence.append(f"Procedure context found in clinical notes")
        
        # Lower confidence for generic predicates
        confidence = max(0.3, self._calculate_evidence_confidence(supporting_evidence, contradicting_evidence) * 0.7)
        reasoning = f"Generic predicate '{predicate_type}({original_object})' verification with limited context."
        
        return {
            'supporting_evidence': supporting_evidence,
            'contradicting_evidence': contradicting_evidence,
            'confidence_score': confidence,
            'reasoning': reasoning
        }
    
    async def _verify_symptom_predicate(
        self,
        normalized_symptom: str,
        original_symptom: str,
        patient_data: Dict[str, Any],
        negation: bool = False
    ) -> Dict[str, Any]:
        """
        Verify symptom predicate against patient data
        """
        supporting_evidence = []
        contradicting_evidence = []
        
        # Check patient reported symptoms
        patient_symptoms = patient_data.get('symptoms', [])
        
        # Direct symptom matching
        symptom_found = False
        for reported_symptom in patient_symptoms:
            if self._semantic_similarity(normalized_symptom, reported_symptom.lower().replace(' ', '_')) > 0.7:
                supporting_evidence.append(f"Patient reported symptom matches: {reported_symptom}")
                symptom_found = True
            elif self._semantic_similarity(original_symptom, reported_symptom.lower()) > 0.7:
                supporting_evidence.append(f"Original symptom term matches patient report: {reported_symptom}")
                symptom_found = True
        
        if not symptom_found:
            contradicting_evidence.append(f"No direct patient report of symptom: {original_symptom}")
        
        # Check clinical notes
        clinical_notes = patient_data.get('clinical_notes', '')
        if clinical_notes:
            if normalized_symptom.replace('_', ' ') in clinical_notes.lower():
                supporting_evidence.append(f"Symptom mentioned in clinical notes")
            elif original_symptom.lower() in clinical_notes.lower():
                supporting_evidence.append(f"Original symptom term found in clinical notes")
        
        # Check chief complaint
        chief_complaint = patient_data.get('chief_complaint', '')
        if chief_complaint and self._semantic_similarity(original_symptom, chief_complaint.lower()) > 0.6:
            supporting_evidence.append(f"Symptom matches chief complaint: {chief_complaint}")
        
        # Check vital signs correlation
        vitals = patient_data.get('vitals', {})
        vital_correlation = self._check_vital_symptom_correlation(normalized_symptom, vitals)
        if vital_correlation:
            supporting_evidence.append(f"Vital signs support symptom: {vital_correlation}")
        
        # Calculate confidence
        confidence = self._calculate_evidence_confidence(supporting_evidence, contradicting_evidence)
        
        reasoning = f"Symptom '{original_symptom}' verification based on {len(supporting_evidence)} supporting and {len(contradicting_evidence)} contradicting evidence points."
        
        return {
            'supporting_evidence': supporting_evidence,
            'contradicting_evidence': contradicting_evidence,
            'confidence_score': confidence,
            'reasoning': reasoning
        }
    
    async def _verify_medication_predicate(
        self,
        normalized_medication: str,
        original_medication: str,
        patient_data: Dict[str, Any],
        negation: bool = False
    ) -> Dict[str, Any]:
        """
        Verify medication predicate against patient data
        """
        supporting_evidence = []
        contradicting_evidence = []
        
        # Check current medications
        current_medications = patient_data.get('current_medications', [])
        medication_found = False
        
        for medication in current_medications:
            # Extract medication name (before dosage info)
            med_name = medication.split()[0].lower()
            orig_med_name = original_medication.split()[0].lower()
            
            if self._semantic_similarity(normalized_medication.split('_')[0], med_name) > 0.8:
                supporting_evidence.append(f"Medication found in current medications: {medication}")
                medication_found = True
            elif self._semantic_similarity(orig_med_name, med_name) > 0.8:
                supporting_evidence.append(f"Original medication matches current list: {medication}")
                medication_found = True
        
        # Check medication history
        medication_history = patient_data.get('medication_history', [])
        for medication in medication_history:
            med_name = medication.split()[0].lower()
            if self._semantic_similarity(normalized_medication.split('_')[0], med_name) > 0.8:
                supporting_evidence.append(f"Medication found in medication history: {medication}")
                medication_found = True
        
        if not medication_found:
            contradicting_evidence.append(f"No record of medication: {original_medication}")
        
        # Check clinical notes for medication mentions
        clinical_notes = patient_data.get('clinical_notes', '')
        if clinical_notes and original_medication.lower() in clinical_notes.lower():
            supporting_evidence.append(f"Medication mentioned in clinical notes")
        
        confidence = self._calculate_evidence_confidence(supporting_evidence, contradicting_evidence)
        reasoning = f"Medication '{original_medication}' verification based on current medications and medication history."
        
        return {
            'supporting_evidence': supporting_evidence,
            'contradicting_evidence': contradicting_evidence,
            'confidence_score': confidence,
            'reasoning': reasoning
        }
    
    async def _verify_lab_value_predicate(
        self,
        normalized_lab: str,
        original_lab: str,
        patient_data: Dict[str, Any],
        negation: bool = False
    ) -> Dict[str, Any]:
        """
        Verify lab value predicate against patient data
        """
        supporting_evidence = []
        contradicting_evidence = []
        
        # Parse lab name and value from the predicate object
        lab_name, lab_value_info = self._parse_lab_predicate_object(normalized_lab, original_lab)
        
        # Check lab results
        lab_results = patient_data.get('lab_results', {})
        
        lab_found = False
        for lab_key, lab_value in lab_results.items():
            if self._semantic_similarity(lab_name, lab_key.lower().replace(' ', '_')) > 0.8:
                lab_found = True
                
                # If we have specific value information, check it
                if lab_value_info:
                    if self._verify_lab_value_range(lab_value_info, lab_value, lab_name):
                        supporting_evidence.append(f"Lab value matches expectation: {lab_key} = {lab_value} ({lab_value_info})")
                    else:
                        contradicting_evidence.append(f"Lab value doesn't match expectation: {lab_key} = {lab_value} (expected {lab_value_info})")
                else:
                    supporting_evidence.append(f"Lab test found: {lab_key} = {lab_value}")
        
        if not lab_found:
            contradicting_evidence.append(f"No lab results found for: {original_lab}")
        
        confidence = self._calculate_evidence_confidence(supporting_evidence, contradicting_evidence)
        reasoning = f"Lab value '{original_lab}' verification based on available lab results."
        
        return {
            'supporting_evidence': supporting_evidence,
            'contradicting_evidence': contradicting_evidence,
            'confidence_score': confidence,
            'reasoning': reasoning
        }
    
    async def _verify_vital_sign_predicate(
        self,
        normalized_vital: str,
        original_vital: str,
        patient_data: Dict[str, Any],
        negation: bool = False
    ) -> Dict[str, Any]:
        """
        Verify vital sign predicate against patient data
        """
        supporting_evidence = []
        contradicting_evidence = []
        
        # Parse vital name and value from the predicate object
        vital_name, vital_value_info = self._parse_vital_predicate_object(normalized_vital, original_vital)
        
        # Check vitals
        vitals = patient_data.get('vitals', {})
        
        vital_found = False
        for vital_key, vital_value in vitals.items():
            if self._semantic_similarity(vital_name, vital_key.lower().replace(' ', '_')) > 0.8:
                vital_found = True
                
                # If we have specific value information, check it
                if vital_value_info:
                    if self._verify_vital_value_range(vital_value_info, vital_value, vital_name):
                        supporting_evidence.append(f"Vital sign matches expectation: {vital_key} = {vital_value} ({vital_value_info})")
                    else:
                        contradicting_evidence.append(f"Vital sign doesn't match expectation: {vital_key} = {vital_value} (expected {vital_value_info})")
                else:
                    supporting_evidence.append(f"Vital sign recorded: {vital_key} = {vital_value}")
        
        if not vital_found:
            contradicting_evidence.append(f"No vital signs recorded for: {original_vital}")
        
        confidence = self._calculate_evidence_confidence(supporting_evidence, contradicting_evidence)
        reasoning = f"Vital sign '{original_vital}' verification based on recorded vital signs."
        
        return {
            'supporting_evidence': supporting_evidence,
            'contradicting_evidence': contradicting_evidence,
            'confidence_score': confidence,
            'reasoning': reasoning
        }

    async def extract_predicates_from_text_advanced(
        self,
        explanation_text: str,
        context: Optional[str] = None
    ) -> List[AdvancedFOLPredicate]:
        """
        Advanced predicate extraction using multiple NLP techniques

        Args:
            explanation_text: The medical explanation text to analyze
            context: Additional context (optional)

        Returns:
            List of advanced FOL predicates with rich metadata
        """
        logger.info(f"Extracting advanced predicates from text ({len(explanation_text)} chars)")

        predicates = []

        try:
            # Step 1: Extract medical entities using advanced NLP
            entities = self.ner.extract_medical_entities_advanced(explanation_text)
            logger.info(f"Extracted {len(entities)} medical entities")

            # Step 2: Generate predicates for each entity (with parallel processing if many entities)
            if len(entities) > 3:  # Use parallel processing for larger entity sets
                predicates = await self._generate_predicates_parallel(entities, explanation_text, context)
            else:
                # Sequential processing for smaller sets
                for entity in entities:
                    # Get surrounding context for the entity
                    surrounding_text = self._get_surrounding_context(
                        explanation_text, entity.start_pos, entity.end_pos
                    )

                    # Generate advanced predicate
                    predicate = self.predicate_generator.convert_term_to_predicate_enhanced(
                        term=entity.text,
                        entity_type=entity.entity_type,
                        context=context or "",
                        surrounding_text=surrounding_text
                    )

                    predicates.append(predicate)

            # Step 3: Filter and rank predicates
            filtered_predicates = self._filter_and_rank_predicates(predicates)

            logger.info(f"Generated {len(filtered_predicates)} advanced FOL predicates")
            return filtered_predicates

        except Exception as e:
            logger.error(f"Error in advanced predicate extraction: {e}")
            return []

    async def _generate_predicates_parallel(
        self,
        entities: List[MedicalEntity],
        explanation_text: str,
        context: Optional[str] = None
    ) -> List[AdvancedFOLPredicate]:
        """Generate predicates using parallel processing for better performance"""
        import concurrent.futures

        logger.info(f"Using parallel predicate generation for {len(entities)} entities")

        async def process_entity(entity: MedicalEntity) -> AdvancedFOLPredicate:
            """Process a single entity to generate predicate"""
            try:
                # Get surrounding context for the entity
                surrounding_text = self._get_surrounding_context(
                    explanation_text, entity.start_pos, entity.end_pos
                )

                # Generate advanced predicate
                predicate = self.predicate_generator.convert_term_to_predicate_enhanced(
                    term=entity.text,
                    entity_type=entity.entity_type,
                    context=context or "",
                    surrounding_text=surrounding_text
                )

                return predicate
            except Exception as e:
                logger.warning(f"Error processing entity {entity.text}: {e}")
                # Return a basic predicate as fallback
                return AdvancedFOLPredicate(
                    subject="patient",
                    predicate="has_medical_entity",
                    object=entity.text.lower(),
                    confidence=0.5,
                    evidence_entities=[entity],
                    medical_reasoning=f"Generated with reduced confidence due to processing error: {str(e)}"
                )

        # Create tasks for parallel execution
        tasks = [process_entity(entity) for entity in entities]

        # Execute tasks with controlled concurrency
        semaphore = asyncio.Semaphore(self.max_workers)

        async def limited_task(task):
            async with semaphore:
                return await task

        # Run all tasks with concurrency limit
        limited_tasks = [limited_task(task) for task in tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)

        # Handle any exceptions and collect successful results
        predicates = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Parallel predicate generation failed for entity {i}: {result}")
                # Create fallback predicate
                fallback_predicate = AdvancedFOLPredicate(
                    subject="patient",
                    predicate="has_medical_entity",
                    object=entities[i].text.lower() if i < len(entities) else "unknown",
                    confidence=0.3,
                    evidence_entities=[entities[i]] if i < len(entities) else [],
                    medical_reasoning=f"Fallback predicate due to parallel processing error: {str(result)}"
                )
                predicates.append(fallback_predicate)
            else:
                predicates.append(result)

        logger.info(f"Parallel predicate generation completed: {len(predicates)} predicates generated")
        return predicates

    def _get_surrounding_context(self, text: str, start_pos: int, end_pos: int,
                               window_size: int = 100) -> str:
        """Extract surrounding context for an entity"""
        start = max(0, start_pos - window_size)
        end = min(len(text), end_pos + window_size)

        return text[start:end]

    def _filter_and_rank_predicates(self, predicates: List[AdvancedFOLPredicate]) -> List[AdvancedFOLPredicate]:
        """Filter and rank predicates by quality and relevance"""
        # Remove duplicates based on FOL string
        seen = set()
        unique_predicates = []

        for predicate in predicates:
            fol_key = predicate.to_fol_string()
            if fol_key not in seen:
                seen.add(fol_key)
                unique_predicates.append(predicate)

        # Filter out low-quality predicates
        filtered = []
        for predicate in unique_predicates:
            # Skip very short objects
            if len(predicate.object) < 3:
                continue

            # Skip very low confidence
            if predicate.confidence < 0.4:
                continue

            # Skip generic terms
            if predicate.object.lower() in ["patient", "the", "and", "or", "with", "has", "is"]:
                continue

            filtered.append(predicate)

        # Sort by confidence - ensure all confidences are floats
        try:
            filtered.sort(key=lambda x: float(x.confidence), reverse=True)
        except (TypeError, ValueError) as e:
            logger.warning(f"Error sorting predicates by confidence: {e}")
            # Fallback: sort by a default confidence value
            filtered.sort(key=lambda x: 0.5, reverse=True)

        return filtered

    async def extract_predicates_from_text(
        self,
        explanation_text: str
    ) -> List[str]:
        """
        Legacy method for backward compatibility - returns simple FOL strings
        """
        advanced_predicates = await self.extract_predicates_from_text_advanced(explanation_text)

        # Convert to simple string format for backward compatibility
        fol_strings = []
        for predicate in advanced_predicates:
            fol_strings.append(predicate.to_fol_string())

        return fol_strings
    
    def _semantic_similarity(self, term1: str, term2: str) -> float:
        """
        Calculate semantic similarity between two medical terms
        """
        if not term1 or not term2:
            return 0.0
        
        term1 = term1.lower().strip()
        term2 = term2.lower().strip()
        
        # Exact match
        if term1 == term2:
            return 1.0
        
        # Substring match
        if term1 in term2 or term2 in term1:
            return 0.8
        
        # Jaccard similarity for word sets
        words1 = set(term1.replace('_', ' ').split())
        words2 = set(term2.replace('_', ' ').split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _check_vital_symptom_correlation(self, symptom: str, vitals: Dict[str, Any]) -> Optional[str]:
        """
        Check if vital signs correlate with reported symptoms
        """
        symptom_vital_correlations = {
            'chest_pain': ['heart_rate', 'blood_pressure'],
            'shortness_of_breath': ['respiratory_rate', 'oxygen_saturation'],
            'fever': ['temperature'],
            'dizziness': ['blood_pressure', 'heart_rate'],
            'headache': ['blood_pressure'],
            'fatigue': ['heart_rate', 'blood_pressure']
        }
        
        if symptom in symptom_vital_correlations:
            relevant_vitals = symptom_vital_correlations[symptom]
            correlations = []
            
            for vital in relevant_vitals:
                if vital in vitals:
                    correlations.append(f"{vital}: {vitals[vital]}")
            
            if correlations:
                return ", ".join(correlations)
        
        return None
    
    def _check_condition_lab_correlation(self, condition: str, lab_results: Dict[str, Any]) -> Optional[str]:
        """
        Check if lab results correlate with diagnosed conditions
        """
        condition_lab_correlations = {
            'diabetes': ['glucose', 'hba1c'],
            'myocardial_infarction': ['troponin', 'ck_mb'],
            'kidney_disease': ['creatinine', 'bun'],
            'anemia': ['hemoglobin', 'hematocrit'],
            'liver_disease': ['alt', 'ast', 'bilirubin'],
            'infection': ['wbc', 'crp']
        }
        
        if condition in condition_lab_correlations:
            relevant_labs = condition_lab_correlations[condition]
            correlations = []
            
            for lab in relevant_labs:
                for lab_key, lab_value in lab_results.items():
                    if lab in lab_key.lower().replace(' ', '_'):
                        correlations.append(f"{lab_key}: {lab_value}")
            
            if correlations:
                return ", ".join(correlations)
        
        return None
    
    def _parse_lab_predicate_object(self, normalized_lab: str, original_lab: str) -> Tuple[str, Optional[str]]:
        """
        Parse lab predicate object to extract lab name and value information
        """
        # Check for value information in the format lab_name:value or lab_name_value
        if ':' in normalized_lab:
            parts = normalized_lab.split(':', 1)
            return parts[0], parts[1] if len(parts) > 1 else None
        elif '_' in normalized_lab:
            # Look for value indicators like elevated, high, low, normal
            value_indicators = ['elevated', 'high', 'low', 'normal', 'abnormal']
            parts = normalized_lab.split('_')
            for indicator in value_indicators:
                if indicator in parts:
                    lab_name = '_'.join([p for p in parts if p != indicator])
                    return lab_name, indicator
        
        return normalized_lab, None
    
    def _parse_vital_predicate_object(self, normalized_vital: str, original_vital: str) -> Tuple[str, Optional[str]]:
        """
        Parse vital predicate object to extract vital name and value information
        """
        # Similar to lab parsing
        if ':' in normalized_vital:
            parts = normalized_vital.split(':', 1)
            return parts[0], parts[1] if len(parts) > 1 else None
        elif '_' in normalized_vital:
            # Look for value indicators
            value_indicators = ['high', 'low', 'normal', 'elevated', 'decreased']
            parts = normalized_vital.split('_')
            for indicator in value_indicators:
                if indicator in parts:
                    vital_name = '_'.join([p for p in parts if p != indicator])
                    return vital_name, indicator
        
        return normalized_vital, None
    
    def _verify_lab_value_range(self, expected_range: str, actual_value: Any, lab_name: str) -> bool:
        """
        Verify if actual lab value matches expected range/description
        """
        try:
            if isinstance(actual_value, (int, float)):
                # For numeric values, check ranges
                if expected_range.lower() in ['elevated', 'high']:
                    return self._is_lab_value_elevated(lab_name, actual_value)
                elif expected_range.lower() in ['low', 'decreased']:
                    return self._is_lab_value_low(lab_name, actual_value)
                elif expected_range.lower() == 'normal':
                    return self._is_lab_value_normal(lab_name, actual_value)
                else:
                    # Try to parse specific numeric value
                    import re
                    numeric_match = re.search(r'(\d+(?:\.\d+)?)', expected_range)
                    if numeric_match:
                        expected_value = float(numeric_match.group(1))
                        # Allow 20% variance
                        return abs(actual_value - expected_value) <= (expected_value * 0.2)
            
            # For string values, do text matching
            return expected_range.lower() in str(actual_value).lower()
            
        except Exception:
            return False
    
    def _verify_vital_value_range(self, expected_range: str, actual_value: Any, vital_name: str) -> bool:
        """
        Verify if actual vital sign value matches expected range/description
        """
        try:
            if isinstance(actual_value, (int, float)):
                if expected_range.lower() in ['high', 'elevated']:
                    return self._is_vital_value_elevated(vital_name, actual_value)
                elif expected_range.lower() in ['low', 'decreased']:
                    return self._is_vital_value_low(vital_name, actual_value)
                elif expected_range.lower() == 'normal':
                    return self._is_vital_value_normal(vital_name, actual_value)
                else:
                    # Try to parse specific numeric value
                    import re
                    numeric_match = re.search(r'(\d+(?:\.\d+)?)', expected_range)
                    if numeric_match:
                        expected_value = float(numeric_match.group(1))
                        return abs(actual_value - expected_value) <= (expected_value * 0.15)
            
            return expected_range.lower() in str(actual_value).lower()
            
        except Exception:
            return False
    
    def _is_lab_value_elevated(self, lab_name: str, value: float) -> bool:
        """
        Check if lab value is elevated based on normal ranges
        """
        normal_ranges = {
            'glucose': 140,
            'creatinine': 1.2,
            'troponin': 0.04,
            'alt': 40,
            'ast': 40,
            'bilirubin': 1.2,
            'crp': 3.0,
            'wbc': 11.0
        }
        
        for lab_key, upper_limit in normal_ranges.items():
            if lab_key in lab_name.lower():
                return value > upper_limit
        
        return False  # Conservative approach
    
    def _is_lab_value_low(self, lab_name: str, value: float) -> bool:
        """
        Check if lab value is low based on normal ranges
        """
        normal_ranges = {
            'glucose': 70,
            'creatinine': 0.6,
            'hemoglobin': 12.0,
            'hematocrit': 36.0,
            'platelets': 150
        }
        
        for lab_key, lower_limit in normal_ranges.items():
            if lab_key in lab_name.lower():
                return value < lower_limit
        
        return False
    
    def _is_lab_value_normal(self, lab_name: str, value: float) -> bool:
        """
        Check if lab value is within normal range
        """
        return not (self._is_lab_value_elevated(lab_name, value) or self._is_lab_value_low(lab_name, value))
    
    def _is_vital_value_elevated(self, vital_name: str, value: float) -> bool:
        """
        Check if vital sign value is elevated
        """
        normal_upper_limits = {
            'heart_rate': 100,
            'blood_pressure': 140,  # systolic
            'temperature': 99.5,
            'respiratory_rate': 20
        }
        
        for vital_key, upper_limit in normal_upper_limits.items():
            if vital_key in vital_name.lower():
                return value > upper_limit
        
        return False
    
    def _is_vital_value_low(self, vital_name: str, value: float) -> bool:
        """
        Check if vital sign value is low
        """
        normal_lower_limits = {
            'heart_rate': 60,
            'blood_pressure': 90,  # systolic
            'temperature': 97.0,
            'respiratory_rate': 12,
            'oxygen_saturation': 95
        }
        
        for vital_key, lower_limit in normal_lower_limits.items():
            if vital_key in vital_name.lower():
                return value < lower_limit
        
        return False
    
    def _is_vital_value_normal(self, vital_name: str, value: float) -> bool:
        """
        Check if vital sign value is within normal range
        """
        return not (self._is_vital_value_elevated(vital_name, value) or self._is_vital_value_low(vital_name, value))
    
    def _calculate_evidence_confidence(self, supporting_evidence: List[str], contradicting_evidence: List[str]) -> float:
        """
        Calculate confidence based on supporting and contradicting evidence
        """
        if not supporting_evidence and not contradicting_evidence:
            return 0.5  # Neutral confidence with no evidence
        
        support_count = len(supporting_evidence)
        contradict_count = len(contradicting_evidence)
        total_evidence = support_count + contradict_count
        
        if total_evidence == 0:
            return 0.5
        
        # Base confidence on evidence ratio
        support_ratio = support_count / total_evidence
        
        # Apply evidence strength weighting
        base_confidence = support_ratio
        
        # Boost confidence if we have strong supporting evidence
        if support_count >= 2 and contradict_count == 0:
            base_confidence = min(0.95, base_confidence + 0.1)
        
        # Reduce confidence if we have contradicting evidence
        if contradict_count > support_count:
            base_confidence = max(0.1, base_confidence - 0.2)
        
        return max(0.0, min(1.0, base_confidence))
    
    def _calculate_comprehensive_confidence(
        self,
        verification_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive confidence metrics
        """
        if not verification_results:
            return {
                'overall_confidence': 0.0,
                'verification_success_rate': 0.0,
                'average_predicate_confidence': 0.0,
                'high_confidence_percentage': 0.0
            }
        
        verified_count = sum(1 for result in verification_results if result.get('verified', False))
        total_count = len(verification_results)
        
        confidence_scores = [result.get('confidence_score', 0.0) for result in verification_results]
        average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        high_confidence_count = sum(1 for score in confidence_scores if score >= 0.8)
        high_confidence_percentage = (high_confidence_count / total_count) if total_count > 0 else 0.0
        
        verification_success_rate = (verified_count / total_count) if total_count > 0 else 0.0
        
        # Overall confidence combines success rate and average confidence
        overall_confidence = (verification_success_rate * 0.6) + (average_confidence * 0.4)
        
        return {
            'overall_confidence': overall_confidence,
            'verification_success_rate': verification_success_rate,
            'average_predicate_confidence': average_confidence,
            'high_confidence_percentage': high_confidence_percentage
        }
    
    def _generate_detailed_verification_report(
        self,
        extraction_results: Dict[str, Any],
        structured_predicates: List[Dict[str, Any]],
        verification_results: List[Dict[str, Any]],
        confidence_analysis: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive verification report
        """
        # Count verification outcomes
        verified_count = sum(1 for result in verification_results if result.get('verified', False))
        failed_count = len(verification_results) - verified_count
        
        # Generate summary
        if confidence_analysis['overall_confidence'] >= 0.8:
            status = 'FULLY_VERIFIED'
            summary = f"High confidence verification: {verified_count}/{len(verification_results)} predicates verified with {confidence_analysis['overall_confidence']*100:.1f}% overall confidence."
        elif confidence_analysis['overall_confidence'] >= 0.6:
            status = 'PARTIALLY_VERIFIED'
            summary = f"Moderate confidence verification: {verified_count}/{len(verification_results)} predicates verified with {confidence_analysis['overall_confidence']*100:.1f}% overall confidence."
        else:
            status = 'LOW_CONFIDENCE'
            summary = f"Low confidence verification: {verified_count}/{len(verification_results)} predicates verified with {confidence_analysis['overall_confidence']*100:.1f}% overall confidence."
        
        # Generate medical reasoning summary
        medical_reasoning_points = []
        for result in verification_results:
            if result.get('verified', False) and result.get('reasoning'):
                medical_reasoning_points.append(result['reasoning'])
        
        medical_reasoning_summary = " ".join(medical_reasoning_points[:3]) + "..." if len(medical_reasoning_points) > 3 else " ".join(medical_reasoning_points)
        
        return {
            'status': status,
            'verification_summary': summary,
            'total_predicates': len(structured_predicates),
            'verified_predicates': verified_count,
            'failed_predicates': failed_count,
            'overall_confidence': confidence_analysis['overall_confidence'],
            'success_rate': confidence_analysis['verification_success_rate'],
            'average_predicate_confidence': confidence_analysis['average_predicate_confidence'],
            'high_confidence_percentage': confidence_analysis['high_confidence_percentage'],
            'medical_reasoning_summary': medical_reasoning_summary,
            'predicate_verification_details': verification_results,
            'extraction_method': extraction_results.get('extraction_method', 'enhanced_nlp'),
            'raw_extraction_results': extraction_results,
            'confidence_metrics': confidence_analysis
        }
    
    def verify_predicates_against_patient_data_sync(
        self, 
        predicates: List[str], 
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synchronous version of predicate verification for backward compatibility
        """
        logger.info(f"🔍 Synchronous verification of {len(predicates)} predicates")
        
        try:
            import asyncio
            
            # Create new event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async verification
            if loop.is_running():
                # If we're already in an event loop, we can't use asyncio.run()
                # Fall back to basic verification
                return self._basic_predicate_verification(predicates, patient_data)
            else:
                # Create a dummy clinical text for the async method
                clinical_text = "\n".join(predicates)
                return loop.run_until_complete(
                    self.extract_and_verify_predicates(clinical_text, patient_data)
                )
                
        except Exception as e:
            logger.error(f"❌ Synchronous verification failed: {e}")
            return self._basic_predicate_verification(predicates, patient_data)
    
    def _basic_predicate_verification(
        self, 
        predicates: List[str], 
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Basic synchronous predicate verification fallback
        """
        verification_results = []
        verified_count = 0
        
        for i, predicate in enumerate(predicates):
            try:
                # Basic parsing and verification
                parsed = self._parse_predicate_string(predicate)
                if parsed:
                    # Simple verification based on predicate type
                    verified = self._simple_predicate_check(parsed, patient_data)
                    confidence = 0.7 if verified else 0.3
                    
                    if verified:
                        verified_count += 1
                    
                    verification_results.append({
                        'predicate_id': f'predicate_{i+1}',
                        'predicate_string': predicate,
                        'verified': verified,
                        'confidence_score': confidence,
                        'supporting_evidence': ['Basic pattern matching'] if verified else [],
                        'contradicting_evidence': ['No clear evidence found'] if not verified else [],
                        'verification_method': 'basic_sync',
                        'reasoning': f'Basic verification of {parsed["predicate"]} predicate'
                    })
                else:
                    verification_results.append({
                        'predicate_id': f'predicate_{i+1}',
                        'predicate_string': predicate,
                        'verified': False,
                        'confidence_score': 0.0,
                        'supporting_evidence': [],
                        'contradicting_evidence': ['Failed to parse predicate'],
                        'verification_method': 'basic_sync',
                        'reasoning': 'Could not parse predicate structure'
                    })
            except Exception as e:
                verification_results.append({
                    'predicate_id': f'predicate_{i+1}',
                    'predicate_string': predicate,
                    'verified': False,
                    'confidence_score': 0.0,
                    'supporting_evidence': [],
                    'contradicting_evidence': [f'Verification error: {str(e)}'],
                    'verification_method': 'basic_sync',
                    'reasoning': f'Error during verification: {str(e)}'
                })
        
        overall_confidence = verified_count / len(predicates) if predicates else 0.0
        success_rate = verified_count / len(predicates) if predicates else 0.0
        
        return {
            'status': 'VERIFIED' if overall_confidence >= 0.6 else 'PARTIAL',
            'verification_summary': f'Basic verification: {verified_count}/{len(predicates)} predicates verified',
            'total_predicates': len(predicates),
            'verified_predicates': verified_count,
            'failed_predicates': len(predicates) - verified_count,
            'overall_confidence': overall_confidence,
            'success_rate': success_rate,
            'predicate_verification_details': verification_results,
            'extraction_method': 'basic_sync'
        }
    
    def _simple_predicate_check(self, parsed_predicate: Dict[str, Any], patient_data: Dict[str, Any]) -> bool:
        """
        Simple predicate verification for basic fallback
        """
        predicate_type = parsed_predicate.get('predicate', '')
        obj = parsed_predicate.get('object', '').lower()
        
        if predicate_type == 'has_symptom':
            symptoms = patient_data.get('symptoms', [])
            # Enhanced matching for symptoms
            for symptom in symptoms:
                symptom_lower = symptom.lower()
                # Direct match
                if obj in symptom_lower or symptom_lower in obj:
                    return True
                # Word-by-word match
                obj_words = obj.replace('_', ' ').split()
                symptom_words = symptom_lower.split()
                if any(word in symptom_words for word in obj_words if len(word) > 2):
                    return True
            return False
        
        elif predicate_type == 'has_condition':
            conditions = patient_data.get('medical_history', []) + patient_data.get('current_conditions', [])
            return any(obj in condition.lower() for condition in conditions)
        
        elif predicate_type == 'takes_medication':
            medications = patient_data.get('current_medications', [])
            return any(obj.split('_')[0] in medication.lower() for medication in medications)
        
        elif predicate_type == 'has_lab_value':
            lab_results = patient_data.get('lab_results', {})
            return any(obj.split('_')[0] in lab_name.lower() for lab_name in lab_results.keys())
        
        elif predicate_type == 'has_vital_sign':
            vitals = patient_data.get('vitals', {})
            return any(obj.split('_')[0] in vital_name.lower() for vital_name in vitals.keys())
        
        return False
