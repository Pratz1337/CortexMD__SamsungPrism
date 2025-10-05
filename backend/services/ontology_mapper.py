import json
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
import logging
from datetime import datetime

from services.neo4j_service import Neo4jService
from services.umls_client import UMLSClient, UMLSConcept
from services.enhanced_knowledge_graph import EnhancedKnowledgeGraphService
from services.intelligent_cache import get_cache, ontology_cache_key, medical_term_cache_key, cached_result
from config.neo4j_config import get_config

logger = logging.getLogger(__name__)

@dataclass
class MedicalConcept:
    cui: str  # Concept Unique Identifier
    preferred_name: str
    synonyms: List[str]
    semantic_types: List[str]
    definition: Optional[str] = None

class OntologyMapper:
    def __init__(self, use_enhanced_services: bool = True):
        """Initialize ontology mapping service"""
        logger.info("Initializing Ontology Mapper")

        self.use_enhanced_services = use_enhanced_services
        self.neo4j_service = None
        self.umls_client = None

        # Initialize enhanced services if requested
        if use_enhanced_services:
            self._initialize_enhanced_services()

        # Common medical abbreviations and synonyms (fallback)
        self.medical_synonyms = {
            "bp": "blood pressure",
            "hr": "heart rate",
            "temp": "temperature",
            "rr": "respiratory rate",
            "sob": "shortness of breath",
            "doe": "dyspnea on exertion",
            "cp": "chest pain",
            "abd": "abdominal",
            "htn": "hypertension",
            "dm": "diabetes mellitus",
            "cad": "coronary artery disease",
            "chf": "congestive heart failure",
            "copd": "chronic obstructive pulmonary disease",
            "mi": "myocardial infarction",
            "pe": "pulmonary embolism",
            "dvt": "deep vein thrombosis",
            "afib": "atrial fibrillation",
            "vtach": "ventricular tachycardia",
            "vfib": "ventricular fibrillation",
            "acs": "acute coronary syndrome",
            "stemi": "st elevation myocardial infarction",
            "nstemi": "non-st elevation myocardial infarction",
            "cva": "cerebrovascular accident",
            "tia": "transient ischemic attack"
        }

        # Medical concept database (simplified UMLS-like structure for fallback)
        self.medical_concepts = {
            "chest pain": MedicalConcept(
                cui="C0008031",
                preferred_name="Chest Pain",
                synonyms=["chest discomfort", "thoracic pain", "chest tightness", "angina", "chest pressure"],
                semantic_types=["Sign or Symptom"],
                definition="Pain in the chest area, often related to cardiac, pulmonary, or musculoskeletal conditions"
            ),
            "shortness of breath": MedicalConcept(
                cui="C0013404",
                preferred_name="Dyspnea",
                synonyms=["breathlessness", "shortness of breath", "difficulty breathing", "sob", "air hunger"],
                semantic_types=["Sign or Symptom"],
                definition="Difficult or labored breathing"
            ),
            "hypertension": MedicalConcept(
                cui="C0020538",
                preferred_name="Hypertensive disease",
                synonyms=["high blood pressure", "arterial hypertension", "htn", "elevated blood pressure"],
                semantic_types=["Disease or Syndrome"],
                definition="Persistently elevated arterial blood pressure"
            ),
            "diabetes": MedicalConcept(
                cui="C0011847",
                preferred_name="Diabetes Mellitus",
                synonyms=["diabetes mellitus", "dm", "diabetes", "hyperglycemia disorder"],
                semantic_types=["Disease or Syndrome"],
                definition="A group of metabolic disorders characterized by high blood glucose levels"
            ),
            "fever": MedicalConcept(
                cui="C0015967",
                preferred_name="Fever",
                synonyms=["pyrexia", "febrile", "elevated temperature", "hyperthermia"],
                semantic_types=["Sign or Symptom"],
                definition="Elevated body temperature above normal range"
            ),
            "myocardial infarction": MedicalConcept(
                cui="C0027051",
                preferred_name="Myocardial Infarction",
                synonyms=["heart attack", "mi", "acute myocardial infarction", "coronary occlusion"],
                semantic_types=["Disease or Syndrome"],
                definition="Death of heart muscle due to insufficient blood supply"
            ),
            "pneumonia": MedicalConcept(
                cui="C0032285",
                preferred_name="Pneumonia",
                synonyms=["lung infection", "pulmonary infection", "pneumonitis"],
                semantic_types=["Disease or Syndrome"],
                definition="Infection that inflames air sacs in one or both lungs"
            ),
            "aspirin": MedicalConcept(
                cui="C0004057",
                preferred_name="Aspirin",
                synonyms=["acetylsalicylic acid", "asa", "aspirin", "salicylate"],
                semantic_types=["Pharmacologic Substance"],
                definition="Nonsteroidal anti-inflammatory drug used for pain relief and cardiovascular protection"
            ),
            "lisinopril": MedicalConcept(
                cui="C0065374",
                preferred_name="Lisinopril",
                synonyms=["lisinopril", "ace inhibitor", "prinivil", "zestril"],
                semantic_types=["Pharmacologic Substance"],
                definition="ACE inhibitor used to treat high blood pressure and heart failure"
            ),
            "metformin": MedicalConcept(
                cui="C0025598",
                preferred_name="Metformin",
                synonyms=["metformin", "glucophage", "biguanide"],
                semantic_types=["Pharmacologic Substance"],
                definition="Oral diabetes medication that helps control blood glucose levels"
            ),
            "troponin": MedicalConcept(
                cui="C0041199",
                preferred_name="Troponin",
                synonyms=["troponin i", "troponin t", "cardiac troponin", "tn"],
                semantic_types=["Amino Acid, Peptide, or Protein"],
                definition="Cardiac biomarker used to diagnose myocardial infarction"
            ),
            "glucose": MedicalConcept(
                cui="C0017725",
                preferred_name="Glucose",
                synonyms=["blood sugar", "blood glucose", "serum glucose", "dextrose"],
                semantic_types=["Organic Chemical"],
                definition="Simple sugar that is the primary source of energy for cells"
            ),
            "creatinine": MedicalConcept(
                cui="C0010294",
                preferred_name="Creatinine",
                synonyms=["serum creatinine", "cr", "creat"],
                semantic_types=["Organic Chemical"],
                definition="Waste product used to assess kidney function"
            ),
            "blood pressure": MedicalConcept(
                cui="C0005823",
                preferred_name="Blood Pressure",
                synonyms=["bp", "arterial pressure", "systolic pressure", "diastolic pressure"],
                semantic_types=["Organism Function"],
                definition="Pressure exerted by circulating blood on arterial walls"
            ),
            "heart rate": MedicalConcept(
                cui="C0018810",
                preferred_name="Heart Rate",
                synonyms=["hr", "pulse", "cardiac rate", "pulse rate"],
                semantic_types=["Organism Function"],
                definition="Number of heartbeats per unit of time"
            ),
            "temperature": MedicalConcept(
                cui="C0005903",
                preferred_name="Body Temperature",
                synonyms=["temp", "core temperature", "body temp"],
                semantic_types=["Organism Function"],
                definition="Measure of the body's ability to generate and get rid of heat"
            )
        }

        # Build reverse lookup index
        self._build_synonym_index()

    def _initialize_enhanced_services(self):
        """Initialize enhanced ontology services"""
        try:
            # Get Neo4j configuration
            config = get_config()

            # Initialize Neo4j service
            self.neo4j_service = Neo4jService(config)

            # Initialize UMLS client if API key is available
            umls_config = config.get_ontology_config("UMLS")
            if umls_config and umls_config.get("api_key"):
                self.umls_client = UMLSClient(
                    api_key=umls_config["api_key"],
                    version=umls_config.get("version", "current")
                )
                logger.info("Enhanced UMLS client initialized")
            else:
                logger.warning("UMLS API key not configured - using fallback concepts")



        except Exception as e:
            logger.error(f"Failed to initialize enhanced services: {str(e)}")
            logger.info("Falling back to basic ontology mapping")
    
    def _build_synonym_index(self):
        """Build reverse lookup index for synonyms"""
        self.synonym_index = {}
        
        for concept_key, concept in self.medical_concepts.items():
            # Add primary key
            self.synonym_index[concept_key] = concept
            
            # Add preferred name
            self.synonym_index[concept.preferred_name.lower()] = concept
            
            # Add all synonyms
            for synonym in concept.synonyms:
                self.synonym_index[synonym.lower()] = concept
        
        # Add medical abbreviations
        for abbrev, expansion in self.medical_synonyms.items():
            if expansion in self.synonym_index:
                self.synonym_index[abbrev] = self.synonym_index[expansion]
    
    async def normalize_medical_term(self, term: str) -> Optional[MedicalConcept]:
        """Normalize medical term using medical concept database and Neo4j fallback"""
        if not term:
            return None
        
        # Clean and standardize term
        cleaned_term = self._clean_term(term)
        
        logger.debug(f"Normalizing term: '{term}' -> '{cleaned_term}'")
        
        # Direct lookup in local database
        concept = self._lookup_concept(cleaned_term)
        
        if not concept:
            # Try synonym lookup
            synonym_term = self.medical_synonyms.get(cleaned_term.lower())
            if synonym_term:
                concept = self._lookup_concept(synonym_term)
        
        if not concept:
            # Try partial matching
            concept = self._fuzzy_lookup(cleaned_term)
        
        # NEW: Try Neo4j search if local lookup failed
        if not concept and self.neo4j_service:
            concept = await self._search_neo4j_concept(cleaned_term)
        
        if not concept:
            # Create unknown concept
            concept = self._create_unknown_concept(cleaned_term)
        
        logger.debug(f"Normalized '{term}' to '{concept.preferred_name}' (CUI: {concept.cui})")
        return concept
    
    def _lookup_concept(self, term: str) -> Optional[MedicalConcept]:
        """Direct lookup in concept database"""
        return self.synonym_index.get(term.lower())
    
    def _fuzzy_lookup(self, term: str) -> Optional[MedicalConcept]:
        """Fuzzy matching for partial terms"""
        term_lower = term.lower()
        term_words = set(term_lower.split())
        
        best_match = None
        best_score = 0.0
        
        for key, concept in self.synonym_index.items():
            key_words = set(key.split())
            
            # Calculate overlap score
            if key_words and term_words:
                overlap = len(key_words.intersection(term_words))
                score = overlap / max(len(key_words), len(term_words))
                
                # Prefer exact substring matches
                if term_lower in key or key in term_lower:
                    score += 0.5
                
                if score > best_score and score > 0.6:
                    best_score = score
                    best_match = concept
        
        return best_match
    
    def _create_unknown_concept(self, term: str) -> MedicalConcept:
        """Create concept for unknown terms"""
        # Generate pseudo-CUI
        cui = f"C{abs(hash(term)) % 9999999:07d}"
        
        # Determine semantic type based on context clues
        semantic_type = self._infer_semantic_type(term)
        
        return MedicalConcept(
            cui=cui,
            preferred_name=term.title(),
            synonyms=[term],
            semantic_types=[semantic_type],
            definition=f"Medical concept: {term}"
        )
    
    def _infer_semantic_type(self, term: str) -> str:
        """Infer semantic type from term characteristics"""
        term_lower = term.lower()
        
        # Medication indicators
        medication_indicators = ['mg', 'mcg', 'tablet', 'capsule', 'daily', 'bid', 'tid']
        if any(indicator in term_lower for indicator in medication_indicators):
            return "Pharmacologic Substance"
        
        # Lab value indicators  
        lab_indicators = ['level', 'value', 'result', 'elevated', 'high', 'low', 'normal']
        if any(indicator in term_lower for indicator in lab_indicators):
            return "Laboratory or Test Result"
        
        # Symptom indicators
        symptom_indicators = ['pain', 'ache', 'discomfort', 'difficulty', 'trouble', 'feeling']
        if any(indicator in term_lower for indicator in symptom_indicators):
            return "Sign or Symptom"
        
        # Disease indicators
        disease_indicators = ['syndrome', 'disease', 'disorder', 'condition', 'failure', 'infection']
        if any(indicator in term_lower for indicator in disease_indicators):
            return "Disease or Syndrome"
        
        # Default
        return "Medical Concept"
    
    async def _search_neo4j_concept(self, term: str) -> Optional[MedicalConcept]:
        """Search for concept in Neo4j database"""
        try:
            if not self.neo4j_service:
                logger.debug("Neo4j service not available for term search")
                return None
                
            # Try to connect if not already connected
            if not self.neo4j_service.driver:
                await self.neo4j_service.connect()
            
            # Search for concepts in Neo4j
            results = await self.neo4j_service.search_concepts(term, limit=1)
            
            if results:
                result = results[0]  # Take the best match
                return MedicalConcept(
                    cui=result.get('cui', ''),
                    preferred_name=result.get('preferred_name', term),
                    synonyms=result.get('synonyms', []),
                    semantic_types=result.get('semantic_types', []),
                    definition=result.get('definition', '')
                )
                
        except Exception as e:
            logger.warning(f"Failed to search Neo4j for term '{term}': {e}")
            
        return None
    
    def _clean_term(self, term: str) -> str:
        """Clean and standardize medical term"""
        if not term:
            return ""
        
        # Remove extra whitespace and punctuation
        cleaned = ' '.join(term.strip().split())
        
        # Remove dosage information for cleaner matching
        cleaned = re.sub(r'\d+\s*(?:mg|mcg|g|ml|units?|iu)\b', '', cleaned, flags=re.IGNORECASE)
        
        # Remove frequency information
        cleaned = re.sub(r'\b(?:daily|bid|tid|qid|prn|as needed|once|twice)\b', '', cleaned, flags=re.IGNORECASE)
        
        # Remove common prefixes/suffixes that don't affect core meaning
        prefixes_to_remove = ['severe', 'mild', 'moderate', 'acute', 'chronic', 'slight', 'significant']
        words = cleaned.split()
        
        # Filter out non-essential modifiers but keep if they're the only words
        if len(words) > 1:
            filtered_words = [word for word in words if word.lower() not in prefixes_to_remove]
            if filtered_words:  # Make sure we don't end up with empty result
                words = filtered_words
        
        result = ' '.join(words).strip()
        result = re.sub(r'[,;\.!?\s]+$', '', result)
        
        return result if len(result) > 1 else term.strip()
    
    async def map_predicates_to_concepts(self, predicates: List) -> List[Dict]:
        """Map all predicates to normalized medical concepts"""
        mapped_predicates = []
        
        logger.info(f"Mapping {len(predicates)} predicates to medical concepts")
        
        for predicate in predicates:
            # Extract medical term from predicate object
            medical_term = predicate.object
            
            # Handle compound terms (like lab_value:troponin:0.8)
            if ':' in medical_term:
                parts = medical_term.split(':')
                medical_term = parts[0]  # Use the first part for concept mapping
            
            # Map the medical term to a concept
            concept = await self.normalize_medical_term(medical_term)
            
            # Create mapped predicate
            mapped_predicate = predicate.to_dict()
            mapped_predicate['normalized_concept'] = concept.__dict__ if concept else None
            mapped_predicate['original_object'] = predicate.object
            
            # Update object with normalized term if found
            if concept:
                # Preserve any additional information (like lab values)
                if ':' in predicate.object and len(predicate.object.split(':')) > 1:
                    parts = predicate.object.split(':')
                    normalized_object = concept.preferred_name.lower()
                    if len(parts) > 1:
                        normalized_object = ':'.join([normalized_object] + parts[1:])
                else:
                    normalized_object = concept.preferred_name.lower()
                
                mapped_predicate['object'] = normalized_object
                mapped_predicate['semantic_type'] = concept.semantic_types[0] if concept.semantic_types else "Unknown"
            
            mapped_predicates.append(mapped_predicate)
        
        logger.info(f"Successfully mapped {len(mapped_predicates)} predicates to concepts")
        return mapped_predicates
    
    def get_concept_relationships(self, cui: str) -> Dict[str, List[str]]:
        """Get relationships for a medical concept (simplified)"""
        # This would interface with a full ontology in production
        relationships = {
            "is_a": [],
            "part_of": [],
            "treats": [],
            "causes": [],
            "associated_with": []
        }
        
        # Example relationships for common concepts
        concept_relationships = {
            "C0008031": {  # Chest Pain
                "associated_with": ["myocardial infarction", "angina", "pulmonary embolism"],
                "causes": ["cardiac ischemia", "muscle strain", "gastroesophageal reflux"]
            },
            "C0020538": {  # Hypertension
                "causes": ["cardiovascular disease", "stroke", "kidney disease"],
                "treats": ["lisinopril", "amlodipine", "hydrochlorothiazide"]
            },
            "C0027051": {  # Myocardial Infarction
                "associated_with": ["chest pain", "dyspnea", "elevated troponin"],
                "causes": ["coronary artery occlusion", "atherosclerosis"]
            }
        }
        
        return concept_relationships.get(cui, relationships)
    
    async def validate_concept_relationships(self, predicates: List[Dict]) -> Dict[str, float]:
        """Validate logical consistency between mapped concepts"""
        consistency_scores = {}
        
        # Extract all concepts
        concepts = []
        for predicate in predicates:
            concept = predicate.get('normalized_concept')
            if concept:
                concepts.append(concept)
        
        # Check for logical consistency
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                relationship_score = self._calculate_concept_similarity(concept1, concept2)
                pair_key = f"{concept1['cui']}-{concept2['cui']}"
                consistency_scores[pair_key] = relationship_score
        
        return consistency_scores
    
    def _calculate_concept_similarity(self, concept1: Dict, concept2: Dict) -> float:
        """Calculate semantic similarity between two concepts"""
        # Simple implementation - in production, use semantic embeddings
        semantic_type_similarity = 1.0 if concept1.get('semantic_types', [None])[0] == concept2.get('semantic_types', [None])[0] else 0.5

        # Check for common synonyms
        synonyms1 = set(syn.lower() for syn in concept1.get('synonyms', []))
        synonyms2 = set(syn.lower() for syn in concept2.get('synonyms', []))
        synonym_overlap = len(synonyms1.intersection(synonyms2))

        # Combine scores
        similarity = semantic_type_similarity * 0.7 + min(synonym_overlap * 0.3, 0.3)

        return min(similarity, 1.0)

    def _calculate_fuzzy_similarity(self, term1: str, term2: str) -> float:
        """
        Calculate fuzzy similarity between two terms using Levenshtein distance
        
        Args:
            term1: First term to compare
            term2: Second term to compare
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not term1 or not term2:
            return 0.0
            
        # Normalize terms
        term1 = term1.lower().strip()
        term2 = term2.lower().strip()
        
        if term1 == term2:
            return 1.0
            
        # Simple Levenshtein distance implementation
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                s1, s2 = s2, s1
                
            if len(s2) == 0:
                return len(s1)
                
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
                
            return previous_row[-1]
        
        distance = levenshtein_distance(term1, term2)
        max_len = max(len(term1), len(term2))
        
        if max_len == 0:
            return 1.0
            
        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)

    # Enhanced Ontology Mapping Methods

    async def normalize_medical_term_enhanced(self, term: str) -> Optional[MedicalConcept]:
        """
        Enhanced medical term normalization using UMLS and Neo4j

        Args:
            term: Medical term to normalize

        Returns:
            Normalized medical concept with enhanced information
        """
        if not term:
            return None

        cleaned_term = self._clean_term(term)
        logger.debug(f"Enhanced normalizing term: '{term}' -> '{cleaned_term}'")

        # Try Neo4j knowledge graph first
        if self.neo4j_service:
            try:
                async with self.neo4j_service as neo4j:
                    concepts = await neo4j.search_concepts(cleaned_term, limit=5)
                    if concepts:
                        best_concept = concepts[0]  # Already sorted by score
                        return MedicalConcept(
                            cui=best_concept["cui"],
                            preferred_name=best_concept["preferred_name"],
                            synonyms=best_concept["synonyms"],
                            semantic_types=best_concept["semantic_types"],
                            definition=best_concept["definition"]
                        )
            except Exception as e:
                logger.warning(f"Neo4j search failed: {str(e)}")

        # Try UMLS API
        if self.umls_client:
            try:
                async with self.umls_client as umls:
                    results = await umls.search_concepts(cleaned_term, page_size=5)
                    if results:
                        best_result = results[0]  # Already sorted by score
                        # Get full concept details
                        concept_details = await umls.get_concept_details(best_result.cui)
                        if concept_details:
                            logger.info(f"Found UMLS concept: {concept_details.cui}")
                            return MedicalConcept(
                                cui=concept_details.cui,
                                preferred_name=concept_details.preferred_name,
                                synonyms=concept_details.synonyms,
                                semantic_types=concept_details.semantic_types,
                                definition=concept_details.definitions[0] if concept_details.definitions else ""
                            )
            except Exception as e:
                logger.warning(f"UMLS search failed: {str(e)}")
                if "authentication failed" in str(e).lower() or "invalid api key" in str(e).lower():
                    logger.warning("UMLS API key is invalid - disabling UMLS client for this session")
                    self.umls_client = None  # Disable UMLS client to avoid repeated failures



        # Fallback to basic normalization
        logger.info(f"Using fallback normalization for term: {cleaned_term}")
        return await self.normalize_medical_term(cleaned_term)

    async def find_synonyms_enhanced(self, term: str) -> List[str]:
        """
        Find synonyms for a medical term using enhanced services

        Args:
            term: Medical term to find synonyms for

        Returns:
            List of synonyms
        """
        synonyms = []

        # Try Neo4j knowledge graph
        if self.neo4j_service:
            try:
                async with self.neo4j_service as neo4j:
                    graph_synonyms = await neo4j.find_concept_synonyms(term)
                    synonyms.extend(graph_synonyms)
            except Exception as e:
                logger.warning(f"Neo4j synonym search failed: {str(e)}")

        # Try UMLS API
        if self.umls_client:
            try:
                async with self.umls_client as umls:
                    results = await umls.search_concepts(term, page_size=3)
                    for result in results:
                        if result.synonyms:
                            synonyms.extend(result.synonyms)
            except Exception as e:
                logger.warning(f"UMLS synonym search failed: {str(e)}")



        # Remove duplicates and original term
        unique_synonyms = list(set(synonyms))
        if term in unique_synonyms:
            unique_synonyms.remove(term)

        logger.info(f"Found {len(unique_synonyms)} enhanced synonyms for term: {term}")
        return unique_synonyms

    async def get_concept_relationships_enhanced(self, cui: str) -> Dict[str, List[str]]:
        """
        Get enhanced concept relationships using knowledge graph

        Args:
            cui: Concept Unique Identifier

        Returns:
            Dictionary of relationship types and related concepts
        """
        relationships = {
            "is_a": [],
            "part_of": [],
            "treats": [],
            "causes": [],
            "associated_with": [],
            "related_to": []
        }

        # Try Neo4j knowledge graph
        if self.neo4j_service:
            try:
                async with self.neo4j_service as neo4j:
                    related_concepts = await neo4j.get_related_concepts(cui, max_depth=2)

                    for concept in related_concepts:
                        rel_type = concept.get("relationship_type", "related_to")
                        if rel_type in relationships:
                            relationships[rel_type].append(concept.get("preferred_name", ""))
                        else:
                            relationships["related_to"].append(concept.get("preferred_name", ""))

                logger.info(f"Found enhanced relationships for CUI: {cui}")
                return relationships

            except Exception as e:
                logger.warning(f"Neo4j relationship search failed: {str(e)}")

        # Try UMLS API
        if self.umls_client:
            try:
                async with self.umls_client as umls:
                    concept_details = await umls.get_concept_details(cui)
                    if concept_details:
                        for relation in concept_details.relations:
                            rel_type = relation.get("relation", "").lower()
                            related_name = relation.get("relatedIdName", "")

                            # Map UMLS relationship types to our categories
                            if rel_type in ["isa", "is_a"]:
                                relationships["is_a"].append(related_name)
                            elif rel_type in ["part_of", "has_part"]:
                                relationships["part_of"].append(related_name)
                            elif rel_type == "treats":
                                relationships["treats"].append(related_name)
                            elif rel_type == "causes":
                                relationships["causes"].append(related_name)
                            else:
                                relationships["associated_with"].append(related_name)

                logger.info(f"Found UMLS relationships for CUI: {cui}")
                return relationships

            except Exception as e:
                logger.warning(f"UMLS relationship search failed: {str(e)}")

        # Fallback to basic relationships
        return self.get_concept_relationships(cui)

    async def validate_concept_consistency_enhanced(self, concepts: List[str]) -> Dict[str, float]:
        """
        Validate consistency between concepts using knowledge graph

        Args:
            concepts: List of concept CUIs to validate

        Returns:
            Dictionary mapping concept pairs to consistency scores
        """
        consistency_scores = {}

        if not self.neo4j_service:
            # Fallback to basic similarity calculation
            for i, cui1 in enumerate(concepts):
                for cui2 in concepts[i+1:]:
                    concept1 = {"cui": cui1}
                    concept2 = {"cui": cui2}
                    score = self._calculate_concept_similarity(concept1, concept2)
                    consistency_scores[f"{cui1}-{cui2}"] = score
            return consistency_scores

        try:
            async with self.neo4j_service as neo4j:
                for i, cui1 in enumerate(concepts):
                    for cui2 in concepts[i+1:]:
                        try:
                            similarity = await neo4j.get_concept_similarity(cui1, cui2)
                            consistency_scores[f"{cui1}-{cui2}"] = similarity
                        except Exception as e:
                            logger.warning(f"Failed to calculate similarity for {cui1}-{cui2}: {str(e)}")
                            consistency_scores[f"{cui1}-{cui2}"] = 0.0

        except Exception as e:
            logger.error(f"Enhanced consistency validation failed: {str(e)}")
            # Fallback to basic validation
            return await self.validate_concept_relationships([])

        logger.info(f"Calculated consistency scores for {len(consistency_scores)} concept pairs")
        return consistency_scores

    async def import_medical_concepts(self, concepts_data: List[Dict[str, any]]) -> int:
        """
        Import medical concepts into the knowledge graph

        Args:
            concepts_data: List of concept dictionaries with cui, preferred_name, synonyms, etc.

        Returns:
            Number of concepts successfully imported
        """
        if not self.neo4j_service:
            logger.warning("Neo4j service not available for concept import")
            return 0

        try:
            async with self.neo4j_service as neo4j:
                imported_count = await neo4j.batch_import_concepts(concepts_data, "bulk_import")

            logger.info(f"Successfully imported {imported_count} concepts into knowledge graph")
            return imported_count

        except Exception as e:
            logger.error(f"Failed to import concepts: {str(e)}")
            return 0

    async def get_knowledge_subgraph(self, central_concept: str, depth: int = 2) -> Optional[Dict]:
        """
        Get knowledge subgraph around a central concept

        Args:
            central_concept: Central concept CUI
            depth: Traversal depth

        Returns:
            Dictionary containing nodes and relationships
        """
        if not self.neo4j_service:
            logger.warning("Neo4j service not available for subgraph query")
            return None

        try:
            async with self.neo4j_service as neo4j:
                result = await neo4j.get_subgraph(central_concept, depth)

                # Convert to dictionary format
                subgraph = {
                    "nodes": [
                        {
                            "id": node.id,
                            "labels": node.labels,
                            "properties": node.properties
                        }
                        for node in result.nodes
                    ],
                    "relationships": [
                        {
                            "id": rel.id,
                            "type": rel.type,
                            "start_node": rel.start_node_id,
                            "end_node": rel.end_node_id,
                            "properties": rel.properties
                        }
                        for rel in result.relationships
                    ],
                    "metadata": result.metadata
                }

            logger.info(f"Retrieved subgraph with {len(subgraph['nodes'])} nodes and {len(subgraph['relationships'])} relationships")
            return subgraph

        except Exception as e:
            logger.error(f"Failed to get knowledge subgraph: {str(e)}")
            return None

    async def search_medical_literature(self, query: str, max_results: int = 10) -> List[Dict[str, any]]:
        """
        Search medical literature and concepts related to a query

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of relevant medical concepts and information
        """
        results = []

        # Search Neo4j knowledge graph
        if self.neo4j_service:
            try:
                async with self.neo4j_service as neo4j:
                    graph_results = await neo4j.search_concepts(query, limit=max_results)
                    for result in graph_results:
                        result["source"] = "knowledge_graph"
                        results.append(result)
            except Exception as e:
                logger.warning(f"Neo4j literature search failed: {str(e)}")

        # Search UMLS
        if self.umls_client:
            try:
                async with self.umls_client as umls:
                    umls_results = await umls.search_concepts(query, page_size=max_results)
                    for result in umls_results:
                        results.append({
                            "cui": result.cui,
                            "preferred_name": result.name,
                            "synonyms": result.synonyms,
                            "semantic_types": result.semantic_types,
                            "source": "umls",
                            "score": result.score
                        })
            except Exception as e:
                logger.warning(f"UMLS literature search failed: {str(e)}")



        # Fallback: Search local medical concept database if no external results
        if not results:
            try:
                logger.info("External APIs failed, using local medical concept database as fallback")
                
                # Search in local concepts
                normalized_concept = await self.normalize_medical_term(query)
                if normalized_concept and normalized_concept.cui != f"C{abs(hash(query)) % 9999999:07d}":
                    # This is a known concept, not a generated one
                    results.append({
                        "cui": normalized_concept.cui,
                        "preferred_name": normalized_concept.preferred_name,
                        "synonyms": normalized_concept.synonyms,
                        "semantic_types": normalized_concept.semantic_types,
                        "definition": normalized_concept.definition or "",
                        "source": "local_database",
                        "score": 1.0
                    })
                
                # Also try fuzzy matching on all known concepts
                query_lower = query.lower()
                for term, concept in self.medical_concepts.items():
                    if (query_lower in term.lower() or 
                        term.lower() in query_lower or
                        any(query_lower in synonym.lower() for synonym in concept.synonyms)):
                        
                        results.append({
                            "cui": concept.cui,
                            "preferred_name": concept.preferred_name,
                            "synonyms": concept.synonyms,
                            "semantic_types": concept.semantic_types,
                            "definition": concept.definition or "",
                            "source": "local_database",
                            "score": 0.8
                        })
                        
            except Exception as e:
                logger.warning(f"Local database search failed: {str(e)}")

        # Remove duplicates based on CUI
        seen_cuis = set()
        unique_results = []
        for result in results:
            cui = result.get("cui", "")
            if cui and cui not in seen_cuis:
                seen_cuis.add(cui)
                unique_results.append(result)

        logger.info(f"Found {len(unique_results)} unique medical literature results for query: {query}")
        return unique_results[:max_results]

    def analyze_clinical_text(self, text: str) -> Dict[str, any]:
        """
        Analyze clinical text and extract normalized medical terms

        Args:
            text: Clinical text to analyze

        Returns:
            Dictionary with extracted terms, normalized terms, and analysis metadata
        """
        if not text:
            return {
                "extracted_terms": [],
                "normalized_terms": [],
                "term_count": 0,
                "source": "fallback",
                "confidence": 0.0,
                "error": "No text provided"
            }

        try:
            logger.info(f"Analyzing clinical text of length {len(text)}")

            # Extract medical terms using pattern matching
            extracted_terms = self._extract_medical_terms(text)

            if not extracted_terms:
                return {
                    "extracted_terms": [],
                    "normalized_terms": [],
                    "term_count": 0,
                    "source": "fallback",
                    "confidence": 0.0,
                    "message": "No medical terms detected in text"
                }

            # Normalize each term
            normalized_terms = []
            total_confidence = 0.0

            for term in extracted_terms:
                try:
                    # Handle async normalization with proper event loop management
                    if self.use_enhanced_services:
                        try:
                            # Check if we're in an async context
                            loop = asyncio.get_running_loop()
                            # We're in an async context, use sync fallback
                            normalized_concept = self.normalize_medical_term_sync(term)
                        except RuntimeError:
                            # No running loop, safe to use asyncio.run
                            try:
                                normalized_concept = asyncio.run(self.normalize_medical_term_enhanced(term))
                            except RuntimeError as e:
                                if "cannot be called from a running event loop" in str(e):
                                    # Fallback to sync method
                                    normalized_concept = self.normalize_medical_term_sync(term)
                                else:
                                    raise
                    else:
                        try:
                            # Check if we're in an async context
                            loop = asyncio.get_running_loop()
                            # We're in an async context, use sync fallback
                            normalized_concept = self.normalize_medical_term_sync(term)
                        except RuntimeError:
                            # No running loop, safe to use asyncio.run
                            normalized_concept = asyncio.run(self.normalize_medical_term(term))

                    if normalized_concept:
                        normalized_term = {
                            "original_term": term,
                            "normalized_term": normalized_concept.preferred_name,
                            "cui": normalized_concept.cui,
                            "definition": normalized_concept.definition,
                            "semantic_types": normalized_concept.semantic_types,
                            "confidence": 0.9 if normalized_concept.cui.startswith('C') and len(normalized_concept.cui) > 3 else 0.6
                        }
                        normalized_terms.append(normalized_term)
                        total_confidence += normalized_term["confidence"]
                    else:
                        # Create fallback entry
                        normalized_term = {
                            "original_term": term,
                            "normalized_term": term.title(),
                            "cui": None,
                            "definition": None,
                            "semantic_types": ["Unknown"],
                            "confidence": 0.3
                        }
                        normalized_terms.append(normalized_term)
                        total_confidence += 0.3

                except Exception as e:
                    logger.warning(f"Failed to normalize term '{term}': {str(e)}")
                    # Add as unnormalized
                    normalized_term = {
                        "original_term": term,
                        "normalized_term": term.title(),
                        "cui": None,
                        "definition": None,
                        "semantic_types": ["Unknown"],
                        "confidence": 0.1
                    }
                    normalized_terms.append(normalized_term)
                    total_confidence += 0.1

            # Calculate overall confidence
            avg_confidence = total_confidence / len(extracted_terms) if extracted_terms else 0.0

            # Determine source based on available services
            source = "enhanced" if self.use_enhanced_services else "fallback"

            result = {
                "extracted_terms": extracted_terms,
                "normalized_terms": normalized_terms,
                "term_count": len(extracted_terms),
                "source": source,
                "confidence": avg_confidence,
                "analysis_timestamp": datetime.now().isoformat()
            }

            logger.info(f"Clinical text analysis complete: {len(extracted_terms)} terms extracted, {len(normalized_terms)} normalized")

            return result

        except Exception as e:
            logger.error(f"Clinical text analysis failed: {str(e)}")
            return {
                "extracted_terms": [],
                "normalized_terms": [],
                "term_count": 0,
                "source": "error",
                "confidence": 0.0,
                "error": str(e)
            }

    def _extract_medical_terms(self, text: str) -> List[str]:
        """
        Extract medical terms from clinical text using pattern matching

        Args:
            text: Clinical text to extract terms from

        Returns:
            List of extracted medical terms
        """
        if not text:
            return []

        # Convert to lowercase for processing
        text_lower = text.lower()

        # Medical term patterns to look for
        medical_patterns = [
            # Medical conditions and diseases
            r'\b(?:acute|chronic|severe|mild|moderate)?\s*(?:heart|cardiac|myocardial|coronary|pulmonary|respiratory|renal|kidney|hepatic|liver|gastrointestinal|neurological|psychiatric)\s+(?:disease|failure|attack|infarction|syndrome|disorder|condition)\b',

            # Vital signs and measurements
            r'\b(?:blood pressure|bp|heart rate|hr|temperature|temp|respiratory rate|rr|oxygen saturation|spo2|pulse)\b',

            # Laboratory values
            r'\b(?:glucose|creatinine|troponin|cholesterol|triglycerides|hdl|ldl|bilirubin|albumin|hemoglobin|hematocrit|platelets|white blood cells|wbc|red blood cells|rbc)\b',

            # Medications
            r'\b(?:aspirin|lisinopril|metformin|atorvastatin|simvastatin|amlodipine|hydrochlorothiazide|warfarin|heparin|insulin|metoprolol|atenolol|losartan|captopril|enalapril)\b',

            # Symptoms and signs
            r'\b(?:chest pain|dyspnea|shortness of breath|nausea|vomiting|dizziness|syncope|palpitations|edema|fatigue|weakness|confusion|headache|abdominal pain|cough|fever|chills|rash|sweating)\b',

            # Medical procedures and tests
            r'\b(?:echocardiogram|ecg|ekg|electrocardiogram|chest x-ray|cxr|ct scan|mri|ultrasound|catheterization|angiogram|endoscopy|colonoscopy|biopsy)\b',

            # Common medical abbreviations (expanded)
            r'\b(?:mi|cad|chf|copd|dm|htn|cva|tia|pe|dvt|afib|vtach|vfib|acs|stemi|nstemi)\b'
        ]

        extracted_terms = set()

        # Apply each pattern
        for pattern in medical_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Clean the match
                clean_match = match.strip()
                if len(clean_match) > 2:  # Avoid very short matches
                    extracted_terms.add(clean_match.title())  # Title case for consistency

        # Also check against our known medical concepts
        for concept_key in self.medical_concepts.keys():
            if concept_key.lower() in text_lower:
                extracted_terms.add(concept_key.title())

        # Check for abbreviations
        for abbrev, expansion in self.medical_synonyms.items():
            if abbrev.lower() in text_lower:
                extracted_terms.add(expansion.title())

        # Remove duplicates and sort
        result = list(extracted_terms)
        result.sort()

        logger.debug(f"Extracted {len(result)} medical terms from text")
        return result

    # Public API methods for backward compatibility
    @cached_result(ttl_seconds=3600)  # Cache for 1 hour
    def normalize_term(self, term: str) -> Dict[str, any]:
        """
        Public method to normalize a medical term (synchronous wrapper with caching)

        Args:
            term: Medical term to normalize

        Returns:
            Dictionary with normalized term information
        """
        try:
            # Check cache first
            cache = get_cache()
            cache_key_components = ontology_cache_key(term, source='sync_wrapper')

            cached_result = cache.get(cache_key_components)
            if cached_result is not None:
                logger.debug(f"Cache hit for term: {term}")
                return cached_result

            # Use asyncio.run with proper error handling for both enhanced and basic normalization
            if self.use_enhanced_services:
                try:
                    concept = asyncio.run(self.normalize_medical_term_enhanced(term))
                except RuntimeError as e:
                    if "cannot be called from a running event loop" in str(e):
                        # Fallback to sync method
                        concept = self.normalize_medical_term_sync(term)
                    else:
                        raise
            else:
                try:
                    concept = asyncio.run(self.normalize_medical_term(term))
                except RuntimeError as e:
                    if "cannot be called from a running event loop" in str(e):
                        # Fallback to sync method
                        concept = self.normalize_medical_term_sync(term)
                    else:
                        raise

            result = None
            if concept:
                result = {
                    "normalized_term": concept.preferred_name,
                    "cui": concept.cui,
                    "definition": concept.definition,
                    "source": "enhanced" if self.use_enhanced_services else "fallback",
                    "confidence": 0.9 if concept.cui and not concept.cui.startswith('C') else 0.6
                }
            else:
                result = {
                    "normalized_term": term.title(),
                    "cui": None,
                    "definition": None,
                    "source": "not_found",
                    "confidence": 0.0
                }

            # Cache the result
            cache.set(cache_key_components, result)
            return result

        except Exception as e:
            logger.error(f"Term normalization failed: {str(e)}")
            error_result = {
                "normalized_term": term.title(),
                "cui": None,
                "definition": None,
                "source": "error",
                "confidence": 0.0,
                "error": str(e)
            }
            return error_result

    @cached_result(ttl_seconds=1800)  # Cache for 30 minutes
    def get_synonyms(self, term: str) -> Dict[str, any]:
        """
        Public method to get synonyms for a medical term (synchronous wrapper with caching)

        Args:
            term: Medical term to find synonyms for

        Returns:
            Dictionary with synonyms and metadata
        """
        try:
            # Check cache first
            cache = get_cache()
            cache_key = medical_term_cache_key(term, domain='synonyms')

            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for synonyms: {term}")
                return cached_result

            # Use asyncio.run with proper error handling
            synonyms = asyncio.run(self.find_synonyms_enhanced(term))

            result = {
                "synonyms": synonyms,
                "count": len(synonyms),
                "source": "enhanced" if self.use_enhanced_services else "fallback",
                "confidence": 0.8 if synonyms else 0.5
            }

            # Cache the result
            cache.set(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Synonym lookup failed: {str(e)}")
            error_result = {
                "synonyms": [],
                "count": 0,
                "source": "error",
                "confidence": 0.0,
                "error": str(e)
            }
            return error_result

    def search_comprehensive(self, query: str, limit: int = 10) -> Dict[str, any]:
        """
        Public method for comprehensive medical search (synchronous wrapper)

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            Dictionary with search results
        """
        try:
            # Use synchronous search method to avoid asyncio.run() issues
            # Call the async method with proper error handling
            try:
                # Check if we're in an async context
                loop = asyncio.get_running_loop()
                # We're in an async context, this shouldn't happen for sync methods
                raise RuntimeError("Cannot call sync method from async context")
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    # We're in async context, but this is a sync method - this is unexpected
                    raise RuntimeError("Sync method called from async context")
                else:
                    # No running loop, safe to use asyncio.run
                    try:
                        search_result = asyncio.run(self.search_medical_literature(query, max_results=limit))
                        results = search_result
                    except RuntimeError as e:
                        if "cannot be called from a running event loop" in str(e):
                            # Fallback: return empty results
                            results = []
                        else:
                            raise

            return {
                "results": results,
                "count": len(results) if results else 0,
                "search_type": "comprehensive",
                "source": "enhanced" if self.use_enhanced_services else "fallback"
            }
        except Exception as e:
            logger.error(f"Comprehensive search failed: {str(e)}")
            return {
                "results": [],
                "count": 0,
                "search_type": "comprehensive",
                "source": "error",
                "error": str(e)
            }

    def get_config_status(self) -> Dict[str, any]:
        """
        Get the configuration status of ontology services

        Returns:
            Dictionary with service status information
        """
        return {
            "is_configured": self.use_enhanced_services,
            "neo4j_configured": self.neo4j_service is not None,
            "umls_configured": self.umls_client is not None,
            "fallback_available": True,
            "enhanced_services_enabled": self.use_enhanced_services
        }

    # Enhanced Knowledge Graph Reasoning Integration

    async def analyze_patient_symptoms_enhanced(self, symptoms: List[str],
                                              patient_context: Dict[str, any] = None) -> Dict[str, any]:
        """
        Perform comprehensive patient analysis using enhanced knowledge graph reasoning

        Args:
            symptoms: List of patient symptoms
            patient_context: Additional patient information

        Returns:
            Comprehensive analysis results
        """
        try:
            # Initialize enhanced knowledge graph service
            enhanced_service = EnhancedKnowledgeGraphService(self.neo4j_service)

            # Perform comprehensive analysis
            result = await enhanced_service.analyze_patient_symptoms(symptoms, patient_context)

            # Convert to dictionary format
            analysis_result = {
                "symptom_clusters": [
                    {
                        "cluster_id": cluster.cluster_id,
                        "symptoms": cluster.symptoms,
                        "common_diseases": cluster.common_diseases,
                        "severity_score": cluster.severity_score,
                        "confidence": cluster.confidence,
                        "reasoning_path": cluster.reasoning_path
                    }
                    for cluster in result.symptom_clusters
                ],
                "drug_interactions": [
                    {
                        "drug1": interaction.drug1,
                        "drug2": interaction.drug2,
                        "interaction_type": interaction.interaction_type,
                        "severity": interaction.severity,
                        "description": interaction.description,
                        "evidence_level": interaction.evidence_level,
                        "recommendations": interaction.recommendations
                    }
                    for interaction in result.drug_interactions
                ],
                "comorbidity_analysis": {
                    "primary_condition": result.comorbidity_analysis.primary_condition,
                    "comorbidities": result.comorbidity_analysis.comorbidities,
                    "risk_factors": result.comorbidity_analysis.risk_factors,
                    "management_recommendations": result.comorbidity_analysis.management_recommendations,
                    "evidence_strength": result.comorbidity_analysis.evidence_strength
                },
                "execution_time_ms": result.execution_time_ms,
                "reasoning_paths": result.reasoning_paths,
                "analysis_timestamp": datetime.now().isoformat()
            }

            logger.info(f"Enhanced patient analysis completed in {result.execution_time_ms:.2f}ms")
            return analysis_result

        except Exception as e:
            logger.error(f"Enhanced patient analysis failed: {str(e)}")
            return {
                "symptom_clusters": [],
                "drug_interactions": [],
                "comorbidity_analysis": {},
                "execution_time_ms": 0,
                "reasoning_paths": [],
                "error": str(e)
            }

    async def optimize_knowledge_graph_performance(self, query_type: str) -> Dict[str, any]:
        """
        Optimize knowledge graph performance for specific query types

        Args:
            query_type: Type of query to optimize ("symptom_clustering", "drug_interaction", "comorbidity_analysis")

        Returns:
            Optimization results
        """
        try:
            enhanced_service = EnhancedKnowledgeGraphService(self.neo4j_service)
            result = await enhanced_service.optimize_query_performance(query_type)

            logger.info(f"Performance optimization completed for {query_type}")
            return result

        except Exception as e:
            logger.error(f"Performance optimization failed: {str(e)}")
            return {
                "status": "error",
                "query_type": query_type,
                "error": str(e)
            }

    def get_enhanced_reasoning_status(self) -> Dict[str, any]:
        """
        Get the status of enhanced reasoning capabilities

        Returns:
            Dictionary with reasoning capability status
        """
        enhanced_available = False
        capabilities = []

        # Check if enhanced services are available
        if self.neo4j_service:
            enhanced_available = True
            capabilities.append("graph-based_symptom_clustering")
            capabilities.append("drug_interaction_analysis")
            capabilities.append("comorbidity_reasoning")
            capabilities.append("optimized_cypher_queries")
            capabilities.append("real_time_performance")

        if self.umls_client:
            capabilities.append("umls_integration")

        return {
            "enhanced_reasoning_available": enhanced_available,
            "capabilities": capabilities,
            "neo4j_connected": self.neo4j_service is not None,
            "umls_available": self.umls_client is not None,
            "performance_optimization_ready": enhanced_available
        }
    
    def normalize_medical_term_sync(self, term: str) -> Optional[MedicalConcept]:
        """
        Synchronous fallback method for medical term normalization
        Uses basic pattern matching and local database lookup without async calls
        """
        try:
            logger.debug(f"Normalizing term (sync): '{term}'")
            
            # Clean and preprocess the term
            cleaned_term = term.strip().lower()
            if not cleaned_term or len(cleaned_term) < 2:
                return None
            
            # Use synonym mapping if available
            synonym_mapping = getattr(self, 'synonym_mapping', {
                'chest pain': 'Chest Pain',
                'shortness of breath': 'Dyspnea',
                'abdominal pain': 'Abdominal Pain',
                'headache': 'Headache',
                'fever': 'Fever',
                'mi': 'Myocardial Infarction',
                'pe': 'Pulmonary Embolism'
            })
            
            if cleaned_term in synonym_mapping:
                canonical_term = synonym_mapping[cleaned_term]
                return MedicalConcept(
                    cui=f'SYNONYM_{hash(canonical_term) % 1000000}',
                    preferred_name=canonical_term,
                    synonyms=[cleaned_term],
                    semantic_types=['Medical_Term']
                )
            
            # Basic fuzzy matching with high-confidence terms
            best_match = None
            best_score = 0.0
            
            # Check against common medical terms
            common_terms = [
                'myocardial infarction', 'pulmonary embolism', 'respiratory rate',
                'blood pressure', 'heart rate', 'temperature', 'chest pain',
                'shortness of breath', 'abdominal pain', 'headache', 'fever',
                'diabetes', 'hypertension', 'hypotension', 'tachycardia', 'bradycardia'
            ]
            
            for common_term in common_terms:
                score = self._calculate_fuzzy_similarity(cleaned_term, common_term)
                if score > best_score and score > 0.8:
                    best_score = score
                    best_match = common_term
            
            if best_match:
                return MedicalConcept(
                    cui=f'FUZZY_{hash(best_match) % 1000000}',
                    preferred_name=best_match.title(),
                    synonyms=[cleaned_term],
                    semantic_types=['Medical_Term']
                )
            
            # Final fallback - return the original term with low confidence
            return MedicalConcept(
                cui=f'FALLBACK_{hash(cleaned_term) % 1000000}',
                preferred_name=cleaned_term.title(),
                synonyms=[],
                semantic_types=['Unknown']
            )
            
        except Exception as e:
            logger.warning(f"Sync normalization failed for term '{term}': {e}")
            return None
