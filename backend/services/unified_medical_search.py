"""
Unified Medical Knowledge Search Service
Combines UMLS, Neo4j Knowledge Graph, and Ontology Mapping
Provides intelligent medical concept exploration and discovery
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import re
import json
from collections import defaultdict
import os

from services.umls_code_lookup_service import UMLSCodeLookupService
from services.neo4j_service import Neo4jService
from services.ontology_mapper import OntologyMapper

logger = logging.getLogger(__name__)

@dataclass
class MedicalConcept:
    """Unified medical concept representation"""
    cui: str
    preferred_name: str
    synonyms: List[str] = field(default_factory=list)
    definitions: List[str] = field(default_factory=list)
    semantic_types: List[str] = field(default_factory=list)
    source_vocabularies: List[str] = field(default_factory=list)
    confidence: float = 0.0
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    hierarchy_level: int = 0
    clinical_relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cui": self.cui,
            "preferred_name": self.preferred_name,
            "synonyms": self.synonyms,
            "definitions": self.definitions,
            "semantic_types": self.semantic_types,
            "source_vocabularies": self.source_vocabularies,
            "confidence": self.confidence,
            "relationships": self.relationships,
            "hierarchy_level": self.hierarchy_level,
            "clinical_relevance_score": self.clinical_relevance_score
        }

@dataclass
class ConceptRelationship:
    """Represents relationships between medical concepts"""
    source_cui: str
    target_cui: str
    relationship_type: str
    strength: float
    source: str
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_cui": self.source_cui,
            "target_cui": self.target_cui,
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "source": self.source,
            "additional_info": self.additional_info
        }

@dataclass
class SearchResult:
    """Comprehensive search result"""
    query: str
    concepts: List[MedicalConcept]
    relationships: List[ConceptRelationship]
    concept_hierarchy: Dict[str, List[str]]
    similar_concepts: List[MedicalConcept]
    clinical_context: Dict[str, Any]
    search_metadata: Dict[str, Any]
    execution_time: float
    total_results: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "concepts": [c.to_dict() for c in self.concepts],
            "relationships": [r.to_dict() for r in self.relationships],
            "concept_hierarchy": self.concept_hierarchy,
            "similar_concepts": [c.to_dict() for c in self.similar_concepts],
            "clinical_context": self.clinical_context,
            "search_metadata": self.search_metadata,
            "execution_time": self.execution_time,
            "total_results": self.total_results
        }

class UnifiedMedicalKnowledgeSearch:
    """
    Unified service for intelligent medical knowledge search
    Combines multiple knowledge sources for comprehensive results
    """
    
    def __init__(self):
        self.umls_service: Optional[UMLSCodeLookupService] = None
        self.neo4j_service: Optional[Neo4jService] = None
        self.ontology_mapper: Optional[OntologyMapper] = None
        
        # Clinical context weights for relevance scoring
        self.semantic_type_weights = {
            "Disease or Syndrome": 1.0,
            "Sign or Symptom": 0.9,
            "Therapeutic or Preventive Procedure": 0.8,
            "Pharmacologic Substance": 0.8,
            "Body Part, Organ, or Organ Component": 0.7,
            "Neoplastic Process": 0.9,
            "Mental or Behavioral Dysfunction": 0.85,
            "Congenital Abnormality": 0.8,
            "Finding": 0.7,
            "Clinical Attribute": 0.6
        }
        
        # Relationship type weights
        self.relationship_weights = {
            "isa": 1.0,
            "part_of": 0.8,
            "has_part": 0.8,
            "causes": 0.9,
            "treats": 0.9,
            "associated_with": 0.6,
            "related_to": 0.5
        }
    
    async def initialize_services(self):
        """Initialize all knowledge services"""
        try:
            # Initialize UMLS service with API key
            umls_api_key = os.getenv('UMLS_API_KEY')
            if umls_api_key:
                self.umls_service = UMLSCodeLookupService(api_key=umls_api_key)
            else:
                logger.warning("UMLS_API_KEY not found in environment variables")
                self.umls_service = None
            
            # Initialize Neo4j service
            self.neo4j_service = Neo4jService()
            await self.neo4j_service.initialize()
            
            # Initialize ontology mapper
            self.ontology_mapper = OntologyMapper(use_enhanced_services=True)
            
            logger.info("✅ Unified Medical Knowledge Search initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize knowledge services: {e}")
            raise
    
    async def intelligent_search(self, 
                                query: str, 
                                search_type: str = "comprehensive",
                                max_results: int = 20,
                                include_relationships: bool = True,
                                include_hierarchy: bool = True,
                                clinical_context: Optional[Dict[str, Any]] = None) -> SearchResult:
        """
        Perform intelligent medical knowledge search across all sources
        
        Args:
            query: Search query (disease, symptom, medication, etc.)
            search_type: Type of search ("comprehensive", "exact", "semantic", "fuzzy")
            max_results: Maximum number of results
            include_relationships: Whether to include concept relationships
            include_hierarchy: Whether to include concept hierarchy
            clinical_context: Additional clinical context for relevance
            
        Returns:
            Comprehensive search result
        """
        start_time = datetime.now()
        
        try:
            # Normalize and enhance query
            enhanced_query = await self._enhance_query(query, clinical_context)
            
            # Search across all sources in parallel
            search_tasks = []
            
            # UMLS search
            if self.umls_service:
                search_tasks.append(self._search_umls(enhanced_query, max_results))
            
            # Neo4j knowledge graph search
            if self.neo4j_service:
                search_tasks.append(self._search_neo4j(enhanced_query, max_results))
            
            # Ontology mapping search
            if self.ontology_mapper:
                search_tasks.append(self._search_ontology(enhanced_query, max_results))
            
            # Execute searches in parallel
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine and deduplicate results
            concepts = await self._merge_and_rank_concepts(search_results, clinical_context)
            
            # Get relationships if requested
            relationships = []
            if include_relationships:
                relationships = await self._get_concept_relationships(concepts[:10])  # Top 10 for relationships
            
            # Get concept hierarchy if requested
            concept_hierarchy = {}
            if include_hierarchy:
                concept_hierarchy = await self._build_concept_hierarchy(concepts[:5])  # Top 5 for hierarchy
            
            # Find similar concepts
            similar_concepts = await self._find_similar_concepts(concepts[0] if concepts else None, max_results=5)
            
            # Generate clinical context
            clinical_context_data = await self._generate_clinical_context(concepts, relationships)
            
            # Execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return SearchResult(
                query=query,
                concepts=concepts[:max_results],
                relationships=relationships,
                concept_hierarchy=concept_hierarchy,
                similar_concepts=similar_concepts,
                clinical_context=clinical_context_data,
                search_metadata={
                    "enhanced_query": enhanced_query,
                    "search_type": search_type,
                    "sources_used": self._get_active_sources(),
                    "query_processing_time": execution_time
                },
                execution_time=execution_time,
                total_results=len(concepts)
            )
            
        except Exception as e:
            logger.error(f"❌ Intelligent search failed: {e}")
            raise
    
    async def concept_exploration(self, cui: str, depth: int = 2) -> Dict[str, Any]:
        """
        Deep exploration of a specific medical concept
        
        Args:
            cui: Concept Unique Identifier
            depth: Exploration depth for relationships
            
        Returns:
            Comprehensive concept exploration data
        """
        try:
            exploration_data = {
                "concept": None,
                "detailed_relationships": [],
                "concept_network": {},
                "clinical_pathways": [],
                "treatment_options": [],
                "related_conditions": [],
                "diagnostic_criteria": []
            }
            
            # Get primary concept details
            if self.neo4j_service:
                concept_details = await self.neo4j_service.get_concept_details(cui)
                if concept_details:
                    exploration_data["concept"] = concept_details
            
            # Get detailed relationships
            if self.neo4j_service:
                relationships = await self.neo4j_service.get_related_concepts(cui, max_depth=depth)
                exploration_data["detailed_relationships"] = relationships
            
            # Build concept network
            exploration_data["concept_network"] = await self._build_concept_network(cui, depth)
            
            # Find clinical pathways
            exploration_data["clinical_pathways"] = await self._find_clinical_pathways(cui)
            
            # Find treatment options
            exploration_data["treatment_options"] = await self._find_treatment_options(cui)
            
            # Find related conditions
            exploration_data["related_conditions"] = await self._find_related_conditions(cui)
            
            return exploration_data
            
        except Exception as e:
            logger.error(f"❌ Concept exploration failed: {e}")
            return {"error": str(e)}
    
    async def clinical_decision_support(self, 
                                      symptoms: List[str],
                                      patient_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Provide clinical decision support based on symptoms and context
        
        Args:
            symptoms: List of patient symptoms
            patient_context: Patient demographics and history
            
        Returns:
            Clinical decision support recommendations
        """
        try:
            support_data = {
                "symptom_analysis": [],
                "differential_diagnosis": [],
                "recommended_tests": [],
                "treatment_pathways": [],
                "risk_factors": [],
                "drug_interactions": [],
                "clinical_guidelines": []
            }
            
            # Analyze each symptom
            for symptom in symptoms:
                analysis = await self.intelligent_search(
                    query=symptom,
                    search_type="comprehensive",
                    max_results=5,
                    clinical_context=patient_context
                )
                support_data["symptom_analysis"].append({
                    "symptom": symptom,
                    "concepts": [c.to_dict() for c in analysis.concepts],
                    "relationships": [r.to_dict() for r in analysis.relationships]
                })
            
            # Generate differential diagnosis
            support_data["differential_diagnosis"] = await self._generate_differential_diagnosis(symptoms, patient_context)
            
            # Recommend diagnostic tests
            support_data["recommended_tests"] = await self._recommend_diagnostic_tests(symptoms, patient_context)
            
            # Find treatment pathways
            support_data["treatment_pathways"] = await self._find_treatment_pathways(symptoms, patient_context)
            
            return support_data
            
        except Exception as e:
            logger.error(f"❌ Clinical decision support failed: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _enhance_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Enhance search query with context and normalization"""
        try:
            # Basic query cleanup
            enhanced = re.sub(r'[^\w\s]', ' ', query).strip().lower()
            
            # Use ontology mapper for term normalization if available
            if self.ontology_mapper:
                normalized_result = self.ontology_mapper.normalize_term(query)
                if normalized_result.get('normalized_term'):
                    enhanced = normalized_result['normalized_term']
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            return query.strip().lower()
    
    async def _search_umls(self, query: str, max_results: int) -> List[MedicalConcept]:
        """Search UMLS knowledge base"""
        concepts = []
        try:
            if not self.umls_service:
                return concepts
            
            # Use UMLS client for search
            async with self.umls_service:
                # Search using UMLS search functionality
                search_results = await self.umls_service.umls_client.search_concepts(query, page_size=max_results)
                
                for result in search_results:
                    # Get detailed concept information
                    concept_details = await self.umls_service.umls_client.get_concept_details(result.cui)
                    # Get detailed concept information
                    concept_details = await self.umls_service.umls_client.get_concept_details(result.cui)
                    
                    if concept_details:
                        concept = MedicalConcept(
                            cui=concept_details.cui,
                            preferred_name=concept_details.preferred_name,
                            synonyms=concept_details.synonyms,
                            definitions=concept_details.definitions,
                            semantic_types=concept_details.semantic_types,
                            source_vocabularies=['UMLS'],
                            confidence=result.score if hasattr(result, 'score') else 0.8
                        )
                        concepts.append(concept)
                        
        except Exception as e:
            logger.warning(f"UMLS search failed: {e}")
        
        return concepts
    
    async def _search_neo4j(self, query: str, max_results: int) -> List[MedicalConcept]:
        """Search Neo4j knowledge graph"""
        concepts = []
        try:
            if not self.neo4j_service:
                return concepts
            
            # Search Neo4j
            search_results = await self.neo4j_service.search_concepts(query, limit=max_results)
            
            for result in search_results:
                concept = MedicalConcept(
                    cui=result.get('cui', ''),
                    preferred_name=result.get('preferred_name', ''),
                    synonyms=result.get('synonyms', []),
                    definitions=[result.get('definition', '')] if result.get('definition') else [],
                    semantic_types=result.get('semantic_types', []),
                    source_vocabularies=['Neo4j'],
                    confidence=result.get('score', 0.0)
                )
                concepts.append(concept)
                
        except Exception as e:
            logger.warning(f"Neo4j search failed: {e}")
        
        return concepts
    
    async def _search_ontology(self, query: str, max_results: int) -> List[MedicalConcept]:
        """Search using ontology mapper"""
        concepts = []
        try:
            if not self.ontology_mapper:
                return concepts
            
            # Search ontology
            search_result = self.ontology_mapper.search_comprehensive(query, limit=max_results)
            results = search_result.get('results', [])
            
            for result in results:
                concept = MedicalConcept(
                    cui=result.get('cui', ''),
                    preferred_name=result.get('preferred_name', result.get('term', '')),
                    synonyms=result.get('synonyms', []),
                    definitions=[result.get('definition', '')] if result.get('definition') else [],
                    source_vocabularies=['Ontology'],
                    confidence=result.get('confidence', 0.0)
                )
                concepts.append(concept)
                
        except Exception as e:
            logger.warning(f"Ontology search failed: {e}")
        
        return concepts
    
    async def _merge_and_rank_concepts(self, search_results: List[Any], context: Optional[Dict[str, Any]] = None) -> List[MedicalConcept]:
        """Merge and rank concepts from all sources"""
        concepts_by_cui = {}
        all_concepts = []
        
        # Collect all concepts
        for result in search_results:
            if isinstance(result, Exception):
                continue
            if isinstance(result, list):
                all_concepts.extend(result)
        
        # Deduplicate by CUI
        for concept in all_concepts:
            if concept.cui and concept.cui in concepts_by_cui:
                # Merge concepts with same CUI
                existing = concepts_by_cui[concept.cui]
                existing.synonyms = list(set(existing.synonyms + concept.synonyms))
                existing.definitions = list(set(existing.definitions + concept.definitions))
                existing.source_vocabularies = list(set(existing.source_vocabularies + concept.source_vocabularies))
                existing.confidence = max(existing.confidence, concept.confidence)
            else:
                concepts_by_cui[concept.cui or f"no_cui_{len(concepts_by_cui)}"] = concept
        
        # Calculate clinical relevance scores
        for concept in concepts_by_cui.values():
            concept.clinical_relevance_score = self._calculate_clinical_relevance(concept, context)
        
        # Sort by clinical relevance and confidence
        sorted_concepts = sorted(
            concepts_by_cui.values(),
            key=lambda c: (c.clinical_relevance_score, c.confidence),
            reverse=True
        )
        
        return sorted_concepts
    
    def _calculate_clinical_relevance(self, concept: MedicalConcept, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate clinical relevance score for a concept"""
        score = concept.confidence
        
        # Weight by semantic types
        for semantic_type in concept.semantic_types:
            weight = self.semantic_type_weights.get(semantic_type, 0.5)
            score *= weight
        
        # Boost score for multiple source vocabularies
        if len(concept.source_vocabularies) > 1:
            score *= 1.2
        
        # Context-based scoring
        if context:
            if context.get('patient_age'):
                # Age-specific adjustments would go here
                pass
            if context.get('patient_gender'):
                # Gender-specific adjustments would go here
                pass
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def _get_concept_relationships(self, concepts: List[MedicalConcept]) -> List[ConceptRelationship]:
        """Get relationships between concepts"""
        relationships = []
        try:
            if not self.neo4j_service:
                return relationships
            
            for concept in concepts:
                if concept.cui:
                    related = await self.neo4j_service.get_related_concepts(concept.cui, max_depth=1)
                    for rel in related:
                        relationship = ConceptRelationship(
                            source_cui=concept.cui,
                            target_cui=rel.get('cui', ''),
                            relationship_type=rel.get('relationship_type', 'related_to'),
                            strength=rel.get('depth', 1.0),
                            source='Neo4j'
                        )
                        relationships.append(relationship)
        
        except Exception as e:
            logger.warning(f"Failed to get relationships: {e}")
        
        return relationships
    
    async def _build_concept_hierarchy(self, concepts: List[MedicalConcept]) -> Dict[str, List[str]]:
        """Build concept hierarchy for top concepts"""
        hierarchy = {}
        try:
            # This would build a hierarchical structure
            # For now, returning basic structure
            for concept in concepts:
                hierarchy[concept.preferred_name] = []
        except Exception as e:
            logger.warning(f"Failed to build hierarchy: {e}")
        
        return hierarchy
    
    async def _find_similar_concepts(self, concept: Optional[MedicalConcept], max_results: int = 5) -> List[MedicalConcept]:
        """Find concepts similar to the given concept"""
        if not concept or not self.neo4j_service:
            return []
        
        try:
            # Use Neo4j similarity search
            related = await self.neo4j_service.get_related_concepts(concept.cui, max_depth=2)
            similar_concepts = []
            
            for rel in related[:max_results]:
                similar_concept = MedicalConcept(
                    cui=rel.get('cui', ''),
                    preferred_name=rel.get('preferred_name', ''),
                    semantic_types=rel.get('semantic_types', []),
                    confidence=1.0 / (rel.get('depth', 1) + 1),  # Closer = higher confidence
                    source_vocabularies=['Neo4j']
                )
                similar_concepts.append(similar_concept)
            
            return similar_concepts
            
        except Exception as e:
            logger.warning(f"Failed to find similar concepts: {e}")
            return []
    
    async def _generate_clinical_context(self, concepts: List[MedicalConcept], relationships: List[ConceptRelationship]) -> Dict[str, Any]:
        """Generate clinical context from search results"""
        context = {
            "primary_semantic_types": [],
            "clinical_domains": [],
            "relationship_summary": {},
            "clinical_significance": ""
        }
        
        try:
            # Analyze semantic types
            semantic_counts = defaultdict(int)
            for concept in concepts:
                for sem_type in concept.semantic_types:
                    semantic_counts[sem_type] += 1
            
            context["primary_semantic_types"] = [
                {"type": k, "count": v} 
                for k, v in sorted(semantic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
            
            # Analyze relationships
            rel_counts = defaultdict(int)
            for rel in relationships:
                rel_counts[rel.relationship_type] += 1
            
            context["relationship_summary"] = dict(rel_counts)
            
            # Generate clinical significance
            if concepts:
                top_concept = concepts[0]
                if "Disease or Syndrome" in top_concept.semantic_types:
                    context["clinical_significance"] = "Primary pathological condition identified"
                elif "Sign or Symptom" in top_concept.semantic_types:
                    context["clinical_significance"] = "Clinical presentation or symptom identified"
                elif "Pharmacologic Substance" in top_concept.semantic_types:
                    context["clinical_significance"] = "Therapeutic agent or medication identified"
                else:
                    context["clinical_significance"] = "Medical concept with clinical relevance"
        
        except Exception as e:
            logger.warning(f"Failed to generate clinical context: {e}")
        
        return context
    
    def _get_active_sources(self) -> List[str]:
        """Get list of active knowledge sources"""
        sources = []
        if self.umls_service:
            sources.append("UMLS")
        if self.neo4j_service:
            sources.append("Neo4j")
        if self.ontology_mapper:
            sources.append("Ontology")
        return sources
    
    # Additional placeholder methods for clinical decision support
    
    async def _build_concept_network(self, cui: str, depth: int) -> Dict[str, Any]:
        """Build concept network for visualization"""
        return {"nodes": [], "links": []}
    
    async def _find_clinical_pathways(self, cui: str) -> List[Dict[str, Any]]:
        """Find clinical pathways for concept"""
        return []
    
    async def _find_treatment_options(self, cui: str) -> List[Dict[str, Any]]:
        """Find treatment options for concept"""
        return []
    
    async def _find_related_conditions(self, cui: str) -> List[Dict[str, Any]]:
        """Find related medical conditions"""
        return []
    
    async def _generate_differential_diagnosis(self, symptoms: List[str], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate differential diagnosis from symptoms"""
        return []
    
    async def _recommend_diagnostic_tests(self, symptoms: List[str], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Recommend diagnostic tests"""
        return []
    
    async def _find_treatment_pathways(self, symptoms: List[str], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find treatment pathways"""
        return []

# Global instance
unified_search_service = UnifiedMedicalKnowledgeSearch()
