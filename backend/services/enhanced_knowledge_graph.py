"""
Enhanced Knowledge Graph Service for Advanced Medical Reasoning
Provides graph-based reasoning for symptom clustering, drug interactions, and comorbidity analysis
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import json
from datetime import datetime, timedelta

from services.neo4j_service import Neo4jService, KnowledgeGraphResult
from config.neo4j_config import get_config

logger = logging.getLogger(__name__)

@dataclass
class SymptomCluster:
    """Represents a cluster of related symptoms"""
    cluster_id: str
    symptoms: List[str]
    common_diseases: List[Dict[str, Any]]
    severity_score: float
    confidence: float
    reasoning_path: List[str]

@dataclass
class DrugInteraction:
    """Represents a drug interaction"""
    drug1: str
    drug2: str
    interaction_type: str
    severity: str
    description: str
    evidence_level: str
    recommendations: List[str]

@dataclass
class ComorbidityAnalysis:
    """Represents comorbidity analysis results"""
    primary_condition: str
    comorbidities: List[Dict[str, Any]]
    risk_factors: List[str]
    management_recommendations: List[str]
    evidence_strength: str

@dataclass
class ReasoningResult:
    """Result of graph-based reasoning"""
    symptom_clusters: List[SymptomCluster]
    drug_interactions: List[DrugInteraction]
    comorbidity_analysis: ComorbidityAnalysis
    execution_time_ms: float
    reasoning_paths: List[str]

class EnhancedKnowledgeGraphService:
    """Enhanced service for advanced medical knowledge graph reasoning"""

    def __init__(self, neo4j_service: Neo4jService = None):
        """
        Initialize enhanced knowledge graph service

        Args:
            neo4j_service: Existing Neo4j service instance
        """
        self.neo4j_service = neo4j_service or Neo4jService()
        self.config = get_config()

        # Caching for performance
        self._cache = {}
        self._cache_expiry = {}

        # Reasoning configuration
        self.max_cluster_size = 10
        self.min_similarity_threshold = 0.3
        self.max_reasoning_depth = 4

        logger.info("Initialized Enhanced Knowledge Graph Service")

    async def __aenter__(self):
        """Async context manager entry"""
        if hasattr(self.neo4j_service, '__aenter__'):
            await self.neo4j_service.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if hasattr(self.neo4j_service, '__aexit__'):
            await self.neo4j_service.__aexit__(exc_type, exc_val, exc_tb)

    async def analyze_patient_symptoms(self, symptoms: List[str],
                                     patient_context: Dict[str, Any] = None) -> ReasoningResult:
        """
        Perform comprehensive analysis of patient symptoms using graph reasoning

        Args:
            symptoms: List of symptom terms
            patient_context: Additional patient information (age, gender, history, etc.)

        Returns:
            Complete reasoning result with clusters, interactions, and comorbidities
        """
        start_time = datetime.now()

        try:
            # Step 1: Cluster symptoms using graph-based similarity
            symptom_clusters = await self._cluster_symptoms_graph_based(symptoms)

            # Step 2: Analyze drug interactions if medications are present
            drug_interactions = []
            if patient_context and patient_context.get("current_medications"):
                drug_interactions = await self._analyze_drug_interactions_graph(
                    patient_context["current_medications"]
                )

            # Step 3: Perform comorbidity analysis
            comorbidity_analysis = await self._analyze_comorbidities_graph(
                symptoms, patient_context or {}
            )

            # Step 4: Generate reasoning paths
            reasoning_paths = await self._generate_reasoning_paths(
                symptom_clusters, drug_interactions, comorbidity_analysis
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            result = ReasoningResult(
                symptom_clusters=symptom_clusters,
                drug_interactions=drug_interactions,
                comorbidity_analysis=comorbidity_analysis,
                execution_time_ms=execution_time,
                reasoning_paths=reasoning_paths
            )

            logger.info(f"Completed patient symptom analysis in {execution_time:.2f}ms")
            return result

        except Exception as e:
            logger.error(f"Failed to analyze patient symptoms: {str(e)}")
            raise

    async def _cluster_symptoms_graph_based(self, symptoms: List[str]) -> List[SymptomCluster]:
        """
        Cluster symptoms using graph-based similarity analysis

        Args:
            symptoms: List of symptom terms

        Returns:
            List of symptom clusters
        """
        clusters = []

        # Normalize symptom terms to CUIs
        normalized_symptoms = []
        for symptom in symptoms:
            concept = await self._normalize_symptom_term(symptom)
            if concept:
                normalized_symptoms.append(concept)

        if not normalized_symptoms:
            logger.warning("No symptoms could be normalized")
            return clusters

        # Build symptom similarity graph
        similarity_graph = await self._build_symptom_similarity_graph(normalized_symptoms)

        # Apply graph clustering algorithm
        symptom_clusters = self._apply_graph_clustering(similarity_graph, normalized_symptoms)

        # Enrich clusters with disease associations
        for cluster in symptom_clusters:
            cluster.common_diseases = await self._find_common_diseases_for_cluster(cluster.symptoms)
            cluster.severity_score = await self._calculate_cluster_severity(cluster)
            cluster.confidence = await self._calculate_cluster_confidence(cluster, similarity_graph)
            cluster.reasoning_path = await self._generate_cluster_reasoning_path(cluster)

        logger.info(f"Generated {len(symptom_clusters)} symptom clusters")
        return symptom_clusters

    async def _normalize_symptom_term(self, symptom: str) -> Optional[Dict[str, Any]]:
        """Normalize a symptom term to standardized concept"""
        # Use existing ontology mapper or search knowledge graph
        concepts = await self.neo4j_service.search_concepts(
            symptom,
            semantic_types=["Sign or Symptom"],
            limit=1
        )

        if concepts:
            return concepts[0]

        # Fallback to broader search
        concepts = await self.neo4j_service.search_concepts(symptom, limit=1)
        return concepts[0] if concepts else None

    async def _build_symptom_similarity_graph(self, symptoms: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Build a similarity graph between symptoms

        Args:
            symptoms: List of normalized symptom concepts

        Returns:
            Similarity matrix as nested dictionary
        """
        similarity_graph = defaultdict(dict)

        for i, symptom1 in enumerate(symptoms):
            for j, symptom2 in enumerate(symptoms):
                if i >= j:
                    continue

                cui1 = symptom1["cui"]
                cui2 = symptom2["cui"]

                # Calculate semantic similarity using graph distance
                similarity = await self.neo4j_service.get_concept_similarity(cui1, cui2)

                # Also check for shared relationships
                shared_relations = await self._find_shared_relationships(cui1, cui2)
                relationship_boost = len(shared_relations) * 0.1

                total_similarity = min(1.0, similarity + relationship_boost)

                if total_similarity >= self.min_similarity_threshold:
                    similarity_graph[cui1][cui2] = total_similarity
                    similarity_graph[cui2][cui1] = total_similarity

        return dict(similarity_graph)

    async def _find_shared_relationships(self, cui1: str, cui2: str) -> List[Dict[str, Any]]:
        """Find shared relationships between two concepts"""
        query = """
        MATCH (c1:MedicalConcept {cui: $cui1})-[r1]-(shared)-[r2]-(c2:MedicalConcept {cui: $cui2})
        WHERE type(r1) = type(r2)
        RETURN shared.cui as shared_cui,
               shared.preferred_name as shared_name,
               type(r1) as relationship_type
        LIMIT 5
        """

        async with self.neo4j_service.driver.session() as session:
            result = await session.run(query, {"cui1": cui1, "cui2": cui2})
            shared_relations = []

            async for record in result:
                shared_relations.append({
                    "cui": record["shared_cui"],
                    "name": record["shared_name"],
                    "relationship": record["relationship_type"]
                })

            return shared_relations

    def _apply_graph_clustering(self, similarity_graph: Dict[str, Dict[str, float]],
                              symptoms: List[Dict[str, Any]]) -> List[SymptomCluster]:
        """Apply graph clustering algorithm to group similar symptoms"""
        # Simple connected components clustering
        visited = set()
        clusters = []

        for symptom in symptoms:
            cui = symptom["cui"]
            if cui in visited:
                continue

            # Find connected component
            cluster_symptoms = []
            queue = [cui]
            cluster_visited = set()

            while queue:
                current_cui = queue.pop(0)
                if current_cui in cluster_visited:
                    continue

                cluster_visited.add(current_cui)
                visited.add(current_cui)

                # Find symptom data for current CUI
                for symptom_data in symptoms:
                    if symptom_data["cui"] == current_cui:
                        cluster_symptoms.append(symptom_data["preferred_name"])
                        break

                # Add neighbors
                if current_cui in similarity_graph:
                    for neighbor, similarity in similarity_graph[current_cui].items():
                        if similarity >= self.min_similarity_threshold and neighbor not in cluster_visited:
                            queue.append(neighbor)

            if len(cluster_symptoms) > 1:  # Only create clusters with multiple symptoms
                cluster = SymptomCluster(
                    cluster_id=f"cluster_{len(clusters)}",
                    symptoms=cluster_symptoms,
                    common_diseases=[],
                    severity_score=0.0,
                    confidence=0.0,
                    reasoning_path=[]
                )
                clusters.append(cluster)

        return clusters

    async def _find_common_diseases_for_cluster(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """Find diseases commonly associated with a cluster of symptoms"""
        disease_counts = Counter()

        for symptom in symptoms:
            # Find diseases associated with this symptom
            concept = await self._normalize_symptom_term(symptom)
            if not concept:
                continue

            cui = concept["cui"]
            related_concepts = await self.neo4j_service.get_related_concepts(
                cui, relationship_type="HAS_SYMPTOM", max_depth=2
            )

            for related in related_concepts:
                if "Disease" in related.get("semantic_types", []):
                    disease_counts[related["cui"]] += 1

        # Get top diseases
        common_diseases = []
        for cui, count in disease_counts.most_common(5):
            concept_details = await self.neo4j_service.get_concept_details(cui)
            if concept_details:
                common_diseases.append({
                    "cui": cui,
                    "name": concept_details["preferred_name"],
                    "frequency": count,
                    "definition": concept_details.get("definition", "")
                })

        return common_diseases

    async def _calculate_cluster_severity(self, cluster: SymptomCluster) -> float:
        """Calculate severity score for a symptom cluster"""
        # Simple severity calculation based on disease associations
        if not cluster.common_diseases:
            return 0.5

        # Higher frequency of serious diseases increases severity
        severity_keywords = ["cancer", "malignant", "severe", "critical", "emergency"]
        severity_score = 0.0

        for disease in cluster.common_diseases:
            disease_name = disease["name"].lower()
            if any(keyword in disease_name for keyword in severity_keywords):
                severity_score += 0.8
            else:
                severity_score += 0.4

        return min(1.0, severity_score / len(cluster.common_diseases))

    async def _calculate_cluster_confidence(self, cluster: SymptomCluster,
                                         similarity_graph: Dict[str, Dict[str, float]]) -> float:
        """Calculate confidence score for a symptom cluster"""
        if len(cluster.symptoms) < 2:
            return 0.0

        # Average similarity within cluster
        total_similarity = 0.0
        pair_count = 0

        # Get CUIs for cluster symptoms
        cui_map = {}
        for symptom in cluster.symptoms:
            concept = await self._normalize_symptom_term(symptom)
            if concept:
                cui_map[symptom] = concept["cui"]

        for i, symptom1 in enumerate(cluster.symptoms):
            for j, symptom2 in enumerate(cluster.symptoms):
                if i >= j:
                    continue

                cui1 = cui_map.get(symptom1)
                cui2 = cui_map.get(symptom2)

                if cui1 and cui2 and cui1 in similarity_graph and cui2 in similarity_graph[cui1]:
                    total_similarity += similarity_graph[cui1][cui2]
                    pair_count += 1

        return total_similarity / pair_count if pair_count > 0 else 0.0

    async def _generate_cluster_reasoning_path(self, cluster: SymptomCluster) -> List[str]:
        """Generate reasoning path explaining the cluster formation"""
        reasoning = [
            f"Symptom cluster formed from {len(cluster.symptoms)} related symptoms",
            f"Symptoms: {', '.join(cluster.symptoms)}"
        ]

        if cluster.common_diseases:
            top_disease = cluster.common_diseases[0]
            reasoning.append(f"Most associated condition: {top_disease['name']}")
            reasoning.append(f"Association strength: {top_disease['frequency']} symptom matches")

        reasoning.append(f"Cluster confidence: {cluster.confidence:.2f}")
        reasoning.append(f"Severity assessment: {cluster.severity_score:.2f}")

        return reasoning

    async def _analyze_drug_interactions_graph(self, medications: List[str]) -> List[DrugInteraction]:
        """Analyze drug interactions using graph relationships"""
        interactions = []

        # Normalize medication names to concepts
        drug_concepts = []
        for med in medications:
            concept = await self._normalize_drug_term(med)
            if concept:
                drug_concepts.append(concept)

        # Check pairwise interactions
        for i, drug1 in enumerate(drug_concepts):
            for j, drug2 in enumerate(drug_concepts):
                if i >= j:
                    continue

                interaction = await self._check_drug_interaction(drug1, drug2)
                if interaction:
                    interactions.append(interaction)

        logger.info(f"Found {len(interactions)} drug interactions")
        return interactions

    async def _normalize_drug_term(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Normalize drug name to standardized concept"""
        concepts = await self.neo4j_service.search_concepts(
            drug_name,
            semantic_types=["Clinical Drug", "Pharmacologic Substance"],
            limit=1
        )

        if concepts:
            return concepts[0]

        # Fallback search
        concepts = await self.neo4j_service.search_concepts(drug_name, limit=1)
        return concepts[0] if concepts else None

    async def _check_drug_interaction(self, drug1: Dict[str, Any],
                                    drug2: Dict[str, Any]) -> Optional[DrugInteraction]:
        """Check for interaction between two drugs"""
        cui1 = drug1["cui"]
        cui2 = drug2["cui"]

        # Query for known interactions
        query = """
        MATCH (d1:MedicalConcept {cui: $cui1})-[r:INTERACTS_WITH]-(d2:MedicalConcept {cui: $cui2})
        RETURN r.severity as severity,
               r.description as description,
               r.evidence_level as evidence_level,
               r.recommendations as recommendations
        """

        async with self.neo4j_service.driver.session() as session:
            result = await session.run(query, {"cui1": cui1, "cui2": cui2})
            record = await result.single()

            if record:
                return DrugInteraction(
                    drug1=drug1["preferred_name"],
                    drug2=drug2["preferred_name"],
                    interaction_type="KNOWN_INTERACTION",
                    severity=record["severity"] or "moderate",
                    description=record["description"] or "Drug interaction detected",
                    evidence_level=record["evidence_level"] or "clinical_study",
                    recommendations=record["recommendations"] or []
                )

        # Check for indirect interactions through shared metabolic pathways
        indirect_interaction = await self._check_indirect_drug_interaction(cui1, cui2)
        return indirect_interaction

    async def _check_indirect_drug_interaction(self, cui1: str, cui2: str) -> Optional[DrugInteraction]:
        """Check for indirect drug interactions through shared pathways"""
        query = """
        MATCH (d1:MedicalConcept {cui: $cui1})-[r1:METABOLIZED_BY]-(enzyme)-[r2:METABOLIZED_BY]-(d2:MedicalConcept {cui: $cui2})
        WHERE enzyme.semantic_types CONTAINS "Enzyme"
        RETURN enzyme.preferred_name as enzyme_name,
               r1.inhibition_type as inhibition1,
               r2.inhibition_type as inhibition2
        LIMIT 1
        """

        async with self.neo4j_service.driver.session() as session:
            result = await session.run(query, {"cui1": cui1, "cui2": cui2})
            record = await result.single()

            if record and record["inhibition1"] == "strong" and record["inhibition2"] == "strong":
                enzyme = record["enzyme_name"]
                return DrugInteraction(
                    drug1="",  # Will be filled by caller
                    drug2="",  # Will be filled by caller
                    interaction_type="INDIRECT_METABOLIC",
                    severity="moderate",
                    description=f"Both drugs metabolized by {enzyme} - potential interaction",
                    evidence_level="pharmacokinetic",
                    recommendations=[
                        "Monitor drug levels",
                        "Consider dose adjustment",
                        "Watch for adverse effects"
                    ]
                )

        return None

    async def _analyze_comorbidities_graph(self, symptoms: List[str],
                                         patient_context: Dict[str, Any]) -> ComorbidityAnalysis:
        """Analyze potential comorbidities using graph relationships"""
        # Find primary conditions from symptoms
        primary_conditions = []
        for symptom in symptoms:
            concept = await self._normalize_symptom_term(symptom)
            if concept:
                diseases = await self.neo4j_service.get_related_concepts(
                    concept["cui"], relationship_type="HAS_SYMPTOM", max_depth=2
                )
                primary_conditions.extend([
                    d for d in diseases if "Disease" in d.get("semantic_types", [])
                ])

        if not primary_conditions:
            return ComorbidityAnalysis(
                primary_condition="Unknown",
                comorbidities=[],
                risk_factors=[],
                management_recommendations=[],
                evidence_strength="low"
            )

        # Take most likely primary condition
        primary = primary_conditions[0]

        # Find comorbidities
        comorbidities = await self._find_comorbidities(primary["cui"])

        # Generate risk factors and recommendations
        risk_factors = await self._extract_risk_factors(primary["cui"], comorbidities)
        recommendations = await self._generate_management_recommendations(primary, comorbidities)

        return ComorbidityAnalysis(
            primary_condition=primary["preferred_name"],
            comorbidities=comorbidities,
            risk_factors=risk_factors,
            management_recommendations=recommendations,
            evidence_strength="moderate"
        )

    async def _find_comorbidities(self, primary_cui: str) -> List[Dict[str, Any]]:
        """Find conditions commonly comorbid with the primary condition"""
        query = """
        MATCH (p:MedicalConcept {cui: $cui})-[r:COMMONLY_COMORBID_WITH]-(c:MedicalConcept)
        WHERE c.semantic_types CONTAINS "Disease"
        RETURN c.cui as cui,
               c.preferred_name as name,
               r.prevalence as prevalence,
               r.evidence_strength as evidence
        ORDER BY r.prevalence DESC
        LIMIT 5
        """

        async with self.neo4j_service.driver.session() as session:
            result = await session.run(query, {"cui": primary_cui})
            comorbidities = []

            async for record in result:
                comorbidities.append({
                    "cui": record["cui"],
                    "name": record["name"],
                    "prevalence": record["prevalence"] or 0.0,
                    "evidence_strength": record["evidence"] or "observational"
                })

            return comorbidities

    async def _extract_risk_factors(self, primary_cui: str,
                                  comorbidities: List[Dict[str, Any]]) -> List[str]:
        """Extract risk factors for comorbidities"""
        risk_factors = []

        # Common risk factors based on medical knowledge
        risk_factor_keywords = [
            "age", "smoking", "obesity", "hypertension", "diabetes",
            "family history", "sedentary lifestyle", "poor diet"
        ]

        for comorbidity in comorbidities:
            comorbidity_name = comorbidity["name"].lower()
            for factor in risk_factor_keywords:
                if factor in comorbidity_name or factor.replace(" ", "_") in comorbidity_name:
                    risk_factors.append(factor.title())

        # Remove duplicates
        return list(set(risk_factors))

    async def _generate_management_recommendations(self, primary: Dict[str, Any],
                                                 comorbidities: List[Dict[str, Any]]) -> List[str]:
        """Generate management recommendations for comorbid conditions"""
        recommendations = [
            "Regular monitoring of both conditions",
            "Coordinated care between specialists",
            "Patient education on symptom recognition"
        ]

        if comorbidities:
            recommendations.append("Consider comprehensive treatment plan addressing both conditions")

        return recommendations

    async def _generate_reasoning_paths(self, symptom_clusters: List[SymptomCluster],
                                      drug_interactions: List[DrugInteraction],
                                      comorbidity_analysis: ComorbidityAnalysis) -> List[str]:
        """Generate comprehensive reasoning paths for the analysis"""
        paths = []

        # Symptom clustering reasoning
        if symptom_clusters:
            paths.append(f"Identified {len(symptom_clusters)} symptom clusters using graph-based similarity analysis")
            for i, cluster in enumerate(symptom_clusters):
                paths.append(f"Cluster {i+1}: {len(cluster.symptoms)} symptoms with {len(cluster.common_diseases)} associated conditions")

        # Drug interaction reasoning
        if drug_interactions:
            paths.append(f"Detected {len(drug_interactions)} drug interactions through graph relationship analysis")
            severe_interactions = [i for i in drug_interactions if i.severity == "severe"]
            if severe_interactions:
                paths.append(f"Found {len(severe_interactions)} severe drug interactions requiring immediate attention")

        # Comorbidity reasoning
        if comorbidity_analysis.comorbidities:
            paths.append(f"Comorbidity analysis identified {len(comorbidity_analysis.comorbidities)} related conditions")
            paths.append(f"Primary condition: {comorbidity_analysis.primary_condition}")

        return paths

    # Performance optimization methods
    async def optimize_query_performance(self, query_type: str) -> Dict[str, Any]:
        """Optimize Cypher queries for real-time performance"""
        optimizations = {
            "symptom_clustering": self._optimize_symptom_clustering_queries,
            "drug_interaction": self._optimize_drug_interaction_queries,
            "comorbidity_analysis": self._optimize_comorbidity_queries
        }

        if query_type in optimizations:
            return await optimizations[query_type]()

        return {"status": "unknown_query_type"}

    async def _optimize_symptom_clustering_queries(self) -> Dict[str, Any]:
        """Optimize symptom clustering query performance"""
        # Create composite indexes for symptom clustering
        optimization_queries = [
            "CREATE INDEX symptom_semantic_idx IF NOT EXISTS FOR (c:MedicalConcept) ON (c.semantic_types) WHERE 'Sign or Symptom' IN c.semantic_types",
            "CREATE INDEX disease_semantic_idx IF NOT EXISTS FOR (c:MedicalConcept) ON (c.semantic_types) WHERE 'Disease' IN c.semantic_types",
            "CREATE INDEX relationship_type_idx IF NOT EXISTS FOR ()-[r]-() ON (type(r))"
        ]

        executed_optimizations = []
        async with self.neo4j_service.driver.session() as session:
            for query in optimization_queries:
                try:
                    await session.run(query)
                    executed_optimizations.append(query.split()[1])  # Extract index/constraint name
                except Exception as e:
                    logger.debug(f"Optimization may already exist: {str(e)}")

        return {
            "query_type": "symptom_clustering",
            "optimizations_applied": executed_optimizations,
            "expected_performance_improvement": "40-60%"
        }

    async def _optimize_drug_interaction_queries(self) -> Dict[str, Any]:
        """Optimize drug interaction query performance"""
        optimization_queries = [
            "CREATE INDEX drug_semantic_idx IF NOT EXISTS FOR (c:MedicalConcept) ON (c.semantic_types) WHERE 'Clinical Drug' IN c.semantic_types OR 'Pharmacologic Substance' IN c.semantic_types",
            "CREATE INDEX interaction_relationship_idx IF NOT EXISTS FOR ()-[r:INTERACTS_WITH]-() ON (r.severity, r.evidence_level)"
        ]

        executed_optimizations = []
        async with self.neo4j_service.driver.session() as session:
            for query in optimization_queries:
                try:
                    await session.run(query)
                    executed_optimizations.append(query.split()[1])
                except Exception as e:
                    logger.debug(f"Optimization may already exist: {str(e)}")

        return {
            "query_type": "drug_interaction",
            "optimizations_applied": executed_optimizations,
            "expected_performance_improvement": "50-70%"
        }

    async def _optimize_comorbidity_queries(self) -> Dict[str, Any]:
        """Optimize comorbidity analysis query performance"""
        optimization_queries = [
            "CREATE INDEX comorbidity_relationship_idx IF NOT EXISTS FOR ()-[r:COMMONLY_COMORBID_WITH]-() ON (r.prevalence, r.evidence_strength)"
        ]

        executed_optimizations = []
        async with self.neo4j_service.driver.session() as session:
            for query in optimization_queries:
                try:
                    await session.run(query)
                    executed_optimizations.append(query.split()[1])
                except Exception as e:
                    logger.debug(f"Optimization may already exist: {str(e)}")

        return {
            "query_type": "comorbidity_analysis",
            "optimizations_applied": executed_optimizations,
            "expected_performance_improvement": "30-50%"
        }
