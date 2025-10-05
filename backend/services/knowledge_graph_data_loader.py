"""
Knowledge Graph Data Loader for CortexMD
Handles comprehensive medical ontology data population for Neo4j
"""

import asyncio
import logging
import json
import csv
import os
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import aiofiles
from datetime import datetime
import hashlib

from services.neo4j_service import Neo4jService
from services.umls_client import UMLSClient
from config.neo4j_config import get_config

logger = logging.getLogger(__name__)

@dataclass
class DataLoadResult:
    """Result of data loading operation"""
    entities_loaded: int
    relationships_loaded: int
    errors: List[str]
    execution_time_seconds: float
    data_source: str

class KnowledgeGraphDataLoader:
    """Comprehensive data loader for medical knowledge graph"""

    def __init__(self, neo4j_service: Neo4jService = None):
        """
        Initialize data loader

        Args:
            neo4j_service: Neo4j service instance
        """
        self.neo4j_service = neo4j_service or Neo4jService()
        self.umls_client = UMLSClient()
        self.config = get_config()

        # Data directories
        self.data_dir = Path("backend/data")
        self.ontology_dir = self.data_dir / "ontologies"
        self.mappings_dir = self.data_dir / "mappings"
        self.cache_dir = self.data_dir / "cache"

        # Create directories
        for dir_path in [self.data_dir, self.ontology_dir, self.mappings_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Loading configuration
        self.batch_size = 1000
        self.max_workers = 4
        self.enable_caching = True

        logger.info("Initialized Knowledge Graph Data Loader")

    async def __aenter__(self):
        """Async context manager entry"""
        if hasattr(self.neo4j_service, '__aenter__'):
            await self.neo4j_service.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if hasattr(self.neo4j_service, '__aexit__'):
            await self.neo4j_service.__aexit__(exc_type, exc_val, exc_tb)

    async def load_comprehensive_medical_data(self) -> Dict[str, DataLoadResult]:
        """
        Load comprehensive medical data from all available sources

        Returns:
            Dictionary of data loading results by source
        """
        results = {}

        # Load data from different sources
        data_sources = [
            ("UMLS", self._load_umls_data),
            
            ("DrugBank", self._load_drugbank_data),
            ("Disease_Ontology", self._load_disease_ontology_data),
            ("Medical_Mappings", self._load_medical_mappings),
            ("Clinical_Relationships", self._load_clinical_relationships)
        ]

        for source_name, loader_func in data_sources:
            try:
                logger.info(f"Loading data from {source_name}...")
                start_time = datetime.now()

                result = await loader_func()

                execution_time = (datetime.now() - start_time).total_seconds()
                result.execution_time_seconds = execution_time

                results[source_name] = result
                logger.info(f"✅ {source_name}: {result.entities_loaded} entities, {result.relationships_loaded} relationships in {execution_time:.2f}s")

            except Exception as e:
                logger.error(f"❌ Failed to load {source_name}: {str(e)}")
                results[source_name] = DataLoadResult(
                    entities_loaded=0,
                    relationships_loaded=0,
                    errors=[str(e)],
                    execution_time_seconds=0.0,
                    data_source=source_name
                )

        return results

    async def _load_umls_data(self) -> DataLoadResult:
        """Load UMLS concepts and relationships"""
        entities_loaded = 0
        relationships_loaded = 0
        errors = []

        try:
            # Get UMLS concepts by semantic type
            semantic_types = [
                "Disease or Syndrome", "Sign or Symptom", "Clinical Drug",
                "Pharmacologic Substance", "Clinical Attribute", "Finding",
                "Body Part, Organ, or Organ Component", "Diagnostic Procedure",
                "Therapeutic or Preventive Procedure"
            ]

            for semantic_type in semantic_types:
                try:
                    concepts = await self.umls_client.search_by_semantic_type(semantic_type, limit=1000)

                    # Load concepts in batches
                    for i in range(0, len(concepts), self.batch_size):
                        batch = concepts[i:i + self.batch_size]
                        batch_result = await self._load_concept_batch(batch, "UMLS", semantic_type)
                        entities_loaded += batch_result["entities"]
                        relationships_loaded += batch_result["relationships"]

                except Exception as e:
                    errors.append(f"UMLS {semantic_type}: {str(e)}")

            # Load UMLS relationships
            relationships_result = await self._load_umls_relationships()
            relationships_loaded += relationships_result

        except Exception as e:
            errors.append(f"UMLS loading failed: {str(e)}")

        return DataLoadResult(
            entities_loaded=entities_loaded,
            relationships_loaded=relationships_loaded,
            errors=errors,
            execution_time_seconds=0.0,  # Will be set by caller
            data_source="UMLS"
        )

    async def _load_concept_batch(self, concepts: List[Dict], source: str,
                                semantic_type: str) -> Dict[str, int]:
        """Load a batch of concepts into Neo4j"""
        entities_created = 0
        relationships_created = 0

        async with self.neo4j_service.driver.session() as session:
            for concept in concepts:
                try:
                    # Create concept node
                    create_query = """
                    MERGE (c:MedicalConcept {cui: $cui})
                    SET c.preferred_name = $preferred_name,
                        c.semantic_types = $semantic_types,
                        c.definition = $definition,
                        c.source = $source,
                        c.last_updated = datetime()
                    """

                    await session.run(create_query, {
                        "cui": concept["cui"],
                        "preferred_name": concept["preferred_name"],
                        "semantic_types": concept.get("semantic_types", [semantic_type]),
                        "definition": concept.get("definition", ""),
                        "source": source
                    })

                    entities_created += 1

                    # Create synonym relationships
                    if concept.get("synonyms"):
                        for synonym in concept["synonyms"][:5]:  # Limit synonyms
                            synonym_query = """
                            MATCH (c:MedicalConcept {cui: $cui})
                            MERGE (s:Synonym {name: $synonym})
                            MERGE (c)-[:HAS_SYNONYM]->(s)
                            """

                            await session.run(synonym_query, {
                                "cui": concept["cui"],
                                "synonym": synonym
                            })

                    relationships_created += len(concept.get("synonyms", [])[:5])

                except Exception as e:
                    logger.debug(f"Failed to load concept {concept.get('cui', 'unknown')}: {str(e)}")

        return {"entities": entities_created, "relationships": relationships_created}

    async def _load_umls_relationships(self) -> int:
        """Load UMLS concept relationships"""
        relationships_loaded = 0

        # Define relationship types to load
        relationship_mappings = {
            "RB": "HAS_SYMPTOM",  # Disease has symptom
            "RN": "TREATS",       # Drug treats disease
            "RO": "CAUSES",       # Condition causes symptom
            "RQ": "ASSOCIATED_WITH",  # General association
        }

        async with self.neo4j_service.driver.session() as session:
            for rel_code, rel_type in relationship_mappings.items():
                try:
                    # Query UMLS for relationships of this type
                    relationships = await self.umls_client.get_relationships_by_type(rel_code, limit=2000)

                    for rel in relationships:
                        try:
                            rel_query = f"""
                            MATCH (c1:MedicalConcept {{cui: $cui1}})
                            MATCH (c2:MedicalConcept {{cui: $cui2}})
                            MERGE (c1)-[:{rel_type}]->(c2)
                            """

                            await session.run(rel_query, {
                                "cui1": rel["cui1"],
                                "cui2": rel["cui2"]
                            })

                            relationships_loaded += 1

                        except Exception as e:
                            logger.debug(f"Failed to create relationship: {str(e)}")

                except Exception as e:
                    logger.debug(f"Failed to load {rel_type} relationships: {str(e)}")

        return relationships_loaded

   

  
    async def _load_drugbank_data(self) -> DataLoadResult:
        """Load DrugBank drug interaction and target data"""
        entities_loaded = 0
        relationships_loaded = 0
        errors = []

        try:
            # Load drug-target relationships
            drug_targets = await self._load_drugbank_drug_targets()
            entities_loaded += drug_targets["entities"]
            relationships_loaded += drug_targets["relationships"]

            # Load drug-drug interactions
            drug_interactions = await self._load_drugbank_interactions()
            relationships_loaded += drug_interactions

        except Exception as e:
            errors.append(f"DrugBank loading failed: {str(e)}")

        return DataLoadResult(
            entities_loaded=entities_loaded,
            relationships_loaded=relationships_loaded,
            errors=errors,
            execution_time_seconds=0.0,
            data_source="DrugBank"
        )

    async def _load_drugbank_drug_targets(self) -> Dict[str, int]:
        """Load DrugBank drug-target relationships"""
        entities_loaded = 0
        relationships_loaded = 0

        # Mock DrugBank data loading (in real implementation, would connect to DrugBank API)
        drugbank_data = [
            {"drug_name": "Aspirin", "targets": ["COX-1", "COX-2"]},
            {"drug_name": "Warfarin", "targets": ["Vitamin K Epoxide Reductase"]},
            {"drug_name": "Metoprolol", "targets": ["Beta-1 Adrenergic Receptor"]},
        ]

        async with self.neo4j_service.driver.session() as session:
            for drug_data in drugbank_data:
                try:
                    # Create drug node if it doesn't exist
                    drug_query = """
                    MERGE (d:MedicalConcept {preferred_name: $drug_name})
                    SET d.semantic_types = CASE
                        WHEN NOT 'Clinical Drug' IN d.semantic_types
                        THEN d.semantic_types + 'Clinical Drug'
                        ELSE d.semantic_types
                        END,
                        d.source = 'DrugBank',
                        d.last_updated = datetime()
                    """

                    await session.run(drug_query, {"drug_name": drug_data["drug_name"]})
                    entities_loaded += 1

                    # Create target relationships
                    for target in drug_data["targets"]:
                        target_query = """
                        MATCH (d:MedicalConcept {preferred_name: $drug_name})
                        MERGE (t:Target {name: $target_name})
                        MERGE (d)-[:TARGETS]->(t)
                        """

                        await session.run(target_query, {
                            "drug_name": drug_data["drug_name"],
                            "target_name": target
                        })

                        relationships_loaded += 1

                except Exception as e:
                    logger.debug(f"Failed to load DrugBank drug-target: {str(e)}")

        return {"entities": entities_loaded, "relationships": relationships_loaded}

    async def _load_drugbank_interactions(self) -> int:
        """Load DrugBank drug-drug interactions"""
        relationships_loaded = 0

        # Mock drug interaction data
        interactions = [
            {
                "drug1": "Aspirin",
                "drug2": "Warfarin",
                "severity": "major",
                "description": "Increased risk of bleeding",
                "recommendations": ["Monitor INR closely", "Consider dose adjustment"]
            },
            {
                "drug1": "Metoprolol",
                "drug2": "Aspirin",
                "severity": "moderate",
                "description": "Potential additive cardiovascular effects",
                "recommendations": ["Monitor blood pressure", "Watch for bradycardia"]
            }
        ]

        async with self.neo4j_service.driver.session() as session:
            for interaction in interactions:
                try:
                    interaction_query = """
                    MATCH (d1:MedicalConcept {preferred_name: $drug1})
                    MATCH (d2:MedicalConcept {preferred_name: $drug2})
                    MERGE (d1)-[r:INTERACTS_WITH]->(d2)
                    SET r.severity = $severity,
                        r.description = $description,
                        r.recommendations = $recommendations,
                        r.evidence_level = 'clinical_study',
                        r.source = 'DrugBank'
                    """

                    await session.run(interaction_query, interaction)
                    relationships_loaded += 1

                except Exception as e:
                    logger.debug(f"Failed to load DrugBank interaction: {str(e)}")

        return relationships_loaded

    async def _load_disease_ontology_data(self) -> DataLoadResult:
        """Load Disease Ontology (DO) data"""
        entities_loaded = 0
        relationships_loaded = 0
        errors = []

        try:
            # Mock Disease Ontology data (in real implementation, would parse DO files)
            diseases = [
                {
                    "doid": "DOID:1234",
                    "name": "Myocardial Infarction",
                    "definition": "Necrosis of heart muscle",
                    "synonyms": ["Heart Attack", "MI"]
                },
                {
                    "doid": "DOID:5678",
                    "name": "Hypertension",
                    "definition": "Persistently elevated blood pressure",
                    "synonyms": ["High Blood Pressure", "HTN"]
                }
            ]

            async with self.neo4j_service.driver.session() as session:
                for disease in diseases:
                    try:
                        # Create disease concept
                        disease_query = """
                        MERGE (d:MedicalConcept {doid: $doid})
                        SET d.preferred_name = $name,
                            d.semantic_types = ['Disease or Syndrome'],
                            d.definition = $definition,
                            d.source = 'Disease_Ontology',
                            d.last_updated = datetime()
                        """

                        await session.run(disease_query, disease)
                        entities_loaded += 1

                        # Create synonym relationships
                        for synonym in disease["synonyms"]:
                            synonym_query = """
                            MATCH (d:MedicalConcept {doid: $doid})
                            MERGE (s:Synonym {name: $synonym})
                            MERGE (d)-[:HAS_SYNONYM]->(s)
                            """

                            await session.run(synonym_query, {
                                "doid": disease["doid"],
                                "synonym": synonym
                            })

                        relationships_loaded += len(disease["synonyms"])

                    except Exception as e:
                        logger.debug(f"Failed to load DO disease: {str(e)}")

        except Exception as e:
            errors.append(f"Disease Ontology loading failed: {str(e)}")

        return DataLoadResult(
            entities_loaded=entities_loaded,
            relationships_loaded=relationships_loaded,
            errors=errors,
            execution_time_seconds=0.0,
            data_source="Disease_Ontology"
        )

    async def _load_medical_mappings(self) -> DataLoadResult:
        """Load cross-references and mappings between different medical vocabularies"""
        entities_loaded = 0
        relationships_loaded = 0
        errors = []

        try:
            # Create mappings between different identifier systems
            mappings = [
                {"umls_cui": "C0004238"},
                {"umls_cui": "C0020538"},
            ]

            async with self.neo4j_service.driver.session() as session:
                for mapping in mappings:
                    try:
                        # Create cross-reference relationships
                        mapping_query = """
                        MATCH (c:MedicalConcept)
                        WHERE c.cui = $umls_cui OR c.sctid = $snomed_id
                        SET c.icd10_code = $icd10,
                            c.mapped_codes = CASE
                                WHEN c.mapped_codes IS NULL THEN [$icd10]
                                WHEN NOT $icd10 IN c.mapped_codes THEN c.mapped_codes + $icd10
                                ELSE c.mapped_codes
                            END
                        """

                        await session.run(mapping_query, mapping)
                        entities_loaded += 1

                    except Exception as e:
                        logger.debug(f"Failed to create medical mapping: {str(e)}")

        except Exception as e:
            errors.append(f"Medical mappings loading failed: {str(e)}")

        return DataLoadResult(
            entities_loaded=entities_loaded,
            relationships_loaded=relationships_loaded,
            errors=errors,
            execution_time_seconds=0.0,
            data_source="Medical_Mappings"
        )

    async def _load_clinical_relationships(self) -> DataLoadResult:
        """Load clinical relationships and comorbidity data"""
        entities_loaded = 0
        relationships_loaded = 0
        errors = []

        try:
            # Load comorbidity relationships
            comorbidities = [
                {
                    "condition1": "Hypertension",
                    "condition2": "Diabetes Mellitus",
                    "prevalence": 0.25,
                    "evidence_strength": "strong"
                },
                {
                    "condition1": "Myocardial Infarction",
                    "condition2": "Heart Failure",
                    "prevalence": 0.35,
                    "evidence_strength": "strong"
                }
            ]

            async with self.neo4j_service.driver.session() as session:
                for comorbidity in comorbidities:
                    try:
                        comorbidity_query = """
                        MATCH (c1:MedicalConcept {preferred_name: $condition1})
                        MATCH (c2:MedicalConcept {preferred_name: $condition2})
                        MERGE (c1)-[r:COMMONLY_COMORBID_WITH]-(c2)
                        SET r.prevalence = $prevalence,
                            r.evidence_strength = $evidence_strength
                        """

                        await session.run(comorbidity_query, comorbidity)
                        relationships_loaded += 1

                    except Exception as e:
                        logger.debug(f"Failed to create comorbidity relationship: {str(e)}")

            # Load risk factor relationships
            risk_factors = await self._load_risk_factor_relationships()
            relationships_loaded += risk_factors

        except Exception as e:
            errors.append(f"Clinical relationships loading failed: {str(e)}")

        return DataLoadResult(
            entities_loaded=entities_loaded,
            relationships_loaded=relationships_loaded,
            errors=errors,
            execution_time_seconds=0.0,
            data_source="Clinical_Relationships"
        )

    async def _load_risk_factor_relationships(self) -> int:
        """Load risk factor relationships"""
        relationships_loaded = 0

        risk_factors = [
            {"condition": "Myocardial Infarction", "risk_factor": "Smoking", "odds_ratio": 2.5},
            {"condition": "Hypertension", "risk_factor": "Obesity", "odds_ratio": 3.2},
            {"condition": "Diabetes Mellitus", "risk_factor": "Family History", "odds_ratio": 2.1},
        ]

        async with self.neo4j_service.driver.session() as session:
            for rf in risk_factors:
                try:
                    rf_query = """
                    MATCH (c:MedicalConcept {preferred_name: $condition})
                    MERGE (rf:RiskFactor {name: $risk_factor})
                    MERGE (rf)-[r:INCREASES_RISK_OF]->(c)
                    SET r.odds_ratio = $odds_ratio,
                        r.evidence_level = 'meta_analysis'
                    """

                    await session.run(rf_query, rf)
                    relationships_loaded += 1

                except Exception as e:
                    logger.debug(f"Failed to create risk factor relationship: {str(e)}")

        return relationships_loaded

    async def create_database_indexes(self) -> Dict[str, Any]:
        """Create optimized database indexes for performance"""
        indexes_created = []

        index_queries = [
            "CREATE INDEX concept_cui_idx IF NOT EXISTS FOR (c:MedicalConcept) ON (c.cui)",
            "CREATE INDEX concept_name_idx IF NOT EXISTS FOR (c:MedicalConcept) ON (c.preferred_name)",
            "CREATE INDEX concept_semantic_idx IF NOT EXISTS FOR (c:MedicalConcept) ON (c.semantic_types)",
            "CREATE INDEX concept_source_idx IF NOT EXISTS FOR (c:MedicalConcept) ON (c.source)",
            "CREATE INDEX relationship_type_idx IF NOT EXISTS FOR ()-[r]-() ON (type(r))",
            "CREATE INDEX synonym_name_idx IF NOT EXISTS FOR (s:Synonym) ON (s.name)",
            "CREATE INDEX target_name_idx IF NOT EXISTS FOR (t:Target) ON (t.name)",
            "CREATE INDEX risk_factor_name_idx IF NOT EXISTS FOR (rf:RiskFactor) ON (rf.name)",
        ]

        async with self.neo4j_service.driver.session() as session:
            for query in index_queries:
                try:
                    await session.run(query)
                    index_name = query.split()[1]  # Extract index name
                    indexes_created.append(index_name)
                    logger.info(f"Created index: {index_name}")
                except Exception as e:
                    logger.debug(f"Index may already exist: {str(e)}")

        return {
            "indexes_created": indexes_created,
            "total_indexes": len(indexes_created),
            "optimization_complete": True
        }

    async def validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality and integrity"""
        validation_results = {}

        async with self.neo4j_service.driver.session() as session:
            # Check total concepts
            concept_count_query = "MATCH (c:MedicalConcept) RETURN count(c) as total_concepts"
            result = await session.run(concept_count_query)
            record = await result.single()
            validation_results["total_concepts"] = record["total_concepts"]

            # Check concepts by source
            source_query = """
            MATCH (c:MedicalConcept)
            RETURN c.source as source, count(c) as count
            ORDER BY count DESC
            """
            result = await session.run(source_query)
            source_counts = {}
            async for record in result:
                source_counts[record["source"]] = record["count"]
            validation_results["concepts_by_source"] = source_counts

            # Check relationship counts
            rel_query = """
            MATCH ()-[r]-()
            RETURN type(r) as relationship_type, count(r) as count
            ORDER BY count DESC
            """
            result = await session.run(rel_query)
            rel_counts = {}
            async for record in result:
                rel_counts[record["relationship_type"]] = record["count"]
            validation_results["relationships_by_type"] = rel_counts

            # Check for orphaned nodes
            orphan_query = """
            MATCH (c:MedicalConcept)
            WHERE NOT (c)-[]->() AND NOT ()-[]->(c)
            RETURN count(c) as orphan_count
            """
            result = await session.run(orphan_query)
            record = await result.single()
            validation_results["orphaned_concepts"] = record["orphan_count"]

            # Data quality score
            total_concepts = validation_results["total_concepts"]
            orphaned_concepts = validation_results["orphaned_concepts"]
            quality_score = (total_concepts - orphaned_concepts) / total_concepts if total_concepts > 0 else 0
            validation_results["data_quality_score"] = round(quality_score * 100, 2)

        return validation_results

    async def export_knowledge_graph(self, output_path: str = None) -> str:
        """Export knowledge graph data for backup or analysis"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"knowledge_graph_export_{timestamp}.json"

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "metadata": {
                "exporter": "CortexMD Knowledge Graph Data Loader",
                "version": "1.0.0"
            },
            "concepts": [],
            "relationships": []
        }

        async with self.neo4j_service.driver.session() as session:
            # Export concepts
            concept_query = """
            MATCH (c:MedicalConcept)
            RETURN c.cui as cui,
                   c.preferred_name as name,
                   c.semantic_types as semantic_types,
                   c.definition as definition,
                   c.source as source
            """
            result = await session.run(concept_query)
            async for record in result:
                export_data["concepts"].append({
                    "cui": record["cui"],
                    "name": record["name"],
                    "semantic_types": record["semantic_types"],
                    "definition": record["definition"],
                    "source": record["source"]
                })

            # Export relationships
            rel_query = """
            MATCH (c1:MedicalConcept)-[r]-(c2:MedicalConcept)
            WHERE id(c1) < id(c2)
            RETURN c1.cui as cui1,
                   c2.cui as cui2,
                   type(r) as relationship_type,
                   properties(r) as properties
            """
            result = await session.run(rel_query)
            async for record in result:
                export_data["relationships"].append({
                    "cui1": record["cui1"],
                    "cui2": record["cui2"],
                    "relationship_type": record["relationship_type"],
                    "properties": record["properties"]
                })

        # Save to file
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(json.dumps(export_data, indent=2))

        logger.info(f"Knowledge graph exported to {output_path}")
        return output_path
