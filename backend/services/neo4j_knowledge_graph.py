"""
Neo4j Knowledge Graph Service for Enhanced FOL Verification
Provides medical knowledge graph capabilities for better FOL reasoning
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeNode:
    """Represents a node in the medical knowledge graph"""
    node_type: str  # Disease, Symptom, Treatment, Medication, etc.
    name: str
    properties: Dict[str, Any]
    relationships: List[Dict[str, Any]]

@dataclass
class FOLGraphRelation:
    """Represents a FOL relation in the graph"""
    subject: str
    predicate: str
    object: str
    confidence: float
    evidence: List[str]
    source: str  # UMLS, SNOMED, ICD10, etc.

class Neo4jKnowledgeGraph:
    """
    Enhanced Neo4j service for medical knowledge graphs
    Provides FOL reasoning capabilities using graph traversal
    """
    
    def __init__(self):
        """Initialize Neo4j connection with proper configuration"""
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'neo4j_cortex_2024')
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')
        self.enabled = os.getenv('NEO4J_ENABLED', 'true').lower() == 'true'
        
        self.driver = None
        self.async_driver = None
        
        if self.enabled:
            try:
                # Initialize synchronous driver
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=50,
                    fetch_size=1000
                )
                
                # Initialize async driver for async operations
                self.async_driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password)
                )
                
                # Test connection
                self._test_connection()
                logger.info(f"✅ Neo4j connected successfully at {self.uri}")
                
                # Initialize schema if needed
                self._initialize_schema()
                
            except ServiceUnavailable:
                logger.error(f"❌ Neo4j service unavailable at {self.uri}")
                self.enabled = False
            except AuthError:
                logger.error(f"❌ Neo4j authentication failed")
                self.enabled = False
            except Exception as e:
                logger.error(f"❌ Neo4j connection failed: {e}")
                self.enabled = False
        else:
            logger.info("Neo4j is disabled. Knowledge graph features will use fallback methods.")
    
    def _test_connection(self):
        """Test Neo4j connection"""
        if not self.driver:
            return False
            
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                return result.single()['test'] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def _initialize_schema(self):
        """Initialize Neo4j schema with medical knowledge graph constraints"""
        if not self.driver:
            return
            
        try:
            with self.driver.session(database=self.database) as session:
                # Create constraints for unique names
                constraints = [
                    "CREATE CONSTRAINT unique_disease IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
                    "CREATE CONSTRAINT unique_symptom IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
                    "CREATE CONSTRAINT unique_medication IF NOT EXISTS FOR (m:Medication) REQUIRE m.name IS UNIQUE",
                    "CREATE CONSTRAINT unique_treatment IF NOT EXISTS FOR (t:Treatment) REQUIRE t.name IS UNIQUE",
                    "CREATE CONSTRAINT unique_anatomy IF NOT EXISTS FOR (a:Anatomy) REQUIRE a.name IS UNIQUE",
                    "CREATE CONSTRAINT unique_lab_test IF NOT EXISTS FOR (l:LabTest) REQUIRE l.name IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        logger.debug(f"Constraint may already exist: {e}")
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX disease_icd10 IF NOT EXISTS FOR (d:Disease) ON (d.icd10_code)",
                    "CREATE INDEX disease_umls IF NOT EXISTS FOR (d:Disease) ON (d.umls_cui)",
                    "CREATE INDEX symptom_umls IF NOT EXISTS FOR (s:Symptom) ON (s.umls_cui)",
                    "CREATE INDEX medication_rxnorm IF NOT EXISTS FOR (m:Medication) ON (m.rxnorm_code)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.debug(f"Index may already exist: {e}")
                
                logger.info("✅ Neo4j schema initialized")
                
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
    
    async def add_medical_knowledge(self, knowledge_type: str, data: Dict[str, Any]) -> bool:
        """
        Add medical knowledge to the graph
        
        Args:
            knowledge_type: Type of knowledge (disease, symptom, treatment, etc.)
            data: Knowledge data including name, properties, and relationships
        """
        if not self.enabled or not self.async_driver:
            return False
        
        try:
            async with self.async_driver.session(database=self.database) as session:
                # Create node based on type
                if knowledge_type == "disease":
                    await session.run("""
                        MERGE (d:Disease {name: $name})
                        SET d += $properties
                    """, name=data['name'], properties=data.get('properties', {}))
                    
                    # Add relationships
                    for rel in data.get('relationships', []):
                        if rel['type'] == 'PRESENTS_WITH':
                            await session.run("""
                                MATCH (d:Disease {name: $disease})
                                MERGE (s:Symptom {name: $symptom})
                                MERGE (d)-[:PRESENTS_WITH {frequency: $frequency}]->(s)
                            """, disease=data['name'], symptom=rel['target'], 
                                frequency=rel.get('frequency', 0.5))
                        
                        elif rel['type'] == 'TREATED_WITH':
                            await session.run("""
                                MATCH (d:Disease {name: $disease})
                                MERGE (t:Treatment {name: $treatment})
                                MERGE (d)-[:TREATED_WITH {efficacy: $efficacy}]->(t)
                            """, disease=data['name'], treatment=rel['target'],
                                efficacy=rel.get('efficacy', 0.5))
                
                logger.info(f"✅ Added {knowledge_type}: {data['name']}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add medical knowledge: {e}")
            return False
    
    async def query_fol_predicate(self, predicate_type: str, subject: str, object: str) -> Dict[str, Any]:
        """
        Query the knowledge graph for FOL predicate verification
        
        Args:
            predicate_type: Type of predicate (has_symptom, has_condition, etc.)
            subject: Subject of predicate (usually 'patient')
            object: Object of predicate (symptom name, condition name, etc.)
        
        Returns:
            Verification result with confidence and evidence
        """
        if not self.enabled or not self.async_driver:
            return {
                'verified': False,
                'confidence': 0.0,
                'evidence': ['Neo4j not available'],
                'graph_support': False
            }
        
        try:
            async with self.async_driver.session(database=self.database) as session:
                
                if predicate_type == 'has_symptom':
                    # Check if symptom exists and get related diseases
                    result = await session.run("""
                        MATCH (s:Symptom {name: $symptom})
                        OPTIONAL MATCH (d:Disease)-[r:PRESENTS_WITH]->(s)
                        RETURN s, collect({
                            disease: d.name,
                            frequency: r.frequency
                        }) as diseases
                    """, symptom=object)
                    
                    record = await result.single()
                    if record and record['s']:
                        return {
                            'verified': True,
                            'confidence': 0.8,
                            'evidence': [f"Symptom '{object}' found in knowledge graph"],
                            'related_diseases': record['diseases'],
                            'graph_support': True
                        }
                
                elif predicate_type == 'has_condition':
                    # Check disease and get symptoms, treatments
                    result = await session.run("""
                        MATCH (d:Disease {name: $disease})
                        OPTIONAL MATCH (d)-[ps:PRESENTS_WITH]->(s:Symptom)
                        OPTIONAL MATCH (d)-[tw:TREATED_WITH]->(t:Treatment)
                        RETURN d,
                               collect(DISTINCT {
                                   symptom: s.name,
                                   frequency: ps.frequency
                               }) as symptoms,
                               collect(DISTINCT {
                                   treatment: t.name,
                                   efficacy: tw.efficacy
                               }) as treatments
                    """, disease=object)
                    
                    record = await result.single()
                    if record and record['d']:
                        return {
                            'verified': True,
                            'confidence': 0.9,
                            'evidence': [f"Condition '{object}' found in knowledge graph"],
                            'expected_symptoms': record['symptoms'],
                            'recommended_treatments': record['treatments'],
                            'graph_support': True
                        }
                
                elif predicate_type == 'takes_medication':
                    # Check medication and related conditions
                    result = await session.run("""
                        MATCH (m:Medication {name: $medication})
                        OPTIONAL MATCH (d:Disease)-[:TREATED_WITH]->(t:Treatment)-[:USES]->(m)
                        RETURN m, collect(DISTINCT d.name) as treats_conditions
                    """, medication=object)
                    
                    record = await result.single()
                    if record and record['m']:
                        return {
                            'verified': True,
                            'confidence': 0.7,
                            'evidence': [f"Medication '{object}' found in knowledge graph"],
                            'treats_conditions': record['treats_conditions'],
                            'graph_support': True
                        }
                
                # Default: not found in graph
                return {
                    'verified': False,
                    'confidence': 0.0,
                    'evidence': [f"'{object}' not found in knowledge graph for {predicate_type}"],
                    'graph_support': True
                }
                
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return {
                'verified': False,
                'confidence': 0.0,
                'evidence': [f'Graph query error: {str(e)}'],
                'graph_support': False
            }
    
    async def get_disease_hierarchy(self, disease_name: str) -> Dict[str, Any]:
        """
        Get disease hierarchy and relationships from the graph
        """
        if not self.enabled or not self.async_driver:
            return {}
        
        try:
            async with self.async_driver.session(database=self.database) as session:
                result = await session.run("""
                    MATCH (d:Disease {name: $disease})
                    OPTIONAL MATCH (d)-[:IS_SUBTYPE_OF*]->(parent:Disease)
                    OPTIONAL MATCH (child:Disease)-[:IS_SUBTYPE_OF]->(d)
                    OPTIONAL MATCH (d)-[:SIMILAR_TO]-(similar:Disease)
                    RETURN d,
                           collect(DISTINCT parent.name) as parent_diseases,
                           collect(DISTINCT child.name) as subtypes,
                           collect(DISTINCT similar.name) as similar_diseases
                """, disease=disease_name)
                
                record = await result.single()
                if record:
                    return {
                        'disease': disease_name,
                        'parent_diseases': record['parent_diseases'],
                        'subtypes': record['subtypes'],
                        'similar_diseases': record['similar_diseases']
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get disease hierarchy: {e}")
        
        return {}
    
    async def infer_fol_relations(self, symptoms: List[str], conditions: List[str]) -> List[FOLGraphRelation]:
        """
        Infer FOL relations using graph traversal and pattern matching
        """
        if not self.enabled or not self.async_driver:
            return []
        
        relations = []
        
        try:
            async with self.async_driver.session(database=self.database) as session:
                # Find diseases that match the symptom pattern
                if symptoms:
                    result = await session.run("""
                        UNWIND $symptoms as symptom
                        MATCH (s:Symptom {name: symptom})
                        MATCH (d:Disease)-[r:PRESENTS_WITH]->(s)
                        WITH d, avg(r.frequency) as avg_freq, 
                             collect(s.name) as matched_symptoms
                        WHERE size(matched_symptoms) >= $min_matches
                        RETURN d.name as disease, avg_freq as confidence,
                               matched_symptoms
                        ORDER BY confidence DESC
                        LIMIT 5
                    """, symptoms=symptoms, min_matches=max(1, len(symptoms) // 2))
                    
                    async for record in result:
                        relations.append(FOLGraphRelation(
                            subject='patient',
                            predicate='likely_has_condition',
                            object=record['disease'],
                            confidence=record['confidence'],
                            evidence=record['matched_symptoms'],
                            source='Neo4j_inference'
                        ))
                
                # Find expected treatments for conditions
                if conditions:
                    result = await session.run("""
                        UNWIND $conditions as condition
                        MATCH (d:Disease {name: condition})
                        MATCH (d)-[r:TREATED_WITH]->(t:Treatment)
                        WHERE r.first_line = true OR r.efficacy > 0.7
                        RETURN d.name as disease, t.name as treatment,
                               r.efficacy as efficacy
                        ORDER BY efficacy DESC
                    """, conditions=conditions)
                    
                    async for record in result:
                        relations.append(FOLGraphRelation(
                            subject='patient',
                            predicate='should_receive_treatment',
                            object=record['treatment'],
                            confidence=record['efficacy'],
                            evidence=[f"Recommended for {record['disease']}"],
                            source='Neo4j_treatment_inference'
                        ))
                
        except Exception as e:
            logger.error(f"Failed to infer FOL relations: {e}")
        
        return relations
    
    async def expand_medical_concept(self, concept: str) -> Dict[str, Any]:
        """
        Expand a medical concept using the knowledge graph
        """
        if not self.enabled or not self.async_driver:
            return {'concept': concept, 'expanded': False}
        
        try:
            async with self.async_driver.session(database=self.database) as session:
                # Try to find the concept as disease, symptom, or treatment
                result = await session.run("""
                    OPTIONAL MATCH (d:Disease {name: $concept})
                    OPTIONAL MATCH (s:Symptom {name: $concept})
                    OPTIONAL MATCH (t:Treatment {name: $concept})
                    OPTIONAL MATCH (m:Medication {name: $concept})
                    
                    WITH coalesce(d, s, t, m) as node
                    WHERE node IS NOT NULL
                    
                    OPTIONAL MATCH (node)-[r]-(related)
                    RETURN node,
                           labels(node) as node_types,
                           properties(node) as properties,
                           collect({
                               type: type(r),
                               direction: CASE 
                                   WHEN startNode(r) = node THEN 'OUTGOING'
                                   ELSE 'INCOMING' 
                               END,
                               related_node: properties(related),
                               related_type: labels(related)[0],
                               relationship_props: properties(r)
                           }) as relationships
                """, concept=concept)
                
                record = await result.single()
                if record and record['node']:
                    return {
                        'concept': concept,
                        'expanded': True,
                        'type': record['node_types'][0] if record['node_types'] else 'Unknown',
                        'properties': record['properties'],
                        'relationships': record['relationships']
                    }
                    
        except Exception as e:
            logger.error(f"Failed to expand concept: {e}")
        
        return {'concept': concept, 'expanded': False}
    
    def close(self):
        """Close Neo4j connections"""
        if self.driver:
            self.driver.close()
        if self.async_driver:
            asyncio.create_task(self.async_driver.close())
        logger.info("Neo4j connections closed")


# Global instance
neo4j_kg = Neo4jKnowledgeGraph()


async def enhance_fol_with_knowledge_graph(
    predicates: List[str],
    patient_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enhance FOL verification using Neo4j knowledge graph
    
    Args:
        predicates: List of FOL predicates to verify
        patient_data: Patient data for verification
    
    Returns:
        Enhanced verification results with graph-based evidence
    """
    if not neo4j_kg.enabled:
        return {
            'enhanced': False,
            'message': 'Neo4j knowledge graph not available'
        }
    
    enhanced_results = []
    
    for predicate in predicates:
        # Parse predicate
        import re
        match = re.match(r'(\w+)\(([\w\s]+),\s*([\w\s_]+)(?:,\s*([\w\s]+))?\)', predicate)
        
        if match:
            pred_type = match.group(1)
            subject = match.group(2)
            obj = match.group(3)
            
            # Query knowledge graph
            graph_result = await neo4j_kg.query_fol_predicate(pred_type, subject, obj)
            
            enhanced_results.append({
                'predicate': predicate,
                'graph_verification': graph_result,
                'enhanced': True
            })
        else:
            enhanced_results.append({
                'predicate': predicate,
                'graph_verification': None,
                'enhanced': False,
                'error': 'Could not parse predicate'
            })
    
    # Infer additional relations
    symptoms = patient_data.get('symptoms', [])
    conditions = patient_data.get('diagnoses', [])
    inferred_relations = await neo4j_kg.infer_fol_relations(symptoms, conditions)
    
    return {
        'enhanced': True,
        'results': enhanced_results,
        'inferred_relations': [
            {
                'predicate': f"{rel.predicate}({rel.subject}, {rel.object})",
                'confidence': rel.confidence,
                'evidence': rel.evidence,
                'source': rel.source
            }
            for rel in inferred_relations
        ],
        'graph_available': neo4j_kg.enabled
    }
