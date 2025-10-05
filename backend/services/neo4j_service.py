"""
Neo4j Knowledge Graph Service for Medical Ontology
Provides graph database operations for medical concepts and relationships
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable, AuthError
import json
import os

from config.neo4j_config import Neo4jConfig, get_config
from services.umls_client import UMLSClient, UMLSConcept

# Enable Neo4j by default, can be disabled via environment variable
NEO4J_ENABLED = os.getenv('NEO4J_ENABLED', 'true').lower() == 'true'


logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    id: str
    labels: List[str]
    properties: Dict[str, Any]

@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph"""
    id: str
    type: str
    start_node_id: str
    end_node_id: str
    properties: Dict[str, Any]

@dataclass
class KnowledgeGraphResult:
    """Result of knowledge graph query"""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    metadata: Dict[str, Any]

class Neo4jService:
    """Service for managing medical knowledge graph in Neo4j"""

    def __init__(self, config: Neo4jConfig = None):
        """
        Initialize Neo4j service

        Args:
            config: Neo4j configuration object
        """
        self.config = config or get_config()
        self.driver: Optional[AsyncDriver] = None
        self.umls_client: Optional[UMLSClient] = None
        self.enabled = NEO4J_ENABLED
        
        if not self.enabled:
            logger.info("Neo4j is disabled. Knowledge graph features will use fallback methods.")
        else:
            logger.info("Initialized Neo4j service")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def connect(self):
        """Connect to Neo4j database"""
        if not self.enabled:
            logger.info("Neo4j is disabled, skipping connection")
            return
            
        try:
            config_dict = self.config.get_connection_config()
            self.driver = AsyncGraphDatabase.driver(
                config_dict["uri"],
                auth=config_dict["auth"],
                database=config_dict["database"]
            )

            # Verify connection
            await self.driver.verify_connectivity()
            logger.info("Connected to Neo4j database")

            # Initialize ontology clients
            await self._initialize_ontology_clients()

            # Set up database schema
            await self._setup_schema()

        except (ServiceUnavailable, AuthError) as e:
            logger.warning(f"Neo4j connection failed: {str(e)}. Knowledge graph features will be limited.")
            self.driver = None
            self.enabled = False

    async def disconnect(self):
        """Disconnect from Neo4j database"""
        if self.driver:
            await self.driver.close()
            logger.info("Disconnected from Neo4j database")

    async def initialize(self):
        """Initialize Neo4j service (alias for connect for compatibility)"""
        await self.connect()

    async def _initialize_ontology_clients(self):
        """Initialize UMLS and SNOMED CT clients"""
        umls_config = self.config.get_ontology_config("UMLS")
        if umls_config and umls_config.get("api_key"):
            self.umls_client = UMLSClient(
                api_key=umls_config["api_key"],
                version=umls_config.get("version", "current")
            )
            logger.info("Initialized UMLS client")

       

    async def _setup_schema(self):
        """Set up database schema with constraints and indexes"""
        async with self.driver.session() as session:
            # Create constraints
            for constraint in self.config.CONSTRAINTS:
                try:
                    await session.run(constraint)
                    logger.debug(f"Created constraint: {constraint}")
                except Exception as e:
                    logger.debug(f"Constraint may already exist: {str(e)}")

            # Create indexes
            for index in self.config.INDEXES:
                try:
                    await session.run(index)
                    logger.debug(f"Created index: {index}")
                except Exception as e:
                    logger.debug(f"Index may already exist: {str(e)}")

            logger.info("Database schema setup completed")

    async def add_medical_concept(self, concept_data: Dict[str, Any],
                                source: str = "manual") -> str:
        """
        Add a medical concept to the knowledge graph

        Args:
            concept_data: Concept information
            source: Source of the concept data

        Returns:
            Node ID of the created concept
        """
        if not self.enabled or not self.driver:
            logger.debug("Neo4j is disabled, skipping concept addition")
            return None
        query = """
        MERGE (c:MedicalConcept {cui: $cui})
        ON CREATE SET
            c.preferred_name = $preferred_name,
            c.synonyms = $synonyms,
            c.semantic_types = $semantic_types,
            c.definition = $definition,
            c.categories = $categories,
            c.related_terms = $related_terms,
            c.source = $source,
            c.created_at = datetime(),
            c.last_updated = datetime()
        ON MATCH SET
            c.last_updated = datetime(),
            c.synonyms = CASE
                WHEN size(c.synonyms) < size($synonyms) THEN $synonyms
                ELSE c.synonyms
            END,
            c.categories = CASE
                WHEN size(coalesce(c.categories, [])) < size($categories) THEN $categories
                ELSE coalesce(c.categories, [])
            END,
            c.related_terms = CASE
                WHEN size(coalesce(c.related_terms, [])) < size($related_terms) THEN $related_terms
                ELSE coalesce(c.related_terms, [])
            END
        RETURN id(c) as node_id, c.cui as cui
        """

        params = {
            "cui": concept_data.get("cui", ""),
            "preferred_name": concept_data.get("preferred_name", concept_data.get("name", "")),
            "synonyms": concept_data.get("synonyms", []),
            "semantic_types": concept_data.get("semantic_types", []),
            "definition": concept_data.get("definition", ""),
            "categories": concept_data.get("categories", []),
            "related_terms": concept_data.get("related_terms", []),
            "source": source
        }

        async with self.driver.session() as session:
            result = await session.run(query, params)
            record = await result.single()

            if record:
                logger.info(f"Added/updated concept: {record['cui']}")
                return str(record["node_id"])

            return None

    async def add_concept_relationship(self, cui1: str, cui2: str,
                                     relationship_type: str,
                                     properties: Dict[str, Any] = None) -> bool:
        """
        Add a relationship between two concepts

        Args:
            cui1: CUI of first concept
            cui2: CUI of second concept
            relationship_type: Type of relationship
            properties: Additional relationship properties

        Returns:
            True if relationship was created
        """
        query = f"""
        MATCH (c1:MedicalConcept {{cui: $cui1}})
        MATCH (c2:MedicalConcept {{cui: $cui2}})
        MERGE (c1)-[r:{relationship_type}]->(c2)
        ON CREATE SET r.created_at = datetime()
        """

        if properties:
            set_clauses = []
            for key, value in properties.items():
                if isinstance(value, str):
                    set_clauses.append(f"r.{key} = '{value}'")
                else:
                    set_clauses.append(f"r.{key} = {value}")
            if set_clauses:
                query += "ON CREATE SET " + ", ".join(set_clauses)

        query += " RETURN id(r) as rel_id"

        params = {
            "cui1": cui1,
            "cui2": cui2
        }

        async with self.driver.session() as session:
            result = await session.run(query, params)
            record = await result.single()

            success = record is not None
            if success:
                logger.info(f"Created relationship {relationship_type}: {cui1} -> {cui2}")

            return success

    async def search_concepts(self, query: str, limit: int = 10,
                            semantic_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for concepts in the knowledge graph

        Args:
            query: Search query
            limit: Maximum number of results
            semantic_types: Filter by semantic types

        Returns:
            List of matching concepts
        """
        if not self.enabled or not self.driver:
            logger.debug("Neo4j is disabled, returning empty search results")
            return []
        # Try fulltext search first, fall back to CONTAINS search
        try:
            # Check if fulltext index exists
            async with self.driver.session() as session:
                index_check = await session.run("""
                    SHOW INDEXES WHERE name = 'concept_fulltext_idx'
                    """)
                index_exists = await index_check.single() is not None

            if not index_exists:
                raise Exception("Fulltext index does not exist")

            base_query = """
            CALL db.index.fulltext.queryNodes("concept_fulltext_idx", $query)
            YIELD node, score
            """

            if semantic_types:
                base_query += """
                WHERE any(type IN $semantic_types WHERE type IN node.semantic_types)
                """

            base_query += """
            RETURN node.cui as cui,
                   node.preferred_name as preferred_name,
                   node.synonyms as synonyms,
                   node.semantic_types as semantic_types,
                   node.definition as definition,
                   score
            ORDER BY score DESC
            LIMIT $limit
            """

            params = {
                "query": query,
                "semantic_types": semantic_types or [],
                "limit": limit
            }

            async with self.driver.session() as session:
                result = await session.run(base_query, params)
                concepts = []

                async for record in result:
                    concepts.append({
                        "cui": record["cui"],
                        "preferred_name": record["preferred_name"],
                        "synonyms": record["synonyms"] or [],
                        "semantic_types": record["semantic_types"] or [],
                        "definition": record["definition"] or "",
                        "score": record["score"]
                    })

                logger.info(f"Found {len(concepts)} concepts for query: {query}")
                return concepts
                
        except Exception as e:
            logger.warning(f"Fulltext search failed, using fallback search: {e}")
            
            # Fallback to simple CONTAINS search
            fallback_query = """
            MATCH (c:MedicalConcept)
            WHERE toLower(c.preferred_name) CONTAINS toLower($query)
               OR any(synonym IN c.synonyms WHERE toLower(synonym) CONTAINS toLower($query))
               OR toLower(c.definition) CONTAINS toLower($query)
            """
            
            if semantic_types:
                fallback_query += """
                AND any(type IN $semantic_types WHERE type IN c.semantic_types)
                """
            
            fallback_query += """
            RETURN c.cui as cui,
                   c.preferred_name as preferred_name,
                   c.synonyms as synonyms,
                   c.semantic_types as semantic_types,
                   c.definition as definition,
                   1.0 as score
            ORDER BY c.preferred_name
            LIMIT $limit
            """
            
            params = {
                "query": query,
                "semantic_types": semantic_types or [],
                "limit": limit
            }

            async with self.driver.session() as session:
                result = await session.run(fallback_query, params)
                concepts = []

                async for record in result:
                    concepts.append({
                        "cui": record["cui"],
                        "preferred_name": record["preferred_name"],
                        "synonyms": record["synonyms"] or [],
                        "semantic_types": record["semantic_types"] or [],
                        "definition": record["definition"] or "",
                        "score": record["score"]
                    })

                logger.info(f"Found {len(concepts)} concepts for query: {query} (fallback search)")
                return concepts

    async def get_concept_details(self, cui: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a concept

        Args:
            cui: Concept Unique Identifier

        Returns:
            Concept details or None if not found
        """
        query = """
        MATCH (c:MedicalConcept {cui: $cui})
        OPTIONAL MATCH (c)-[r]-()
        RETURN c,
               count(r) as relationship_count,
               collect(distinct type(r)) as relationship_types
        """

        async with self.driver.session() as session:
            result = await session.run(query, {"cui": cui})
            record = await result.single()

            if record:
                node = record["c"]
                return {
                    "cui": node["cui"],
                    "preferred_name": node["preferred_name"],
                    "synonyms": node["synonyms"] or [],
                    "semantic_types": node["semantic_types"] or [],
                    "definition": node["definition"] or "",
                    "source": node["source"],
                    "relationship_count": record["relationship_count"],
                    "relationship_types": record["relationship_types"]
                }

            return None

    async def get_related_concepts(self, cui: str, relationship_type: str = None,
                                 max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Get concepts related to a given concept

        Args:
            cui: Concept Unique Identifier
            relationship_type: Filter by relationship type
            max_depth: Maximum traversal depth

        Returns:
            List of related concepts
        """
        if not self.enabled or not self.driver:
            logger.debug("Neo4j is disabled, returning empty related concepts")
            return []
        if relationship_type:
            query = f"""
            MATCH (c:MedicalConcept {{cui: $cui}})-[r:{relationship_type}*1..{max_depth}]-(related)
            RETURN related.cui as cui,
                   related.preferred_name as preferred_name,
                   related.semantic_types as semantic_types,
                   type(r[-1]) as relationship_type,
                   length(r) as depth
            ORDER BY depth, related.preferred_name
            LIMIT 50
            """
        else:
            query = f"""
            MATCH (c:MedicalConcept {{cui: $cui}})-[r*1..{max_depth}]-(related)
            RETURN related.cui as cui,
                   related.preferred_name as preferred_name,
                   related.semantic_types as semantic_types,
                   type(r[-1]) as relationship_type,
                   length(r) as depth
            ORDER BY depth, related.preferred_name
            LIMIT 50
            """

        async with self.driver.session() as session:
            result = await session.run(query, {"cui": cui})
            related_concepts = []

            async for record in result:
                related_concepts.append({
                    "cui": record["cui"],
                    "preferred_name": record["preferred_name"],
                    "semantic_types": record["semantic_types"] or [],
                    "relationship_type": record["relationship_type"],
                    "depth": record["depth"]
                })

            return related_concepts

    async def import_umls_concept(self, cui: str) -> bool:
        """
        Import a concept from UMLS into the knowledge graph

        Args:
            cui: UMLS Concept Unique Identifier

        Returns:
            True if import was successful
        """
        if not self.umls_client:
            logger.warning("UMLS client not available for import")
            return False

        try:
            async with self.umls_client as client:
                concept = await client.get_concept_details(cui)

                if not concept:
                    logger.warning(f"UMLS concept {cui} not found")
                    return False

                # Add concept to graph
                concept_data = {
                    "cui": concept.cui,
                    "preferred_name": concept.preferred_name,
                    "synonyms": concept.synonyms,
                    "semantic_types": concept.semantic_types,
                    "definition": concept.definitions[0] if concept.definitions else ""
                }

                node_id = await self.add_medical_concept(concept_data, "UMLS")

                # Add relationships
                for relation in concept.relations:
                    related_cui = relation.get("related_cui")
                    if related_cui:
                        rel_type = relation.get("relation", "RELATED_TO")
                        # Map UMLS relationship types to Neo4j relationship types
                        neo4j_rel_type = self._map_umls_relationship(rel_type)
                        await self.add_concept_relationship(
                            concept.cui,
                            related_cui,
                            neo4j_rel_type,
                            {"source": "UMLS"}
                        )

                logger.info(f"Successfully imported UMLS concept: {cui}")
                return True

        except Exception as e:
            logger.error(f"Failed to import UMLS concept {cui}: {str(e)}")
            return False


    def _map_umls_relationship(self, umls_rel: str) -> str:
        """Map UMLS relationship types to Neo4j relationship types"""
        mapping = {
            "isa": "IS_A",
            "part_of": "PART_OF",
            "has_part": "HAS_PART",
            "causes": "CAUSES",
            "associated_with": "ASSOCIATED_WITH",
            "treats": "TREATED_BY",
            "has_symptom": "HAS_SYMPTOM"
        }

        return mapping.get(umls_rel.lower(), "RELATED_TO")

    async def find_concept_synonyms(self, term: str) -> List[str]:
        """
        Find all synonyms for a given medical term

        Args:
            term: Medical term to search for

        Returns:
            List of synonyms
        """
        concepts = await self.search_concepts(term, limit=5)

        all_synonyms = []
        for concept in concepts:
            all_synonyms.extend(concept.get("synonyms", []))

        # Remove duplicates and original term
        synonyms = list(set(all_synonyms))
        if term in synonyms:
            synonyms.remove(term)

        logger.info(f"Found {len(synonyms)} synonyms for term: {term}")
        return synonyms

    async def get_concept_similarity(self, cui1: str, cui2: str) -> float:
        """
        Calculate semantic similarity between two concepts

        Args:
            cui1: First concept CUI
            cui2: Second concept CUI

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Get shortest path between concepts
        query = """
        MATCH (c1:MedicalConcept {cui: $cui1})
        MATCH (c2:MedicalConcept {cui: $cui2})
        MATCH path = shortestPath((c1)-[*]-(c2))
        RETURN length(path) as path_length
        """

        async with self.driver.session() as session:
            result = await session.run(query, {"cui1": cui1, "cui2": cui2})
            record = await result.single()

            if record and record["path_length"]:
                # Convert path length to similarity score
                # Shorter paths = higher similarity
                path_length = record["path_length"]
                similarity = max(0, 1.0 - (path_length * 0.1))
                return similarity

            return 0.0

    async def get_subgraph(self, cui: str, depth: int = 2) -> KnowledgeGraphResult:
        """
        Get subgraph around a concept

        Args:
            cui: Central concept CUI
            depth: Traversal depth

        Returns:
            Knowledge graph result with nodes and relationships
        """
        query = f"""
        MATCH (c:MedicalConcept {{cui: $cui}})-[r*0..{depth}]-(related)
        RETURN c, r, related
        LIMIT 100
        """

        async with self.driver.session() as session:
            result = await session.run(query, {"cui": cui})

            nodes = []
            relationships = []
            node_ids = set()

            async for record in result:
                # Process central node
                central_node = record["c"]
                if id(central_node) not in node_ids:
                    nodes.append(GraphNode(
                        id=str(id(central_node)),
                        labels=list(central_node.labels),
                        properties=dict(central_node)
                    ))
                    node_ids.add(id(central_node))

                # Process related node
                related_node = record["related"]
                if id(related_node) not in node_ids:
                    nodes.append(GraphNode(
                        id=str(id(related_node)),
                        labels=list(related_node.labels),
                        properties=dict(related_node)
                    ))
                    node_ids.add(id(related_node))

                # Process relationships
                rels = record["r"]
                for rel in rels:
                    relationships.append(GraphRelationship(
                        id=str(id(rel)),
                        type=rel.type,
                        start_node_id=str(id(rel.start_node)),
                        end_node_id=str(id(rel.end_node)),
                        properties=dict(rel)
                    ))

            return KnowledgeGraphResult(
                nodes=nodes,
                relationships=relationships,
                metadata={"central_concept": cui, "depth": depth}
            )

    async def batch_import_concepts(self, concepts: List[Dict[str, Any]],
                                  source: str = "batch") -> int:
        """
        Batch import multiple concepts

        Args:
            concepts: List of concept data dictionaries
            source: Source identifier

        Returns:
            Number of concepts imported
        """
        imported_count = 0

        for concept_data in concepts:
            try:
                node_id = await self.add_medical_concept(concept_data, source)
                if node_id:
                    imported_count += 1
            except Exception as e:
                logger.error(f"Failed to import concept {concept_data.get('cui', 'unknown')}: {str(e)}")

        logger.info(f"Batch imported {imported_count} concepts from {source}")
        return imported_count

    async def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query and return results
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of records as dictionaries
        """
        if not parameters:
            parameters = {}
            
        async with self.driver.session() as session:
            result = await session.run(query, parameters)
            records = []
            
            async for record in result:
                # Convert neo4j record to dictionary
                record_dict = {}
                for key in record.keys():
                    value = record[key]
                    # Handle different neo4j types
                    if hasattr(value, '__dict__'):
                        # Node or relationship object
                        if hasattr(value, 'labels'):
                            # Node
                            record_dict[key] = dict(value)
                        elif hasattr(value, 'type'):
                            # Relationship
                            record_dict[key] = dict(value)
                        else:
                            record_dict[key] = value
                    else:
                        record_dict[key] = value
                records.append(record_dict)
            
            return records
