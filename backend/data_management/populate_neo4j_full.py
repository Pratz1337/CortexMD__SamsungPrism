#!/usr/bin/env python3
"""
Populate Neo4j with comprehensive medical data for FOL verification
"""

from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your Neo4j credentials
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

class PopulateNeo4j:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        logger.info("Connected to Neo4j")
    
    def clear_existing_data(self):
        """Optional: Clear existing data first"""
        with self.driver.session() as session:
            session.run("MATCH (n:MedicalConcept) DETACH DELETE n")
            logger.info("Cleared existing MedicalConcept nodes")
    
    def populate_medical_concepts(self):
        """Populate with MedicalConcept nodes that the system expects"""
        concepts = [
            # Diseases as MedicalConcepts
            """
            MERGE (c:MedicalConcept:Disease {
                cui: 'C1261473',
                preferred_name: 'soft tissue sarcoma',
                synonyms: ['sarcoma', 'soft tissue tumor', 'STS'],
                semantic_types: ['Neoplastic Process'],
                definition: 'Malignant neoplasm of soft tissue',
                categories: ['oncology', 'sarcoma'],
                source: 'UMLS'
            })
            """,
            
            """
            MERGE (c:MedicalConcept:Disease {
                cui: 'C0023827',
                preferred_name: 'liposarcoma',
                synonyms: ['adipose tissue sarcoma', 'fat cell sarcoma'],
                semantic_types: ['Neoplastic Process'],
                definition: 'Malignant tumor of adipose tissue',
                categories: ['oncology', 'sarcoma'],
                source: 'UMLS'
            })
            """,
            
            """
            MERGE (c:MedicalConcept {
                cui: 'C0577559',
                preferred_name: 'mass',
                synonyms: ['lump', 'tumor', 'swelling', 'growth'],
                semantic_types: ['Sign or Symptom'],
                definition: 'Palpable abnormal structure',
                categories: ['symptom', 'physical finding'],
                source: 'UMLS'
            })
            """,
            
            """
            MERGE (c:MedicalConcept {
                cui: 'C0030193',
                preferred_name: 'pain',
                synonyms: ['ache', 'discomfort', 'soreness'],
                semantic_types: ['Sign or Symptom'],
                definition: 'Unpleasant sensory experience',
                categories: ['symptom'],
                source: 'UMLS'
            })
            """,
            
            """
            MERGE (c:MedicalConcept {
                cui: 'C0027651',
                preferred_name: 'neoplasm',
                synonyms: ['tumor', 'neoplasia', 'growth', 'malignancy'],
                semantic_types: ['Neoplastic Process'],
                definition: 'Abnormal tissue growth',
                categories: ['oncology'],
                source: 'UMLS'
            })
            """,
            
            # Add more searchable terms
            """
            MERGE (c:MedicalConcept {
                cui: 'C0039866',
                preferred_name: 'thigh',
                synonyms: ['upper leg', 'femoral region'],
                semantic_types: ['Body Part, Organ, or Organ Component'],
                definition: 'Upper part of leg',
                categories: ['anatomy'],
                source: 'UMLS'
            })
            """
        ]
        
        with self.driver.session() as session:
            for query in concepts:
                session.run(query)
                logger.info("Added medical concept")
        
        logger.info("✅ Medical concepts populated")
    
    def create_fulltext_indexes(self):
        """Create fulltext indexes for better searching"""
        with self.driver.session() as session:
            try:
                # Drop existing fulltext index if it exists
                session.run("DROP INDEX concept_fulltext_idx IF EXISTS")
            except:
                pass
            
            # Create new fulltext index
            session.run("""
                CREATE FULLTEXT INDEX concept_fulltext_idx IF NOT EXISTS 
                FOR (n:MedicalConcept) 
                ON EACH [n.preferred_name, n.cui, n.definition]
            """)
            
            logger.info("✅ Fulltext indexes created")
    
    def create_relationships(self):
        """Create relationships between concepts"""
        relationships = [
            """
            MATCH (sarcoma:MedicalConcept {preferred_name: 'soft tissue sarcoma'})
            MATCH (mass:MedicalConcept {preferred_name: 'mass'})
            MERGE (sarcoma)-[:PRESENTS_WITH {frequency: 0.9}]->(mass)
            """,
            
            """
            MATCH (sarcoma:MedicalConcept {preferred_name: 'soft tissue sarcoma'})
            MATCH (pain:MedicalConcept {preferred_name: 'pain'})
            MERGE (sarcoma)-[:PRESENTS_WITH {frequency: 0.6}]->(pain)
            """,
            
            """
            MATCH (lipo:MedicalConcept {preferred_name: 'liposarcoma'})
            MATCH (sarcoma:MedicalConcept {preferred_name: 'soft tissue sarcoma'})
            MERGE (lipo)-[:IS_SUBTYPE_OF]->(sarcoma)
            """
        ]
        
        with self.driver.session() as session:
            for query in relationships:
                session.run(query)
            logger.info("✅ Relationships created")
    
    def verify_data(self):
        """Verify the data was loaded correctly"""
        with self.driver.session() as session:
            # Count MedicalConcepts
            result = session.run("MATCH (c:MedicalConcept) RETURN count(c) as count")
            count = result.single()['count']
            logger.info(f"Total MedicalConcepts: {count}")
            
            # Test search
            result = session.run("""
                MATCH (c:MedicalConcept)
                WHERE c.preferred_name CONTAINS 'sarcoma'
                RETURN c.preferred_name as name, c.cui as cui
            """)
            
            logger.info("Sarcoma concepts found:")
            for record in result:
                logger.info(f"  - {record['name']} (CUI: {record['cui']})")
    
    def close(self):
        self.driver.close()

def main():
    print("\n" + "="*60)
    print("   Populating Neo4j with Medical Data")
    print("="*60 + "\n")
    
    populator = PopulateNeo4j()
    
    try:
        # Clear and repopulate
        populator.clear_existing_data()
        populator.populate_medical_concepts()
        populator.create_fulltext_indexes()
        populator.create_relationships()
        populator.verify_data()
        
        print("\n✅ Neo4j populated successfully!")
        print("\nYour knowledge graph now contains medical concepts that will be found by searches.")
        
    finally:
        populator.close()

if __name__ == "__main__":
    main()
