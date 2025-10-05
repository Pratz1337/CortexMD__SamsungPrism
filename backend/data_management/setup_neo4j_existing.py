#!/usr/bin/env python3
"""
Neo4j Setup Script for Existing Neo4j Desktop Installation
Sets up the cortexMD database with medical knowledge graph
"""

import os
import sys
from neo4j import GraphDatabase
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Neo4j connection details from your setup
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
NEO4J_DATABASE = "neo4j"

class Neo4jSetup:
    def __init__(self):
        """Initialize Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            logger.info("✅ Connected to Neo4j Desktop (cortexMD)")
            self.verify_connection()
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            logger.info("Please ensure Neo4j Desktop is running and the cortexMD database is started")
            sys.exit(1)
    
    def verify_connection(self):
        """Verify Neo4j connection and check for GDS plugin"""
        with self.driver.session() as session:
            # Check connection
            result = session.run("RETURN 1 as test")
            if result.single()['test'] == 1:
                logger.info("✅ Neo4j connection verified")
            
            # Check for GDS (Graph Data Science) plugin
            try:
                result = session.run("RETURN gds.version() as version")
                version = result.single()['version']
                logger.info(f"✅ Graph Data Science plugin installed (version: {version})")
            except:
                logger.warning("⚠️ Graph Data Science plugin not detected")
                logger.info("To install GDS plugin:")
                logger.info("1. Copy backend/neo4j_plugins/graph-data-science.jar to Neo4j plugins folder")
                logger.info("2. Restart Neo4j Desktop")
    
    def create_constraints_and_indexes(self):
        """Create constraints and indexes for optimal performance"""
        logger.info("📊 Creating constraints and indexes...")
        
        constraints = [
            # Unique constraints
            "CREATE CONSTRAINT unique_disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT unique_symptom_name IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT unique_medication_name IF NOT EXISTS FOR (m:Medication) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT unique_treatment_name IF NOT EXISTS FOR (t:Treatment) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT unique_anatomy_name IF NOT EXISTS FOR (a:Anatomy) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT unique_lab_test_name IF NOT EXISTS FOR (l:LabTest) REQUIRE l.name IS UNIQUE",
            "CREATE CONSTRAINT unique_medical_concept_cui IF NOT EXISTS FOR (c:MedicalConcept) REQUIRE c.cui IS UNIQUE"
        ]
        
        indexes = [
            # Performance indexes
            "CREATE INDEX disease_icd10 IF NOT EXISTS FOR (d:Disease) ON (d.icd10_code)",
            "CREATE INDEX disease_umls IF NOT EXISTS FOR (d:Disease) ON (d.umls_cui)",
            "CREATE INDEX symptom_umls IF NOT EXISTS FOR (s:Symptom) ON (s.umls_cui)",
            "CREATE INDEX medication_rxnorm IF NOT EXISTS FOR (m:Medication) ON (m.rxnorm_code)",
            "CREATE INDEX concept_semantic_type IF NOT EXISTS FOR (c:MedicalConcept) ON (c.semantic_type)",
            # Text search indexes
            "CREATE TEXT INDEX disease_fulltext IF NOT EXISTS FOR (d:Disease) ON (d.name, d.description)",
            "CREATE TEXT INDEX symptom_fulltext IF NOT EXISTS FOR (s:Symptom) ON (s.name, s.description)"
        ]
        
        with self.driver.session() as session:
            # Create constraints
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"  ✓ Created: {constraint.split('CREATE CONSTRAINT')[1].split('IF')[0].strip()}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.debug(f"  • Constraint already exists")
                    else:
                        logger.warning(f"  ⚠️ Failed to create constraint: {e}")
            
            # Create indexes
            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"  ✓ Created: {index.split('CREATE')[1].split('IF')[0].strip()}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.debug(f"  • Index already exists")
                    else:
                        logger.warning(f"  ⚠️ Failed to create index: {e}")
    
    def load_medical_knowledge_graph(self):
        """Load comprehensive medical knowledge for sarcomas and related conditions"""
        logger.info("🏥 Loading medical knowledge graph...")
        
        cypher_queries = [
            # Create Disease nodes
            """
            MERGE (sarcoma:Disease:MedicalConcept {
                name: 'Soft Tissue Sarcoma',
                cui: 'C1261473',
                icd10_code: 'C49.9',
                umls_cui: 'C1261473',
                description: 'Malignant neoplasm arising from mesenchymal tissues',
                severity: 'high',
                category: 'oncology',
                semantic_type: 'Neoplastic Process'
            })
            """,
            
            """
            MERGE (liposarcoma:Disease:MedicalConcept {
                name: 'Liposarcoma',
                cui: 'C0023827',
                icd10_code: 'C49.4',
                umls_cui: 'C0023827',
                description: 'Malignant tumor of adipose tissue',
                severity: 'high',
                category: 'oncology',
                subtype_of: 'Soft Tissue Sarcoma',
                semantic_type: 'Neoplastic Process'
            })
            """,
            
            """
            MERGE (dediff_liposarcoma:Disease:MedicalConcept {
                name: 'Dedifferentiated Liposarcoma',
                cui: 'C1336077',
                icd10_code: 'C49.4',
                umls_cui: 'C1336077',
                description: 'High-grade sarcoma with areas of well-differentiated liposarcoma',
                severity: 'very_high',
                category: 'oncology',
                grade: 'high',
                semantic_type: 'Neoplastic Process'
            })
            """,
            
            """
            MERGE (myxofibrosarcoma:Disease:MedicalConcept {
                name: 'Myxofibrosarcoma',
                cui: 'C0334515',
                icd10_code: 'C49.9',
                umls_cui: 'C0334515',
                description: 'Malignant fibroblastic tumor with prominent myxoid stroma',
                severity: 'high',
                category: 'oncology',
                grade: 'variable',
                semantic_type: 'Neoplastic Process'
            })
            """,
            
            # Create Symptom nodes
            """
            MERGE (mass:Symptom:MedicalConcept {
                name: 'Mass',
                cui: 'C0577559',
                description: 'Palpable lump or swelling',
                semantic_type: 'Sign or Symptom'
            })
            """,
            
            """
            MERGE (pain:Symptom:MedicalConcept {
                name: 'Pain',
                cui: 'C0030193',
                description: 'Unpleasant sensory and emotional experience',
                semantic_type: 'Sign or Symptom'
            })
            """,
            
            """
            MERGE (swelling:Symptom:MedicalConcept {
                name: 'Swelling',
                cui: 'C0038999',
                description: 'Abnormal enlargement or protuberance',
                semantic_type: 'Sign or Symptom'
            })
            """,
            
            """
            MERGE (limping:Symptom:MedicalConcept {
                name: 'Limping',
                cui: 'C0023216',
                description: 'Abnormal gait pattern',
                semantic_type: 'Sign or Symptom'
            })
            """,
            
            """
            MERGE (weight_loss:Symptom:MedicalConcept {
                name: 'Weight Loss',
                cui: 'C0043096',
                description: 'Decrease in body weight',
                semantic_type: 'Sign or Symptom'
            })
            """,
            
            # Create Anatomy nodes
            """
            MERGE (thigh:Anatomy:MedicalConcept {
                name: 'Thigh',
                cui: 'C0039866',
                region: 'Lower Extremity',
                semantic_type: 'Body Part, Organ, or Organ Component'
            })
            """,
            
            """
            MERGE (retroperitoneum:Anatomy:MedicalConcept {
                name: 'Retroperitoneum',
                cui: 'C0035359',
                region: 'Abdomen',
                semantic_type: 'Body Space or Junction'
            })
            """,
            
            """
            MERGE (extremity:Anatomy:MedicalConcept {
                name: 'Extremity',
                cui: 'C0015385',
                region: 'Limbs',
                semantic_type: 'Body Part, Organ, or Organ Component'
            })
            """,
            
            # Create Treatment nodes
            """
            MERGE (surgery:Treatment:MedicalConcept {
                name: 'Surgical Resection',
                cui: 'C0015252',
                type: 'surgical',
                description: 'Complete surgical removal of tumor',
                semantic_type: 'Therapeutic or Preventive Procedure'
            })
            """,
            
            """
            MERGE (radiation:Treatment:MedicalConcept {
                name: 'Radiation Therapy',
                cui: 'C0034618',
                type: 'radiation',
                description: 'Use of ionizing radiation to kill cancer cells',
                semantic_type: 'Therapeutic or Preventive Procedure'
            })
            """,
            
            """
            MERGE (chemotherapy:Treatment:MedicalConcept {
                name: 'Chemotherapy',
                cui: 'C0013216',
                type: 'systemic',
                description: 'Use of drugs to destroy cancer cells',
                semantic_type: 'Therapeutic or Preventive Procedure'
            })
            """,
            
            # Create Relationships
            """
            MATCH (sarcoma:Disease {name: 'Soft Tissue Sarcoma'})
            MATCH (mass:Symptom {name: 'Mass'})
            MERGE (sarcoma)-[:PRESENTS_WITH {frequency: 0.9, confidence: 0.95}]->(mass)
            """,
            
            """
            MATCH (sarcoma:Disease {name: 'Soft Tissue Sarcoma'})
            MATCH (pain:Symptom {name: 'Pain'})
            MERGE (sarcoma)-[:PRESENTS_WITH {frequency: 0.6, confidence: 0.85}]->(pain)
            """,
            
            """
            MATCH (sarcoma:Disease {name: 'Soft Tissue Sarcoma'})
            MATCH (swelling:Symptom {name: 'Swelling'})
            MERGE (sarcoma)-[:PRESENTS_WITH {frequency: 0.8, confidence: 0.90}]->(swelling)
            """,
            
            """
            MATCH (liposarcoma:Disease {name: 'Liposarcoma'})
            MATCH (sarcoma:Disease {name: 'Soft Tissue Sarcoma'})
            MERGE (liposarcoma)-[:IS_SUBTYPE_OF]->(sarcoma)
            """,
            
            """
            MATCH (dediff:Disease {name: 'Dedifferentiated Liposarcoma'})
            MATCH (liposarcoma:Disease {name: 'Liposarcoma'})
            MERGE (dediff)-[:IS_SUBTYPE_OF]->(liposarcoma)
            """,
            
            """
            MATCH (myxo:Disease {name: 'Myxofibrosarcoma'})
            MATCH (sarcoma:Disease {name: 'Soft Tissue Sarcoma'})
            MERGE (myxo)-[:IS_SUBTYPE_OF]->(sarcoma)
            """,
            
            """
            MATCH (sarcoma:Disease {name: 'Soft Tissue Sarcoma'})
            MATCH (thigh:Anatomy {name: 'Thigh'})
            MERGE (sarcoma)-[:OCCURS_IN {frequency: 0.4, confidence: 0.85}]->(thigh)
            """,
            
            """
            MATCH (liposarcoma:Disease {name: 'Liposarcoma'})
            MATCH (retro:Anatomy {name: 'Retroperitoneum'})
            MERGE (liposarcoma)-[:OCCURS_IN {frequency: 0.45, confidence: 0.90}]->(retro)
            """,
            
            """
            MATCH (sarcoma:Disease {name: 'Soft Tissue Sarcoma'})
            MATCH (surgery:Treatment {name: 'Surgical Resection'})
            MERGE (sarcoma)-[:TREATED_WITH {efficacy: 0.85, first_line: true, confidence: 0.95}]->(surgery)
            """,
            
            """
            MATCH (sarcoma:Disease {name: 'Soft Tissue Sarcoma'})
            MATCH (radiation:Treatment {name: 'Radiation Therapy'})
            MERGE (sarcoma)-[:TREATED_WITH {efficacy: 0.65, adjuvant: true, confidence: 0.85}]->(radiation)
            """,
            
            """
            MATCH (dediff:Disease {name: 'Dedifferentiated Liposarcoma'})
            MATCH (chemo:Treatment {name: 'Chemotherapy'})
            MERGE (dediff)-[:TREATED_WITH {efficacy: 0.5, for_advanced: true, confidence: 0.75}]->(chemo)
            """
        ]
        
        with self.driver.session() as session:
            for query in cypher_queries:
                try:
                    session.run(query)
                    # Extract entity name from query for logging
                    if "MERGE" in query and "name:" in query:
                        entity_name = query.split("name:")[1].split("'")[1]
                        logger.info(f"  ✓ Created/Updated: {entity_name}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Query failed: {e}")
        
        logger.info("✅ Medical knowledge graph loaded successfully")
    
    def create_fol_rules(self):
        """Create FOL inference rules in the graph"""
        logger.info("🔗 Creating FOL inference rules...")
        
        rules = [
            # Rule: If patient has mass and pain, likely has sarcoma
            """
            CREATE (rule1:FOLRule {
                name: 'Sarcoma Inference Rule 1',
                condition: 'has_symptom(patient, mass) AND has_symptom(patient, pain)',
                conclusion: 'likely_has_condition(patient, soft_tissue_sarcoma)',
                confidence: 0.7
            })
            """,
            
            # Rule: If sarcoma in thigh, recommend MRI
            """
            CREATE (rule2:FOLRule {
                name: 'Imaging Recommendation Rule',
                condition: 'has_condition(patient, sarcoma) AND location(mass, thigh)',
                conclusion: 'recommend_test(patient, mri_thigh)',
                confidence: 0.9
            })
            """,
            
            # Rule: High-grade sarcoma requires multimodal treatment
            """
            CREATE (rule3:FOLRule {
                name: 'Treatment Planning Rule',
                condition: 'has_condition(patient, high_grade_sarcoma)',
                conclusion: 'requires_treatment(patient, surgery) AND requires_treatment(patient, radiation)',
                confidence: 0.85
            })
            """
        ]
        
        with self.driver.session() as session:
            for rule in rules:
                try:
                    session.run(rule)
                    logger.info(f"  ✓ Created FOL rule")
                except Exception as e:
                    logger.warning(f"  ⚠️ Rule creation failed: {e}")
    
    def verify_setup(self):
        """Verify the knowledge graph setup"""
        logger.info("\n🔍 Verifying knowledge graph setup...")
        
        with self.driver.session() as session:
            # Count nodes by type
            node_counts = {}
            for label in ['Disease', 'Symptom', 'Treatment', 'Anatomy', 'MedicalConcept']:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count = result.single()['count']
                node_counts[label] = count
                logger.info(f"  • {label}: {count} nodes")
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()['count']
            logger.info(f"  • Total relationships: {rel_count}")
            
            # Test a sample query
            result = session.run("""
                MATCH (d:Disease {name: 'Soft Tissue Sarcoma'})-[:PRESENTS_WITH]->(s:Symptom)
                RETURN d.name as disease, collect(s.name) as symptoms
            """)
            
            record = result.single()
            if record:
                logger.info(f"\n✅ Sample query successful:")
                logger.info(f"  Disease: {record['disease']}")
                logger.info(f"  Symptoms: {', '.join(record['symptoms'])}")
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
        logger.info("\n🔒 Neo4j connection closed")

def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("   Neo4j Knowledge Graph Setup for CortexMD")
    print("="*60 + "\n")
    
    # Check if plugin file exists
    plugin_path = Path("neo4j_plugins/graph-data-science.jar")
    if plugin_path.exists():
        print(f"✅ Found GDS plugin at: {plugin_path}")
        print("⚠️  Please ensure this file is copied to your Neo4j plugins folder")
        print("   Typical location: C:\\Users\\Dell\\.Neo4jDesktop\\relate-data\\dbmss\\[dbms-id]\\plugins\\")
        print()
    
    try:
        # Initialize Neo4j connection
        setup = Neo4jSetup()
        
        # Create schema
        setup.create_constraints_and_indexes()
        
        # Load medical knowledge
        setup.load_medical_knowledge_graph()
        
        # Create FOL rules
        setup.create_fol_rules()
        
        # Verify setup
        setup.verify_setup()
        
        # Close connection
        setup.close()
        
        print("\n" + "="*60)
        print("✨ Neo4j Knowledge Graph Setup Complete!")
        print("="*60)
        print("\n📊 Access Neo4j Browser at: http://localhost:7474")
        print("🔑 Credentials: neo4j / 12345678")
        print("\n🚀 Your FOL verification system is now enhanced with knowledge graphs!")
        
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
