"""
Neo4j Configuration for Medical Knowledge Graph
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Neo4jConfig:
    """Configuration for Neo4j connection and knowledge graph setup"""

    # Neo4j connection settings
    URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    # Graph configuration
    NODE_LABELS = {
        "MedicalConcept": "MedicalConcept",
        "Disease": "Disease",
        "Symptom": "Symptom",
        "Medication": "Medication",
        "LabTest": "LabTest",
        "Procedure": "Procedure",
        "Anatomy": "Anatomy"
    }

    RELATIONSHIP_TYPES = {
        "HAS_SYMPTOM": "HAS_SYMPTOM",
        "TREATED_BY": "TREATED_BY",
        "CAUSES": "CAUSES",
        "ASSOCIATED_WITH": "ASSOCIATED_WITH",
        "PART_OF": "PART_OF",
        "IS_A": "IS_A",
        "RELATED_TO": "RELATED_TO",
        "HAS_LAB_VALUE": "HAS_LAB_VALUE",
        "PERFORMS_PROCEDURE": "PERFORMS_PROCEDURE"
    }

    # Ontology sources
    ONTOLOGY_SOURCES = {
        "UMLS": {
            "name": "Unified Medical Language System",
            "base_url": "https://uts-ws.nlm.nih.gov",
            "api_key": os.getenv("UMLS_API_KEY"),
            "version": "2023AA"
        },

        "ICD10": {
            "name": "International Classification of Diseases v10",
            "base_url": "https://icd.who.int",
            "version": "2023"
        }
    }

    # Graph schema constraints
    CONSTRAINTS = [
        "CREATE CONSTRAINT concept_cui_unique IF NOT EXISTS FOR (c:MedicalConcept) REQUIRE c.cui IS UNIQUE",
        "CREATE CONSTRAINT disease_name_unique IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT medication_name_unique IF NOT EXISTS FOR (m:Medication) REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT symptom_name_unique IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE"
    ]

    # Indexes for performance
    INDEXES = [
        "CREATE INDEX concept_name_idx IF NOT EXISTS FOR (c:MedicalConcept) ON (c.preferred_name)",
        "CREATE INDEX concept_synonym_idx IF NOT EXISTS FOR (c:MedicalConcept) ON (c.synonyms)",
        "CREATE INDEX concept_semantic_type_idx IF NOT EXISTS FOR (c:MedicalConcept) ON (c.semantic_types)",
        "CREATE FULLTEXT INDEX concept_fulltext_idx IF NOT EXISTS FOR (c:MedicalConcept) ON EACH [c.preferred_name, c.synonyms, c.definition]"
    ]

    @classmethod
    def get_connection_config(cls) -> dict:
        """Get Neo4j connection configuration"""
        return {
            "uri": cls.URI,
            "auth": (cls.USERNAME, cls.PASSWORD),
            "database": cls.DATABASE
        }

    @classmethod
    def get_ontology_config(cls, source: str) -> Optional[dict]:
        """Get configuration for specific ontology source"""
        return cls.ONTOLOGY_SOURCES.get(source.upper())

    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            print(f"Warning: Missing required Neo4j environment variables: {missing_vars}")
            return False

        return True

# Environment-specific configurations
class DevelopmentConfig(Neo4jConfig):
    """Development environment configuration"""
    URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

class ProductionConfig(Neo4jConfig):
    """Production environment configuration"""
    URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Configuration factory
def get_config(env: str = None) -> Neo4jConfig:
    """Get configuration based on environment"""
    env = env or os.getenv("ENV", "development").lower()

    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "dev": DevelopmentConfig,
        "prod": ProductionConfig
    }

    return configs.get(env, DevelopmentConfig)()
