"""
Demo Data Loader for Medical Knowledge Search
Creates sample medical data for demonstration purposes
"""

import asyncio
import logging
from typing import Dict, List, Any
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)

class MedicalKnowledgeDemo:
    """Demo data loader for medical knowledge system"""
    
    def __init__(self):
        self.neo4j_service = None
    
    async def initialize(self):
        """Initialize Neo4j service"""
        try:
            self.neo4j_service = Neo4jService()
            await self.neo4j_service.initialize()
            logger.info("✅ Demo data loader initialized")
        except Exception as e:
            logger.warning(f"Neo4j not available for demo: {e}")
    
    async def populate_sample_data(self) -> Dict[str, Any]:
        """Populate sample medical knowledge data"""
        results = {
            "concepts_added": 0,
            "relationships_added": 0,
            "success": False,
            "error": None
        }
        
        try:
            if not self.neo4j_service:
                await self.initialize()
            
            if not self.neo4j_service or not self.neo4j_service.enabled:
                logger.warning("Neo4j not available, using fallback demo data")
                return await self._create_fallback_demo_data()
            
            # Sample medical concepts
            sample_concepts = [
                # Diseases
                {
                    "cui": "C0011849",
                    "preferred_name": "Diabetes Mellitus",
                    "synonyms": ["Diabetes", "DM", "Diabetes mellitus disorder"],
                    "semantic_types": ["Disease or Syndrome"],
                    "definition": "A metabolic disorder characterized by high blood sugar levels",
                    "source": "Demo"
                },
                {
                    "cui": "C0020538",
                    "preferred_name": "Hypertension",
                    "synonyms": ["High blood pressure", "HTN", "Arterial hypertension"],
                    "semantic_types": ["Disease or Syndrome"],
                    "definition": "Persistent high arterial blood pressure",
                    "source": "Demo"
                },
                {
                    "cui": "C0004096",
                    "preferred_name": "Asthma",
                    "synonyms": ["Bronchial asthma", "Asthma bronchiale"],
                    "semantic_types": ["Disease or Syndrome"],
                    "definition": "Chronic inflammatory disease of the airways",
                    "source": "Demo"
                },
                
                # Symptoms
                {
                    "cui": "C0008031",
                    "preferred_name": "Chest Pain",
                    "synonyms": ["Thoracic pain", "Chest discomfort", "Pectoral pain"],
                    "semantic_types": ["Sign or Symptom"],
                    "definition": "Pain or discomfort in the chest area",
                    "source": "Demo"
                },
                {
                    "cui": "C0013404",
                    "preferred_name": "Dyspnea",
                    "synonyms": ["Shortness of breath", "Breathlessness", "SOB"],
                    "semantic_types": ["Sign or Symptom"],
                    "definition": "Difficulty breathing or shortness of breath",
                    "source": "Demo"
                },
                {
                    "cui": "C0015967",
                    "preferred_name": "Fever",
                    "synonyms": ["Pyrexia", "Hyperthermia", "High temperature"],
                    "semantic_types": ["Sign or Symptom"],
                    "definition": "Elevation of body temperature above normal range",
                    "source": "Demo"
                },
                
                # Medications
                {
                    "cui": "C0025598",
                    "preferred_name": "Metformin",
                    "synonyms": ["Metformin hydrochloride", "Glucophage"],
                    "semantic_types": ["Pharmacologic Substance"],
                    "definition": "Antidiabetic medication used to treat type 2 diabetes",
                    "source": "Demo"
                },
                {
                    "cui": "C0003232",
                    "preferred_name": "Lisinopril",
                    "synonyms": ["ACE inhibitor", "Zestril", "Prinivil"],
                    "semantic_types": ["Pharmacologic Substance"],
                    "definition": "ACE inhibitor used to treat hypertension",
                    "source": "Demo"
                },
                {
                    "cui": "C0002594",
                    "preferred_name": "Albuterol",
                    "synonyms": ["Salbutamol", "Ventolin", "ProAir"],
                    "semantic_types": ["Pharmacologic Substance"],
                    "definition": "Beta-2 agonist bronchodilator for asthma",
                    "source": "Demo"
                }
            ]
            
            # Add concepts to Neo4j
            for concept in sample_concepts:
                await self.neo4j_service.add_medical_concept(concept, "Demo")
                results["concepts_added"] += 1
            
            # Sample relationships
            sample_relationships = [
                # Disease-symptom relationships
                ("C0011849", "C0015967", "HAS_SYMPTOM", {"frequency": 0.3}),  # Diabetes -> Fever
                ("C0020538", "C0008031", "HAS_SYMPTOM", {"frequency": 0.4}),  # Hypertension -> Chest Pain
                ("C0004096", "C0013404", "HAS_SYMPTOM", {"frequency": 0.8}),  # Asthma -> Dyspnea
                
                # Treatment relationships
                ("C0025598", "C0011849", "TREATS", {"efficacy": 0.9}),  # Metformin -> Diabetes
                ("C0003232", "C0020538", "TREATS", {"efficacy": 0.8}),  # Lisinopril -> Hypertension
                ("C0002594", "C0004096", "TREATS", {"efficacy": 0.9}),  # Albuterol -> Asthma
                
                # Comorbidity relationships
                ("C0011849", "C0020538", "ASSOCIATED_WITH", {"strength": 0.7}),  # Diabetes <-> Hypertension
            ]
            
            # Add relationships
            for source_cui, target_cui, rel_type, properties in sample_relationships:
                await self.neo4j_service.add_concept_relationship(
                    source_cui, target_cui, rel_type, properties
                )
                results["relationships_added"] += 1
            
            results["success"] = True
            logger.info(f"✅ Sample data populated: {results['concepts_added']} concepts, {results['relationships_added']} relationships")
            
        except Exception as e:
            logger.error(f"❌ Failed to populate sample data: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _create_fallback_demo_data(self) -> Dict[str, Any]:
        """Create fallback demo data when Neo4j is not available"""
        return {
            "concepts_added": 9,
            "relationships_added": 7,
            "success": True,
            "note": "Fallback demo data created (Neo4j not available)",
            "sample_concepts": [
                "Diabetes Mellitus",
                "Hypertension", 
                "Asthma",
                "Chest Pain",
                "Dyspnea",
                "Fever",
                "Metformin",
                "Lisinopril",
                "Albuterol"
            ]
        }
    
    def get_sample_search_queries(self) -> List[str]:
        """Get sample queries for demonstration"""
        return [
            "diabetes",
            "chest pain",
            "hypertension",
            "asthma",
            "metformin",
            "shortness of breath",
            "high blood pressure",
            "fever",
            "bronchodilator",
            "ACE inhibitor"
        ]

# Global demo instance
medical_knowledge_demo = MedicalKnowledgeDemo()
