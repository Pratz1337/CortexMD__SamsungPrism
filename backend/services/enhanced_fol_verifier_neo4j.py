"""
Enhanced FOL Verifier with Neo4j Knowledge Graph Integration
This module integrates Neo4j knowledge graph for improved medical predicate verification
"""

import re
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import os

from services.fol_logic_engine import DeterministicFOLVerifier, Predicate, Term
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)

@dataclass
class Neo4jVerificationResult:
    """Result from Neo4j knowledge graph verification"""
    verified: bool
    confidence: float
    reasoning: str
    knowledge_path: Optional[List[str]] = None
    related_concepts: Optional[List[str]] = None

class EnhancedFOLVerifierWithNeo4j(DeterministicFOLVerifier):
    """
    Enhanced FOL verifier that uses Neo4j knowledge graph for medical reasoning
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize Neo4j service if enabled
        self.neo4j_enabled = os.getenv('NEO4J_ENABLED', 'false').lower() == 'true'
        self.neo4j_service = None
        
        if self.neo4j_enabled:
            try:
                self.neo4j_service = Neo4jService()
                # Connect to Neo4j asynchronously will be done in verify methods
                logger.info("✅ Neo4j knowledge graph service initialized")
            except Exception as e:
                logger.warning(f"⚠️ Neo4j service initialization failed: {e}")
                self.neo4j_enabled = False
    
    def evaluate_predicate(self, predicate: Predicate, patient_data: Dict) -> Tuple[bool, float]:
        """
        Evaluate a predicate using the parent class's deterministic logic
        Delegates to parent class implementation
        """
        # Store patient data for evaluation
        self.patient_data = patient_data
        self.bindings = {}
        
        # Call parent class method
        return super().evaluate_predicate(predicate)
    
    async def verify_predicate_enhanced(
        self, 
        predicate: Predicate, 
        patient_data: Dict,
        use_neo4j: bool = True
    ) -> Tuple[bool, float, str]:
        """
        Enhanced predicate verification using both deterministic logic and Neo4j knowledge graph
        
        Args:
            predicate: FOL predicate to verify
            patient_data: Patient data for verification
            use_neo4j: Whether to use Neo4j for additional verification
            
        Returns:
            Tuple of (verified, confidence, reasoning)
        """
        # First, try deterministic verification
        det_verified, det_confidence = self.evaluate_predicate(predicate, patient_data)
        det_reasoning = self._generate_reasoning(predicate, det_verified, det_confidence)
        
        # If Neo4j is not enabled or not requested, return deterministic results
        if not self.neo4j_enabled or not use_neo4j or not self.neo4j_service:
            return det_verified, det_confidence, det_reasoning
        
        # Enhance with Neo4j knowledge graph verification
        try:
            neo4j_result = await self._verify_with_neo4j(predicate, patient_data)
            
            # Combine results
            combined_verified, combined_confidence, combined_reasoning = self._combine_verification_results(
                det_verified, det_confidence, det_reasoning,
                neo4j_result
            )
            
            return combined_verified, combined_confidence, combined_reasoning
            
        except Exception as e:
            logger.warning(f"Neo4j verification failed, falling back to deterministic: {e}")
            return det_verified, det_confidence, det_reasoning
    
    async def _verify_with_neo4j(self, predicate: Predicate, patient_data: Dict) -> Neo4jVerificationResult:
        """
        Verify predicate using Neo4j knowledge graph
        """
        if not self.neo4j_service:
            return Neo4jVerificationResult(False, 0.0, "Neo4j service not available")
        
        try:
            # Extract predicate components
            pred_name = predicate.name
            pred_args = [str(arg.value) for arg in predicate.arguments]
            
            # Handle different predicate types
            if pred_name == "has_symptom":
                return await self._verify_symptom_neo4j(pred_args, patient_data)
            elif pred_name == "has_condition":
                return await self._verify_condition_neo4j(pred_args, patient_data)
            elif pred_name == "takes_medication":
                return await self._verify_medication_neo4j(pred_args, patient_data)
            elif pred_name == "has_lab_value":
                return await self._verify_lab_neo4j(pred_args, patient_data)
            else:
                # Default verification
                return await self._verify_generic_neo4j(pred_name, pred_args, patient_data)
                
        except Exception as e:
            logger.error(f"Neo4j verification error: {e}")
            return Neo4jVerificationResult(False, 0.0, f"Error: {str(e)}")
    
    async def _verify_symptom_neo4j(self, args: List[str], patient_data: Dict) -> Neo4jVerificationResult:
        """Verify symptom using Neo4j knowledge graph"""
        if len(args) < 2:
            return Neo4jVerificationResult(False, 0.0, "Invalid symptom predicate")
        
        symptom = args[1].replace('_', ' ')
        patient_symptoms = patient_data.get('symptoms', [])
        
        # Direct match check
        for patient_symptom in patient_symptoms:
            if symptom.lower() in patient_symptom.lower() or patient_symptom.lower() in symptom.lower():
                return Neo4jVerificationResult(True, 0.95, f"Direct symptom match: {symptom}")
        
        # Use Neo4j to find related symptoms
        try:
            # Query Neo4j for symptom relationships
            query = """
            MATCH (s1:Symptom {name: $symptom})-[r:RELATED_TO|INDICATES|CAUSED_BY*1..2]-(s2:Symptom)
            RETURN s2.name as related_symptom, type(r) as relationship
            LIMIT 10
            """
            
            # Ensure Neo4j is connected
            if not self.neo4j_service.driver:
                await self.neo4j_service.connect()
            
            related_symptoms = await self.neo4j_service.execute_query(
                query, 
                {"symptom": symptom.lower()}
            )
            
            # Check if any related symptoms match patient symptoms
            for record in related_symptoms:
                related = record.get('related_symptom', '')
                for patient_symptom in patient_symptoms:
                    if related.lower() in patient_symptom.lower():
                        return Neo4jVerificationResult(
                            True, 
                            0.75,
                            f"Related symptom match: {symptom} related to {related}",
                            knowledge_path=[symptom, related]
                        )
            
            # Check for symptom-disease relationships
            disease_query = """
            MATCH (s:Symptom {name: $symptom})-[:INDICATES]->(d:Disease)
            RETURN d.name as disease
            LIMIT 10
            """
            
            diseases = await self.neo4j_service.execute_query(
                disease_query,
                {"symptom": symptom.lower()}
            )
            
            # Check if patient has related conditions
            patient_conditions = patient_data.get('conditions', [])
            for record in diseases:
                disease = record.get('disease', '')
                for condition in patient_conditions:
                    if disease.lower() in condition.lower():
                        return Neo4jVerificationResult(
                            True,
                            0.70,
                            f"Symptom {symptom} indicates condition {disease} which patient has",
                            knowledge_path=[symptom, "indicates", disease]
                        )
            
        except Exception as e:
            logger.error(f"Neo4j symptom verification error: {e}")
        
        return Neo4jVerificationResult(False, 0.1, f"No evidence for symptom: {symptom}")
    
    async def _verify_condition_neo4j(self, args: List[str], patient_data: Dict) -> Neo4jVerificationResult:
        """Verify medical condition using Neo4j knowledge graph"""
        if len(args) < 2:
            return Neo4jVerificationResult(False, 0.0, "Invalid condition predicate")
        
        condition = args[1].replace('_', ' ')
        
        # Check diagnosis and conditions
        diagnosis = patient_data.get('diagnosis', '')
        conditions = patient_data.get('conditions', [])
        
        # Direct match
        if condition.lower() in diagnosis.lower():
            return Neo4jVerificationResult(True, 0.95, f"Condition matches diagnosis: {condition}")
        
        for patient_condition in conditions:
            if condition.lower() in patient_condition.lower():
                return Neo4jVerificationResult(True, 0.90, f"Direct condition match: {condition}")
        
        # Use Neo4j for semantic matching
        try:
            # Find related conditions
            query = """
            MATCH (c1:Disease {name: $condition})-[r:IS_A|SUBTYPE_OF|RELATED_TO*1..2]-(c2:Disease)
            RETURN c2.name as related_condition, type(r) as relationship
            LIMIT 10
            """
            
            related_conditions = await self.neo4j_service.execute_query(
                query,
                {"condition": condition.lower()}
            )
            
            for record in related_conditions:
                related = record.get('related_condition', '')
                if related.lower() in diagnosis.lower():
                    return Neo4jVerificationResult(
                        True,
                        0.80,
                        f"Related condition match: {condition} related to {related}",
                        knowledge_path=[condition, "related_to", related]
                    )
            
            # Check symptoms that indicate this condition
            symptom_query = """
            MATCH (s:Symptom)-[:INDICATES]->(d:Disease {name: $condition})
            RETURN s.name as symptom
            LIMIT 10
            """
            
            indicating_symptoms = await self.neo4j_service.execute_query(
                symptom_query,
                {"condition": condition.lower()}
            )
            
            patient_symptoms = patient_data.get('symptoms', [])
            matching_symptoms = []
            
            for record in indicating_symptoms:
                symptom = record.get('symptom', '')
                for patient_symptom in patient_symptoms:
                    if symptom.lower() in patient_symptom.lower():
                        matching_symptoms.append(symptom)
            
            if len(matching_symptoms) >= 2:
                return Neo4jVerificationResult(
                    True,
                    0.75,
                    f"Multiple symptoms indicate {condition}: {', '.join(matching_symptoms)}",
                    related_concepts=matching_symptoms
                )
            elif len(matching_symptoms) == 1:
                return Neo4jVerificationResult(
                    True,
                    0.60,
                    f"Symptom {matching_symptoms[0]} indicates {condition}",
                    related_concepts=matching_symptoms
                )
            
        except Exception as e:
            logger.error(f"Neo4j condition verification error: {e}")
        
        return Neo4jVerificationResult(False, 0.1, f"No evidence for condition: {condition}")
    
    async def _verify_medication_neo4j(self, args: List[str], patient_data: Dict) -> Neo4jVerificationResult:
        """Verify medication using Neo4j knowledge graph"""
        if len(args) < 2:
            return Neo4jVerificationResult(False, 0.0, "Invalid medication predicate")
        
        medication = args[1].replace('_', ' ')
        patient_meds = patient_data.get('current_medications', [])
        
        # Direct match
        for patient_med in patient_meds:
            if medication.lower() in patient_med.lower():
                return Neo4jVerificationResult(True, 0.95, f"Direct medication match: {medication}")
        
        # Use Neo4j for medication relationships
        try:
            # Find related medications (same class, similar effects)
            query = """
            MATCH (m1:Medication {name: $medication})-[r:SAME_CLASS|ALTERNATIVE_TO]-(m2:Medication)
            RETURN m2.name as related_med
            LIMIT 10
            """
            
            related_meds = await self.neo4j_service.execute_query(
                query,
                {"medication": medication.lower()}
            )
            
            for record in related_meds:
                related = record.get('related_med', '')
                for patient_med in patient_meds:
                    if related.lower() in patient_med.lower():
                        return Neo4jVerificationResult(
                            True,
                            0.70,
                            f"Patient takes related medication: {related} (alternative to {medication})",
                            knowledge_path=[medication, "alternative_to", related]
                        )
            
            # Check if medication is indicated for patient's conditions
            condition_query = """
            MATCH (m:Medication {name: $medication})-[:TREATS]->(d:Disease)
            RETURN d.name as disease
            LIMIT 10
            """
            
            treated_conditions = await self.neo4j_service.execute_query(
                condition_query,
                {"medication": medication.lower()}
            )
            
            patient_conditions = patient_data.get('conditions', [])
            diagnosis = patient_data.get('diagnosis', '')
            
            for record in treated_conditions:
                disease = record.get('disease', '')
                if disease.lower() in diagnosis.lower() or any(disease.lower() in c.lower() for c in patient_conditions):
                    return Neo4jVerificationResult(
                        True,
                        0.65,
                        f"Medication {medication} is indicated for patient's condition: {disease}",
                        knowledge_path=[medication, "treats", disease]
                    )
            
        except Exception as e:
            logger.error(f"Neo4j medication verification error: {e}")
        
        return Neo4jVerificationResult(False, 0.1, f"No evidence for medication: {medication}")
    
    async def _verify_lab_neo4j(self, args: List[str], patient_data: Dict) -> Neo4jVerificationResult:
        """Verify lab values using Neo4j knowledge graph"""
        if len(args) < 3:
            return Neo4jVerificationResult(False, 0.0, "Invalid lab value predicate")
        
        lab_name = args[1].replace('_', ' ')
        expected_value = args[2]
        
        # First check actual lab values
        labs = patient_data.get('lab_results', {})
        
        for lab_key, lab_value in labs.items():
            if lab_name.lower() in lab_key.lower():
                # Compare values
                is_match, confidence = self._compare_lab_value(lab_value, expected_value, lab_name)
                if is_match:
                    return Neo4jVerificationResult(
                        True,
                        confidence,
                        f"Lab value {lab_name} matches expected: {expected_value}"
                    )
        
        # Use Neo4j to infer from related conditions
        try:
            # Find conditions that typically show this lab abnormality
            query = """
            MATCH (l:LabTest {name: $lab_name})-[:ABNORMAL_IN]->(d:Disease)
            WHERE l.abnormal_value = $expected_value
            RETURN d.name as disease
            LIMIT 10
            """
            
            associated_conditions = await self.neo4j_service.execute_query(
                query,
                {"lab_name": lab_name.lower(), "expected_value": expected_value}
            )
            
            patient_conditions = patient_data.get('conditions', [])
            diagnosis = patient_data.get('diagnosis', '')
            
            for record in associated_conditions:
                disease = record.get('disease', '')
                if disease.lower() in diagnosis.lower() or any(disease.lower() in c.lower() for c in patient_conditions):
                    return Neo4jVerificationResult(
                        True,
                        0.60,
                        f"Lab abnormality {lab_name}={expected_value} is consistent with patient's {disease}",
                        knowledge_path=[lab_name, "abnormal_in", disease]
                    )
            
        except Exception as e:
            logger.error(f"Neo4j lab verification error: {e}")
        
        return Neo4jVerificationResult(False, 0.1, f"No evidence for lab value: {lab_name}={expected_value}")
    
    async def _verify_generic_neo4j(self, pred_name: str, args: List[str], patient_data: Dict) -> Neo4jVerificationResult:
        """Generic Neo4j verification for other predicate types"""
        try:
            # Try to find any relevant relationships in the knowledge graph
            if len(args) >= 2:
                entity = args[1].replace('_', ' ')
                
                query = """
                MATCH (n {name: $entity})-[r]-(m)
                RETURN type(r) as relationship, labels(m) as target_labels, m.name as target_name
                LIMIT 10
                """
                
                relationships = await self.neo4j_service.execute_query(
                    query,
                    {"entity": entity.lower()}
                )
                
                if relationships:
                    related_concepts = [r.get('target_name', '') for r in relationships]
                    return Neo4jVerificationResult(
                        False,
                        0.3,
                        f"Found related concepts for {entity} in knowledge graph",
                        related_concepts=related_concepts
                    )
            
        except Exception as e:
            logger.error(f"Neo4j generic verification error: {e}")
        
        return Neo4jVerificationResult(False, 0.0, "Unable to verify using knowledge graph")
    
    def _combine_verification_results(
        self,
        det_verified: bool,
        det_confidence: float,
        det_reasoning: str,
        neo4j_result: Neo4jVerificationResult
    ) -> Tuple[bool, float, str]:
        """
        Combine deterministic and Neo4j verification results
        """
        # Weighted combination (60% deterministic, 40% Neo4j)
        det_weight = 0.6
        neo4j_weight = 0.4
        
        # Convert boolean to confidence score
        det_score = det_confidence if det_verified else (1 - det_confidence)
        neo4j_score = neo4j_result.confidence if neo4j_result.verified else (1 - neo4j_result.confidence)
        
        # Combined confidence
        combined_confidence = det_weight * det_score + neo4j_weight * neo4j_score
        
        # Determine if verified (threshold at 0.5)
        combined_verified = combined_confidence >= 0.5
        
        # Generate combined reasoning
        combined_reasoning = f"Deterministic: {det_reasoning} (conf: {det_confidence:.2f})\n"
        combined_reasoning += f"Knowledge Graph: {neo4j_result.reasoning} (conf: {neo4j_result.confidence:.2f})\n"
        
        if neo4j_result.knowledge_path:
            combined_reasoning += f"Knowledge Path: {' -> '.join(neo4j_result.knowledge_path)}\n"
        
        if neo4j_result.related_concepts:
            combined_reasoning += f"Related Concepts: {', '.join(neo4j_result.related_concepts)}\n"
        
        combined_reasoning += f"Combined Confidence: {combined_confidence:.2f}"
        
        return combined_verified, combined_confidence, combined_reasoning
    
    def parse_predicate_string(self, pred_str: str) -> Predicate:
        """Parse a predicate string into a Predicate object"""
        import re
        
        # Parse predicate string format: predicate_name(arg1, arg2, ...)
        match = re.match(r'(\¬)?(\w+)\((.*)\)', pred_str)
        if not match:
            raise ValueError(f"Invalid predicate format: {pred_str}")
        
        negated = bool(match.group(1))
        name = match.group(2)
        args_str = match.group(3)
        
        # Parse arguments
        args = []
        for arg in args_str.split(','):
            arg = arg.strip()
            if arg.isupper() and len(arg) == 1:  # Variable
                args.append(Term(arg, 'variable'))
            else:  # Constant
                args.append(Term(arg, 'constant', value=arg))
        
        return Predicate(name, args, negated)
    
    def _generate_reasoning(self, predicate: Predicate, verified: bool, confidence: float) -> str:
        """Generate reasoning explanation for verification result"""
        pred_str = str(predicate)
        
        if verified:
            if confidence >= 0.9:
                return f"Strong evidence supports {pred_str}"
            elif confidence >= 0.7:
                return f"Good evidence supports {pred_str}"
            elif confidence >= 0.5:
                return f"Moderate evidence supports {pred_str}"
            else:
                return f"Weak evidence supports {pred_str}"
        else:
            if confidence <= 0.1:
                return f"No evidence found for {pred_str}"
            elif confidence <= 0.3:
                return f"Limited evidence for {pred_str}"
            else:
                return f"Insufficient evidence for {pred_str}"


async def test_enhanced_verifier():
    """Test the enhanced FOL verifier with Neo4j integration"""
    verifier = EnhancedFOLVerifierWithNeo4j()
    
    # Test patient data
    patient_data = {
        "symptoms": ["headache", "nausea"],
        "diagnosis": "intracranial neoplasm",
        "conditions": ["hypertension"],
        "current_medications": ["aspirin"],
        "lab_results": {
            "glucose": 95,
            "hemoglobin": 14.5
        }
    }
    
    # Test predicates
    test_predicates = [
        "has_symptom(patient, headache)",
        "has_condition(patient, neoplasm)",
        "takes_medication(patient, aspirin)",
        "has_lab_value(patient, glucose, normal)"
    ]
    
    print("Testing Enhanced FOL Verifier with Neo4j")
    print("=" * 50)
    
    for pred_str in test_predicates:
        # Parse predicate
        predicate = verifier.parse_predicate_string(pred_str)
        
        # Verify with enhancement
        verified, confidence, reasoning = await verifier.verify_predicate_enhanced(
            predicate,
            patient_data
        )
        
        print(f"\nPredicate: {pred_str}")
        print(f"Verified: {verified}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Reasoning: {reasoning}")
        print("-" * 30)


if __name__ == "__main__":
    # Run test
    asyncio.run(test_enhanced_verifier())
