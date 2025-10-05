"""
Dynamic AI Confidence Engine for CortexMD
Real-time confidence scoring using Hybrid AI (Groq + Gemini)
"""
import os
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import re
from concurrent.futures import ThreadPoolExecutor
import logging

# Import hybrid AI engine
try:
    from .hybrid_ai_engine import HybridAIEngine, TaskType
except ImportError:
    from ai_models.hybrid_ai_engine import HybridAIEngine, TaskType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConfidenceMetrics:
    """Detailed confidence metrics for medical diagnosis"""
    symptom_match_score: float  # How well symptoms match the diagnosis
    evidence_quality_score: float  # Quality of supporting evidence
    medical_literature_score: float  # Alignment with medical literature
    uncertainty_score: float  # Measure of diagnostic uncertainty
    overall_confidence: float  # Final confidence score
    reasoning: List[str]  # Explanation of confidence factors
    risk_factors: List[str]  # Identified risk factors
    contradictory_evidence: List[str]  # Evidence against the diagnosis

@dataclass
class DiagnosisEvidence:
    """Evidence supporting or contradicting a diagnosis"""
    evidence_type: str  # 'symptom', 'lab', 'imaging', 'history'
    evidence_text: str
    support_strength: float  # -1.0 to 1.0 (negative = contradictory)
    confidence: float  # 0.0 to 1.0
    medical_relevance: float  # 0.0 to 1.0

class DynamicConfidenceEngine:
    """AI-powered dynamic confidence scoring using Hybrid AI system"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the confidence engine with hybrid AI models"""
        
        # Initialize hybrid AI engine (Groq + Gemini)
        self.hybrid_ai = HybridAIEngine()
        
        # Medical knowledge base for validation
        self.medical_conditions = self._load_medical_knowledge()
        
        logger.info("Dynamic Confidence Engine initialized with Hybrid AI (Groq + Gemini)")
        
    def _load_medical_knowledge(self) -> Dict[str, Any]:
        """Load medical knowledge base for validation"""
        return {
            'cardiovascular': {
                'conditions': ['acute coronary syndrome', 'myocardial infarction', 'angina', 'heart failure'],
                'symptoms': ['chest pain', 'shortness of breath', 'palpitations', 'fatigue'],
                'risk_factors': ['hypertension', 'diabetes', 'smoking', 'family history']
            },
            'respiratory': {
                'conditions': ['pneumonia', 'asthma', 'copd', 'pulmonary embolism'],
                'symptoms': ['cough', 'shortness of breath', 'wheezing', 'chest pain'],
                'risk_factors': ['smoking', 'allergies', 'environmental exposure']
            },
            'endocrine': {
                'conditions': ['diabetes', 'thyroid disorder', 'adrenal insufficiency'],
                'symptoms': ['fatigue', 'weight changes', 'mood changes', 'temperature intolerance'],
                'risk_factors': ['family history', 'obesity', 'autoimmune conditions']
            }
        }
    
    async def calculate_diagnosis_confidence(
        self, 
        primary_diagnosis: str,
        patient_data: Dict[str, Any],
        reasoning_paths: List[str],
        differential_diagnoses: Optional[List[str]] = None
    ) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence metrics using Hybrid AI with XAI reasoning
        
        This method now includes XAI (Explainable AI) capabilities:
        1. Generates transparent reasoning about diagnosis confidence
        2. Identifies specific evidence supporting or contradicting the diagnosis
        3. Provides explainable confidence scores
        """
        
        logger.info(f"Calculating XAI-enhanced confidence for diagnosis: {primary_diagnosis}")
        
        try:
            # Use hybrid AI for comprehensive confidence analysis with XAI
            confidence_data = await self._hybrid_xai_confidence_analysis(
                primary_diagnosis,
                patient_data,
                reasoning_paths,
                differential_diagnoses
            )
            
            # Extract metrics from AI response
            symptom_score = confidence_data.get('symptom_match_score', 0.5)
            evidence_score = confidence_data.get('evidence_quality_score', 0.5)
            literature_score = confidence_data.get('medical_literature_score', 0.5)
            uncertainty_score = confidence_data.get('uncertainty_score', 0.5)
            overall_confidence = confidence_data.get('overall_confidence', 0.5)
            
            # Generate additional analysis
            risk_factors = self._identify_risk_factors(patient_data)
            contradictory_evidence = confidence_data.get('contradictory_evidence', [])
            reasoning = confidence_data.get('reasoning', [])
            
            # Add XAI-specific reasoning
            xai_explanation = confidence_data.get('xai_explanation', '')
            if xai_explanation:
                reasoning.insert(0, f"XAI Analysis: {xai_explanation}")
            
            return ConfidenceMetrics(
                symptom_match_score=symptom_score,
                evidence_quality_score=evidence_score,
                medical_literature_score=literature_score,
                uncertainty_score=uncertainty_score,
                overall_confidence=overall_confidence,
                reasoning=reasoning,
                risk_factors=risk_factors,
                contradictory_evidence=contradictory_evidence
            )
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            # Return conservative confidence metrics on error
            return ConfidenceMetrics(
                symptom_match_score=0.3,
                evidence_quality_score=0.3,
                medical_literature_score=0.3,
                uncertainty_score=0.7,
                overall_confidence=0.3,
                reasoning=[f"Error in confidence calculation: {str(e)}"],
                risk_factors=[],
                contradictory_evidence=["Unable to assess contradictory evidence due to error"]
            )
    
    async def _hybrid_xai_confidence_analysis(
        self,
        primary_diagnosis: str,
        patient_data: Dict[str, Any],
        reasoning_paths: List[str],
        differential_diagnoses: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform XAI-enhanced confidence analysis using hybrid AI
        
        This generates explainable confidence metrics with transparent reasoning
        """
        # Prepare patient summary
        symptoms_str = ', '.join(patient_data.get('symptoms', [])) if 'symptoms' in patient_data else 'Not documented'
        history_str = ', '.join(patient_data.get('medical_history', [])) if 'medical_history' in patient_data else 'Not documented'
        
        xai_prompt = f"""
Analyze the confidence level for this medical diagnosis with transparent, explainable reasoning.

PRIMARY DIAGNOSIS: {primary_diagnosis}

PATIENT DATA:
- Symptoms: {symptoms_str}
- Medical History: {history_str}
- Clinical Notes: {patient_data.get('clinical_notes', 'Not available')[:200]}

REASONING PATHS: {' '.join(reasoning_paths[:3])}

Provide a JSON response with:
{{
  "symptom_match_score": <0.0-1.0>,
  "evidence_quality_score": <0.0-1.0>,
  "medical_literature_score": <0.0-1.0>,
  "uncertainty_score": <0.0-1.0>,
  "overall_confidence": <0.0-1.0>,
  "reasoning": ["Reason 1", "Reason 2", "Reason 3"],
  "contradictory_evidence": ["Any contradicting evidence"],
  "xai_explanation": "Transparent explanation of confidence scoring and how patient data supports or challenges the diagnosis"
}}

Focus on explainability - make it clear WHY the confidence is at this level.
"""
        
        try:
            # Use hybrid AI engine
            response = await self.hybrid_ai.generate_response(
                prompt=xai_prompt,
                task_type=TaskType.CONFIDENCE_ANALYSIS
            )
            
            # Parse JSON from response
            import json
            import re
            
            content = response.content
            # Clean JSON markers
            content = re.sub(r'```json?\s*', '', content)
            content = re.sub(r'```\s*$', '', content)
            
            # Find JSON object
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                confidence_data = json.loads(json_str)
                return confidence_data
            else:
                logger.warning("Could not parse JSON from XAI confidence response")
                return self._generate_fallback_confidence_data(primary_diagnosis, patient_data)
                
        except Exception as e:
            logger.error(f"XAI confidence analysis failed: {e}")
            return self._generate_fallback_confidence_data(primary_diagnosis, patient_data)
    
    def _generate_fallback_confidence_data(
        self,
        primary_diagnosis: str,
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate fallback confidence data when AI analysis fails"""
        symptoms = patient_data.get('symptoms', [])
        has_history = bool(patient_data.get('medical_history', []))
        
        # Basic confidence based on data availability
        base_confidence = 0.5
        if symptoms:
            base_confidence += 0.1 * min(len(symptoms), 3)
        if has_history:
            base_confidence += 0.1
        
        base_confidence = min(base_confidence, 0.85)
        
        return {
            'symptom_match_score': base_confidence,
            'evidence_quality_score': base_confidence * 0.9,
            'medical_literature_score': 0.6,
            'uncertainty_score': 1.0 - base_confidence,
            'overall_confidence': base_confidence,
            'reasoning': [
                f"Confidence based on {len(symptoms)} documented symptoms",
                "Medical history context available" if has_history else "Limited medical history",
                "Fallback confidence assessment used"
            ],
            'contradictory_evidence': [],
            'xai_explanation': f"Confidence assessment for {primary_diagnosis} is based on available patient data. Limited AI analysis - clinical validation recommended."
        }
    
    def _identify_risk_factors(self, patient_data: Dict[str, Any]) -> List[str]:
        """Identify risk factors from patient data"""
        risk_factors = []
        
        # Medical history risk factors
        if 'medical_history' in patient_data:
            history = patient_data['medical_history']
            if isinstance(history, list):
                risk_factors.extend(history)
            elif isinstance(history, str):
                risk_factors.append(history)
        
        # Demographic risk factors
        if 'age' in patient_data:
            age = patient_data['age']
            if isinstance(age, (int, float)) and age > 65:
                risk_factors.append('advanced age')
        
        # Lifestyle risk factors
        if 'smoking' in str(patient_data).lower():
            risk_factors.append('smoking history')
        
        return risk_factors
    
    async def calculate_confidence_metrics(
        self,
        primary_diagnosis: str,
        evidence_list: List[Dict[str, Any]],
        patient_input
    ) -> ConfidenceMetrics:
        """Calculate confidence metrics with FOL evidence integration"""
        
        logger.info(f"Calculating FOL-enhanced confidence metrics for: {primary_diagnosis}")
        
        try:
            # Prepare patient data from input
            patient_data = {}
            if hasattr(patient_input, 'fhir_data') and patient_input.fhir_data:
                patient_data.update(patient_input.fhir_data)
            if hasattr(patient_input, 'text_data') and patient_input.text_data:
                patient_data['clinical_notes'] = patient_input.text_data
            
            # Calculate base confidence using existing method
            base_confidence = await self.calculate_diagnosis_confidence(
                primary_diagnosis,
                patient_data,
                [f"Evidence from {len(evidence_list)} FOL predicates"]
            )
            
            # Enhance with FOL evidence
            fol_evidence_score = self._calculate_fol_evidence_score(evidence_list)
            fol_consistency_score = self._calculate_fol_consistency_score(evidence_list)
            
            # Adjust confidence based on FOL evidence
            enhanced_confidence = self._integrate_fol_confidence(
                base_confidence, fol_evidence_score, fol_consistency_score
            )
            
            return enhanced_confidence
            
        except Exception as e:
            logger.error(f"Error calculating FOL-enhanced confidence metrics: {e}")
            # Return basic confidence metrics on error
            return ConfidenceMetrics(
                symptom_match_score=0.5,
                evidence_quality_score=0.3,
                medical_literature_score=0.4,
                uncertainty_score=0.6,
                overall_confidence=0.4,
                reasoning=[f"FOL confidence calculation error: {str(e)}"],
                risk_factors=[],
                contradictory_evidence=["Unable to assess FOL evidence due to error"]
            )
    
    def _calculate_fol_evidence_score(self, evidence_list: List[Dict[str, Any]]) -> float:
        """Calculate evidence quality score from FOL predicates"""
        if not evidence_list:
            return 0.0
        
        total_score = 0.0
        for evidence in evidence_list:
            # Weight by support strength and confidence
            support_strength = evidence.get('support_strength', 0.0)
            confidence = evidence.get('confidence', 0.0)
            medical_relevance = evidence.get('medical_relevance', 0.5)
            
            # Calculate weighted score
            evidence_score = (support_strength * 0.4) + (confidence * 0.4) + (medical_relevance * 0.2)
            total_score += evidence_score
        
        return min(1.0, total_score / len(evidence_list))
    
    def _calculate_fol_consistency_score(self, evidence_list: List[Dict[str, Any]]) -> float:
        """Calculate consistency score across FOL evidence"""
        if not evidence_list:
            return 0.0
        
        # Count supporting vs contradicting evidence
        supporting_count = sum(1 for e in evidence_list if e.get('support_strength', 0.0) > 0.5)
        contradicting_count = sum(1 for e in evidence_list if e.get('support_strength', 0.0) < -0.5)
        neutral_count = len(evidence_list) - supporting_count - contradicting_count
        
        # Calculate consistency based on evidence distribution
        if len(evidence_list) == 0:
            return 0.0
        
        # Higher consistency when more evidence supports and less contradicts
        consistency_score = (supporting_count + (neutral_count * 0.5)) / len(evidence_list)
        
        # Penalize contradicting evidence
        if contradicting_count > 0:
            contradiction_penalty = contradicting_count / len(evidence_list)
            consistency_score = consistency_score * (1.0 - contradiction_penalty)
        
        return min(1.0, consistency_score)
    
    def _integrate_fol_confidence(
        self,
        base_confidence: ConfidenceMetrics,
        fol_evidence_score: float,
        fol_consistency_score: float
    ) -> ConfidenceMetrics:
        """Integrate FOL scores with base confidence metrics"""
        
        # Enhance evidence quality with FOL evidence
        enhanced_evidence_quality = (
            base_confidence.evidence_quality_score * 0.6 +
            fol_evidence_score * 0.4
        )
        
        # Reduce uncertainty based on FOL consistency
        enhanced_uncertainty = base_confidence.uncertainty_score * (1.0 - fol_consistency_score * 0.3)
        
        # Calculate enhanced overall confidence
        enhanced_overall = (
            base_confidence.overall_confidence * 0.5 +
            fol_evidence_score * 0.3 +
            fol_consistency_score * 0.2
        )
        
        # Add FOL-specific reasoning
        enhanced_reasoning = base_confidence.reasoning.copy()
        enhanced_reasoning.append(f"FOL evidence quality: {fol_evidence_score:.2f}")
        enhanced_reasoning.append(f"FOL consistency score: {fol_consistency_score:.2f}")
        
        if fol_evidence_score > 0.8:
            enhanced_reasoning.append("Strong FOL predicate verification supports diagnosis")
        elif fol_evidence_score < 0.4:
            enhanced_reasoning.append("Weak FOL predicate verification - additional evidence needed")
        
        return ConfidenceMetrics(
            symptom_match_score=base_confidence.symptom_match_score,
            evidence_quality_score=enhanced_evidence_quality,
            medical_literature_score=base_confidence.medical_literature_score,
            uncertainty_score=enhanced_uncertainty,
            overall_confidence=min(1.0, enhanced_overall),
            reasoning=enhanced_reasoning,
            risk_factors=base_confidence.risk_factors,
            contradictory_evidence=base_confidence.contradictory_evidence
        )

class FOLLogicEngine:
    """Deterministic logic engine for FOL verification"""

    def __init__(self, ontology: Dict[str, Any]):
        self.ontology = ontology

    def validate_predicates(self, predicates: List[str], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate logical predicates against patient data and ontology"""
        results = []
        for predicate in predicates:
            result = self._validate_predicate(predicate, patient_data)
            results.append(result)
        return {
            "total_predicates": len(predicates),
            "verified_predicates": sum(1 for r in results if r["verified"]),
            "failed_predicates": sum(1 for r in results if not r["verified"]),
            "predicate_details": results,
        }

    def _validate_predicate(self, predicate: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single predicate deterministically"""
        # Example: "patient has fever" -> Check if "fever" exists in symptoms
        tokens = predicate.lower().split()
        if "patient" in tokens and "has" in tokens:
            condition = tokens[-1]
            verified = condition in patient_data.get("symptoms", [])
            return {
                "predicate": predicate,
                "verified": verified,
                "evidence": f"Matched condition '{condition}' in patient symptoms" if verified else "No match found",
            }
        return {
            "predicate": predicate,
            "verified": False,
            "evidence": "Unsupported predicate structure",
        }

# Utility functions for integration
async def calculate_dynamic_confidence(
    diagnosis: str,
    patient_data: Dict[str, Any],
    reasoning_paths: List[str],
    differential_diagnoses: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> ConfidenceMetrics:
    """Utility function to calculate confidence with dynamic engine"""
    
    engine = DynamicConfidenceEngine(api_key=api_key)
    return await engine.calculate_diagnosis_confidence(
        diagnosis, patient_data, reasoning_paths, differential_diagnoses
    )
