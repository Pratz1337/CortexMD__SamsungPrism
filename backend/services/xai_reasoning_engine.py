"""
XAI (Explainable AI) Reasoning Engine for CortexMD
Provides transparent, explainable medical reasoning with FOL verification
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class XAIReasoning:
    """Complete XAI reasoning result"""
    diagnosis: str
    xai_explanation: str  # Human-readable explanation of reasoning
    supporting_evidence: List[str]  # Evidence supporting the diagnosis
    contradicting_evidence: List[str]  # Evidence against the diagnosis
    confidence_level: str  # HIGH, MEDIUM, LOW
    confidence_score: float  # 0.0 to 1.0
    fol_predicates: List[Dict[str, Any]]  # FOL predicates extracted
    fol_verification_result: Dict[str, Any]  # FOL verification results
    clinical_recommendations: List[str]
    reasoning_quality: str  # EXCELLENT, GOOD, FAIR, POOR
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            'diagnosis': self.diagnosis,
            'xai_explanation': self.xai_explanation,
            'supporting_evidence': self.supporting_evidence,
            'contradicting_evidence': self.contradicting_evidence,
            'confidence_level': self.confidence_level,
            'confidence_score': self.confidence_score,
            'fol_predicates': self.fol_predicates,
            'fol_verification': self.fol_verification_result,
            'clinical_recommendations': self.clinical_recommendations,
            'reasoning_quality': self.reasoning_quality,
            'timestamp': self.timestamp
        }


class XAIReasoningEngine:
    """
    XAI Reasoning Engine that combines:
    1. LLM-based reasoning generation (verifies diagnosis against user input)
    2. FOL predicate extraction from reasoning
    3. FOL verification against patient data
    4. Transparent, explainable output
    """
    
    def __init__(self):
        """Initialize XAI reasoning engine with required services"""
        logger.info("🧠 Initializing XAI Reasoning Engine")
        
        # Import FOL verification service
        try:
            from services.fol_verification_v2 import FOLVerificationV2
            self.fol_verifier = FOLVerificationV2()
            logger.info("✅ FOL Verification service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize FOL service: {e}")
            raise
        
        # Import confidence engine
        try:
            from ai_models.confidence_engine import DynamicConfidenceEngine
            self.confidence_engine = DynamicConfidenceEngine()
            logger.info("✅ Confidence Engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Confidence Engine: {e}")
            raise
        
        logger.info("🧠 XAI Reasoning Engine ready")
    
    async def generate_xai_reasoning(
        self,
        diagnosis: str,
        patient_data: Dict[str, Any],
        clinical_context: str,
        reasoning_paths: Optional[List[str]] = None,
        patient_id: Optional[str] = None
    ) -> XAIReasoning:
        """
        Generate complete XAI reasoning for a diagnosis
        
        Process:
        1. Generate transparent medical reasoning using LLM
        2. Extract FOL predicates from reasoning
        3. Verify predicates against patient data
        4. Assess reasoning quality and confidence
        5. Provide clinical recommendations
        
        Args:
            diagnosis: Primary diagnosis
            patient_data: Patient data dictionary with symptoms, history, etc.
            clinical_context: Clinical notes/explanation
            reasoning_paths: Optional reasoning paths from diagnosis
            patient_id: Optional patient identifier
            
        Returns:
            Complete XAI reasoning result
        """
        logger.info(f"🧠 Generating XAI reasoning for diagnosis: {diagnosis}")
        
        try:
            # Step 1: Run FOL verification (which now includes XAI reasoning generation)
            logger.info("📋 Step 1: Running FOL verification with XAI reasoning")
            fol_result = await self.fol_verifier.verify_medical_explanation(
                explanation_text=clinical_context,
                patient_data=patient_data,
                patient_id=patient_id,
                diagnosis=diagnosis,
                context={'reasoning_paths': reasoning_paths}
            )
            
            # Extract XAI explanation from FOL result
            xai_explanation = fol_result.get('xai_reasoning', '')
            medical_reasoning = fol_result.get('medical_reasoning_summary', '')
            
            # Step 2: Calculate confidence with XAI
            logger.info("📊 Step 2: Calculating confidence with XAI analysis")
            confidence_metrics = await self.confidence_engine.calculate_diagnosis_confidence(
                primary_diagnosis=diagnosis,
                patient_data=patient_data,
                reasoning_paths=reasoning_paths or []
            )
            
            # Step 3: Extract supporting and contradicting evidence
            supporting_evidence = self._extract_supporting_evidence(
                fol_result, patient_data, confidence_metrics
            )
            contradicting_evidence = confidence_metrics.contradictory_evidence
            
            # Step 4: Determine reasoning quality
            reasoning_quality = self._assess_reasoning_quality(
                fol_result, confidence_metrics
            )
            
            # Step 5: Generate clinical recommendations
            clinical_recommendations = self._generate_xai_recommendations(
                fol_result, confidence_metrics, reasoning_quality
            )
            
            # Combine XAI explanation with medical reasoning
            combined_explanation = f"{xai_explanation}\n\n{medical_reasoning}"
            
            # Create XAI reasoning result
            xai_reasoning = XAIReasoning(
                diagnosis=diagnosis,
                xai_explanation=combined_explanation,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                confidence_level=fol_result.get('confidence_level', 'MEDIUM'),
                confidence_score=confidence_metrics.overall_confidence,
                fol_predicates=fol_result.get('predicates', []),
                fol_verification_result=fol_result,
                clinical_recommendations=clinical_recommendations,
                reasoning_quality=reasoning_quality
            )
            
            logger.info(f"✅ XAI reasoning generated successfully")
            logger.info(f"   Confidence: {xai_reasoning.confidence_level} ({xai_reasoning.confidence_score:.2f})")
            logger.info(f"   Quality: {reasoning_quality}")
            logger.info(f"   FOL Predicates: {len(xai_reasoning.fol_predicates)}")
            
            return xai_reasoning
            
        except Exception as e:
            logger.error(f"XAI reasoning generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return minimal XAI reasoning on error
            return XAIReasoning(
                diagnosis=diagnosis,
                xai_explanation=f"XAI reasoning generation encountered an error: {str(e)}",
                supporting_evidence=["Error in analysis"],
                contradicting_evidence=[],
                confidence_level="LOW",
                confidence_score=0.3,
                fol_predicates=[],
                fol_verification_result={'status': 'ERROR', 'error': str(e)},
                clinical_recommendations=["Clinical validation required due to analysis error"],
                reasoning_quality="POOR"
            )
    
    def _extract_supporting_evidence(
        self,
        fol_result: Dict[str, Any],
        patient_data: Dict[str, Any],
        confidence_metrics: Any
    ) -> List[str]:
        """Extract supporting evidence from FOL and confidence analysis"""
        evidence = []
        
        # Evidence from verified FOL predicates
        predicates = fol_result.get('predicates', [])
        verified_predicates = [p for p in predicates if p.get('verified', False)]
        
        for pred in verified_predicates[:5]:  # Top 5 verified predicates
            pred_type = pred.get('type', 'unknown')
            obj = pred.get('object', '')
            evidence_list = pred.get('evidence', [])
            
            if evidence_list:
                evidence.append(f"{pred_type.replace('_', ' ').title()}: {obj} - {evidence_list[0]}")
        
        # Evidence from confidence metrics
        if confidence_metrics.reasoning:
            for reason in confidence_metrics.reasoning[:3]:
                if "support" in reason.lower() or "confirm" in reason.lower():
                    evidence.append(reason)
        
        # Evidence from patient symptoms
        symptoms = patient_data.get('symptoms', [])
        if symptoms:
            evidence.append(f"Patient reported {len(symptoms)} relevant symptoms")
        
        # If no evidence found, add default
        if not evidence:
            evidence.append("Diagnosis based on clinical assessment")
        
        return evidence[:7]  # Limit to 7 pieces of evidence
    
    def _assess_reasoning_quality(
        self,
        fol_result: Dict[str, Any],
        confidence_metrics: Any
    ) -> str:
        """Assess the quality of the XAI reasoning"""
        # Calculate quality score
        fol_success_rate = fol_result.get('success_rate', 0.0)
        confidence_score = confidence_metrics.overall_confidence
        total_predicates = fol_result.get('total_predicates', 0)
        
        # Quality based on multiple factors
        quality_score = (fol_success_rate * 0.4) + (confidence_score * 0.4)
        
        # Bonus for having many predicates (more comprehensive analysis)
        if total_predicates >= 10:
            quality_score += 0.1
        elif total_predicates >= 5:
            quality_score += 0.05
        
        # Penalty for low predicate count
        if total_predicates < 3:
            quality_score -= 0.15
        
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Determine quality category
        if quality_score >= 0.8:
            return "EXCELLENT"
        elif quality_score >= 0.6:
            return "GOOD"
        elif quality_score >= 0.4:
            return "FAIR"
        else:
            return "POOR"
    
    def _generate_xai_recommendations(
        self,
        fol_result: Dict[str, Any],
        confidence_metrics: Any,
        reasoning_quality: str
    ) -> List[str]:
        """Generate XAI-specific clinical recommendations"""
        recommendations = []
        
        # Recommendations from FOL verification
        fol_recommendations = fol_result.get('clinical_recommendations', [])
        recommendations.extend(fol_recommendations[:3])
        
        # Recommendations based on reasoning quality
        if reasoning_quality == "POOR":
            recommendations.append("⚠️ XAI analysis suggests additional clinical data needed for reliable diagnosis")
            recommendations.append("Consider comprehensive patient history review and diagnostic tests")
        elif reasoning_quality == "FAIR":
            recommendations.append("XAI analysis shows moderate confidence - verify with additional clinical assessment")
        elif reasoning_quality == "EXCELLENT":
            recommendations.append("✅ XAI analysis shows high confidence with strong evidence base")
        
        # Recommendations based on uncertainty
        if confidence_metrics.uncertainty_score > 0.6:
            recommendations.append("High diagnostic uncertainty detected - consider differential diagnoses")
        
        # Check for contradicting evidence
        if confidence_metrics.contradictory_evidence:
            recommendations.append(f"⚠️ {len(confidence_metrics.contradictory_evidence)} contradicting factors identified - review carefully")
        
        # Limit to 5 recommendations
        return recommendations[:5]


# Utility function for easy integration
async def generate_xai_reasoning_for_diagnosis(
    diagnosis: str,
    patient_data: Dict[str, Any],
    clinical_context: str,
    reasoning_paths: Optional[List[str]] = None,
    patient_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Utility function to generate XAI reasoning
    
    Returns dictionary with complete XAI analysis
    """
    engine = XAIReasoningEngine()
    xai_result = await engine.generate_xai_reasoning(
        diagnosis=diagnosis,
        patient_data=patient_data,
        clinical_context=clinical_context,
        reasoning_paths=reasoning_paths,
        patient_id=patient_id
    )
    return xai_result.to_dict()
