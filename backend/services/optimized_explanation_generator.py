"""
Optimized Medical Explanation Generator
High-performance explanation generation with single-pass verification
"""
import asyncio
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import time

try:
    from ..core.models import MedicalExplanation, DiagnosisResult, PatientInput
    from .optimized_fol_verification_service import (
        OptimizedFOLVerificationService, 
        create_fast_patient_data_structure
    )
except ImportError:
    from core.models import MedicalExplanation, DiagnosisResult, PatientInput
    from services.optimized_fol_verification_service import (
        OptimizedFOLVerificationService, 
        create_fast_patient_data_structure
    )

logger = logging.getLogger(__name__)

@dataclass
class OptimizedExplanationResult:
    """Result from optimized explanation generation"""
    explanations: List[MedicalExplanation]
    verification_report: Optional[Dict[str, Any]]
    generation_time: float
    verification_time: float

class OptimizedExplanationGenerator:
    """
    High-performance explanation generator that eliminates redundant verification loops
    """
    
    def __init__(self, model=None):
        self.model = model
        self.fol_service = OptimizedFOLVerificationService()
        logger.info("Optimized Explanation Generator initialized")
    
    async def generate_explanations_with_verification(
        self,
        diagnosis_result: DiagnosisResult,
        patient_input: PatientInput
    ) -> OptimizedExplanationResult:
        """
        Generate explanations with single-pass verification (no loops!)
        """
        start_time = time.time()
        
        try:
            # Step 1: Generate explanations in one pass
            explanations = await self._generate_explanations_batch(
                diagnosis_result, patient_input
            )
            generation_time = time.time() - start_time
            
            if not explanations:
                return OptimizedExplanationResult(
                    explanations=[],
                    verification_report=None,
                    generation_time=generation_time,
                    verification_time=0.0
                )
            
            # Step 2: Single verification pass for ALL explanations
            verification_start = time.time()
            verification_report = await self._batch_verify_all_explanations(
                explanations, patient_input
            )
            verification_time = time.time() - verification_start
            
            # Step 3: Update explanation confidence based on verification
            self._update_explanation_confidence(explanations, verification_report)
            
            logger.info(
                f"Generated {len(explanations)} explanations in {generation_time:.2f}s "
                f"with verification in {verification_time:.2f}s"
            )
            
            return OptimizedExplanationResult(
                explanations=explanations,
                verification_report=verification_report,
                generation_time=generation_time,
                verification_time=verification_time
            )
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return OptimizedExplanationResult(
                explanations=self._create_fallback_explanations(diagnosis_result),
                verification_report=None,
                generation_time=time.time() - start_time,
                verification_time=0.0
            )
    
    async def _generate_explanations_batch(
        self,
        diagnosis_result: DiagnosisResult,
        patient_input: PatientInput
    ) -> List[MedicalExplanation]:
        """
        Generate explanations in a single batch operation
        """
        # Create comprehensive context
        patient_context = self._extract_patient_context(patient_input)
        
        # Create optimized prompt for batch generation
        prompt = self._create_batch_explanation_prompt(
            diagnosis_result, patient_context
        )
        
        try:
            if self.model:
                response = self.model.generate_content(prompt)
                explanation_text = response.text.strip()
            else:
                # Fallback to structured explanation generation
                explanation_text = self._generate_structured_explanations(
                    diagnosis_result, patient_context
                )
            
            # Parse explanations efficiently
            explanations = self._parse_explanations_fast(explanation_text)
            
            # Create MedicalExplanation objects with base confidence
            medical_explanations = []
            for i, explanation in enumerate(explanations):
                confidence = self._calculate_base_confidence(
                    explanation, diagnosis_result.confidence_score, i
                )
                
                medical_explanations.append(
                    MedicalExplanation(
                        id=f"explanation_{i+1}",
                        explanation=explanation.strip(),
                        confidence=confidence,
                        verified=False  # Will be updated after verification
                    )
                )
            
            return medical_explanations[:5]  # Limit to 5 explanations
            
        except Exception as e:
            logger.error(f"Batch explanation generation failed: {e}")
            return self._create_fallback_explanations(diagnosis_result)
    
    async def _batch_verify_all_explanations(
        self,
        explanations: List[MedicalExplanation],
        patient_input: PatientInput
    ) -> Dict[str, Any]:
        """
        Verify ALL explanations in a single batch operation
        """
        if not explanations:
            return {}
        
        # Combine all explanations into one verification request
        combined_text = "\n\n".join([exp.explanation for exp in explanations])
        patient_data = create_fast_patient_data_structure(patient_input)
        
        # Single verification call for all explanations
        verification_report = await self.fol_service.verify_medical_explanation(
            explanation_text=combined_text,
            patient_data=patient_data,
            patient_id=patient_input.patient_id or "batch_verification"
        )
        
        return {
            'total_predicates': verification_report.total_predicates,
            'verified_predicates': verification_report.verified_predicates,
            'overall_confidence': verification_report.overall_confidence,
            'medical_reasoning_summary': verification_report.medical_reasoning_summary,
            'verification_time': verification_report.verification_time,
            'detailed_results': verification_report.detailed_results
        }
    
    def _update_explanation_confidence(
        self,
        explanations: List[MedicalExplanation],
        verification_report: Dict[str, Any]
    ):
        """
        Update explanation confidence based on verification results
        """
        if not verification_report:
            return
        
        overall_verification_confidence = verification_report.get('overall_confidence', 0.0)
        success_rate = verification_report.get('verified_predicates', 0) / max(
            verification_report.get('total_predicates', 1), 1
        )
        
        for explanation in explanations:
            # Boost confidence for verified explanations
            if success_rate >= 0.7:
                explanation.confidence = min(0.95, explanation.confidence * 1.15)
                explanation.verified = True
            elif success_rate >= 0.4:
                explanation.confidence = min(0.85, explanation.confidence * 1.05)
                explanation.verified = True
            else:
                explanation.confidence = max(0.3, explanation.confidence * 0.9)
                explanation.verified = False
    
    def _extract_patient_context(self, patient_input: PatientInput) -> Dict[str, Any]:
        """
        Extract patient context for explanation generation
        """
        context = {
            'symptoms': [],
            'vital_signs': {},
            'lab_results': {},
            'imaging': [],
            'medical_history': [],
            'patient_info': ''
        }
        
        # Extract from text data
        if patient_input.text_data:
            text = patient_input.text_data.lower()
            
            # Quick symptom extraction
            symptoms = []
            symptom_keywords = [
                'pain', 'fever', 'nausea', 'vomiting', 'headache', 'fatigue',
                'weakness', 'dizziness', 'shortness of breath', 'cough',
                'chest pain', 'abdominal pain', 'back pain', 'flank pain'
            ]
            
            for keyword in symptom_keywords:
                if keyword in text:
                    symptoms.append(keyword)
            
            context['symptoms'] = symptoms[:5]  # Limit to top 5
            context['patient_info'] = patient_input.text_data[:300]
        
        # Extract from FHIR data
        if patient_input.fhir_data:
            fhir = patient_input.fhir_data
            if 'observations' in fhir:
                context['vital_signs'].update(fhir['observations'])
            if 'conditions' in fhir:
                context['medical_history'].extend(fhir['conditions'])
        
        return context
    
    def _create_batch_explanation_prompt(
        self,
        diagnosis_result: DiagnosisResult,
        patient_context: Dict[str, Any]
    ) -> str:
        """
        Create optimized prompt for batch explanation generation
        """
        # Build patient summary
        patient_summary = ""
        if patient_context['patient_info']:
            patient_summary = f"Patient presents with: {patient_context['patient_info'][:200]}"
        
        if patient_context['symptoms']:
            symptoms_text = ", ".join(patient_context['symptoms'][:3])
            patient_summary += f" Key symptoms include: {symptoms_text}."
        
        if patient_context['vital_signs']:
            vs_items = list(patient_context['vital_signs'].items())[:2]
            if vs_items:
                vs_text = ", ".join([f"{k}: {v}" for k, v in vs_items])
                patient_summary += f" Vital signs: {vs_text}."
        
        # Create comprehensive prompt
        prompt = f"""
MEDICAL CASE ANALYSIS:
{patient_summary or "Patient case for medical analysis."}

PRIMARY DIAGNOSIS: {diagnosis_result.primary_diagnosis}
DIAGNOSTIC CONFIDENCE: {diagnosis_result.confidence_score:.1%}

CLINICAL REASONING PROVIDED:
{chr(10).join(diagnosis_result.reasoning_paths[:2]) if diagnosis_result.reasoning_paths else "Clinical analysis performed"}

TASK: Generate exactly 5 comprehensive medical explanations (150-200 words each) that explain why this diagnosis is appropriate for this patient.

REQUIREMENTS:
- Use patient-specific clinical data provided above
- Include detailed medical reasoning and pathophysiology
- Reference relevant symptoms, vital signs, and clinical findings
- Provide clear, evidence-based medical explanations
- Each explanation should offer unique perspective on the diagnosis
- Write in complete paragraphs without headers

FORMAT: Provide exactly 5 paragraphs separated by double line breaks (\\n\\n). Each paragraph should be a complete medical explanation.
"""
        
        return prompt
    
    def _generate_structured_explanations(
        self,
        diagnosis_result: DiagnosisResult,
        patient_context: Dict[str, Any]
    ) -> str:
        """
        Fallback structured explanation generation
        """
        diagnosis = diagnosis_result.primary_diagnosis
        
        explanations = [
            f"The diagnosis of {diagnosis} is supported by the patient's clinical presentation and symptomatology. "
            f"Based on the assessment, there is {diagnosis_result.confidence_score:.1%} confidence in this diagnosis. "
            f"The clinical findings are consistent with the expected manifestations of {diagnosis}, "
            f"and the patient's reported symptoms align with the typical presentation pattern.",
            
            f"From a pathophysiological perspective, {diagnosis} involves specific disease mechanisms that explain "
            f"the patient's current clinical state. The diagnostic confidence of {diagnosis_result.confidence_score:.1%} "
            f"reflects the strength of clinical evidence supporting this conclusion. "
            f"The patient's presentation follows the expected disease progression and symptom complex.",
            
            f"The differential diagnostic process led to {diagnosis} as the most likely explanation for the patient's "
            f"clinical findings. With {diagnosis_result.confidence_score:.1%} confidence, this diagnosis best fits "
            f"the available clinical data. The patient's symptoms and examination findings support this conclusion "
            f"over alternative diagnostic possibilities.",
            
            f"Clinical correlation strongly supports the diagnosis of {diagnosis} in this patient case. "
            f"The diagnostic assessment yields {diagnosis_result.confidence_score:.1%} confidence based on "
            f"comprehensive evaluation of presenting symptoms and clinical indicators. "
            f"This diagnosis explains the patient's clinical syndrome effectively.",
            
            f"Evidence-based analysis confirms {diagnosis} as the primary diagnostic consideration. "
            f"The {diagnosis_result.confidence_score:.1%} confidence level reflects robust clinical support "
            f"for this diagnosis. The patient's clinical presentation, symptoms, and available data "
            f"collectively support this diagnostic conclusion with appropriate medical reasoning."
        ]
        
        return "\n\n".join(explanations)
    
    def _parse_explanations_fast(self, text: str) -> List[str]:
        """
        Fast explanation parsing without complex processing
        """
        # Split on double newlines first
        parts = text.split('\n\n')
        
        explanations = []
        for part in parts:
            part = part.strip()
            if len(part) >= 100:  # Minimum explanation length
                # Remove common prefixes/headers
                part = re.sub(r'^(?:\d+\.\s*|Explanation \d+:\s*|•\s*)', '', part, flags=re.IGNORECASE)
                explanations.append(part)
        
        # If not enough explanations, split by sentences
        if len(explanations) < 3:
            sentences = text.split('.')
            current_explanation = ""
            explanations = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    current_explanation += sentence + ". "
                    if len(current_explanation) >= 150:  # Target length
                        explanations.append(current_explanation.strip())
                        current_explanation = ""
        
        return explanations[:5]  # Limit to 5 explanations
    
    def _calculate_base_confidence(
        self, 
        explanation: str, 
        base_confidence: float, 
        index: int
    ) -> float:
        """
        Calculate base confidence for explanation
        """
        # Start with base confidence
        confidence = base_confidence
        
        # Adjust based on explanation quality
        word_count = len(explanation.split())
        if word_count >= 100:
            confidence *= 1.1  # Boost for detailed explanations
        elif word_count < 50:
            confidence *= 0.8  # Reduce for brief explanations
        
        # Slight reduction for later explanations (natural ordering)
        confidence *= (0.95 ** index)
        
        # Medical terminology bonus
        medical_terms = [
            'diagnosis', 'symptoms', 'clinical', 'medical', 'patient',
            'condition', 'disease', 'syndrome', 'pathology', 'treatment'
        ]
        term_count = sum(1 for term in medical_terms if term in explanation.lower())
        confidence *= (1 + term_count * 0.02)  # Small boost per medical term
        
        return min(0.95, max(0.1, confidence))
    
    def _create_fallback_explanations(
        self, 
        diagnosis_result: DiagnosisResult
    ) -> List[MedicalExplanation]:
        """
        Create fallback explanations when generation fails
        """
        diagnosis = diagnosis_result.primary_diagnosis
        
        return [
            MedicalExplanation(
                id="fallback_1",
                explanation=f"The diagnosis of {diagnosis} is supported by clinical assessment and available patient data.",
                confidence=0.6,
                verified=False
            ),
            MedicalExplanation(
                id="fallback_2",
                explanation=f"Patient presentation is consistent with {diagnosis} based on comprehensive evaluation.",
                confidence=0.5,
                verified=False
            )
        ]

# Utility function for easy integration
async def generate_optimized_explanations(
    diagnosis_result: DiagnosisResult,
    patient_input: PatientInput,
    model=None
) -> OptimizedExplanationResult:
    """
    Quick function to generate optimized explanations with verification
    """
    generator = OptimizedExplanationGenerator(model)
    return await generator.generate_explanations_with_verification(
        diagnosis_result, patient_input
    )
