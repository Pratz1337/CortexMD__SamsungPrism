# XAI (Explainable AI) System for CortexMD

## Overview

The XAI (Explainable AI) system in CortexMD provides transparent, verifiable, and explainable medical reasoning for AI-generated diagnoses. This system bridges the gap between AI predictions and clinical understanding by:

1. **Verifying AI Diagnosis Against Patient Input** - Uses LLMs to analyze how the diagnosis relates to actual patient data
2. **Generating Transparent Reasoning** - Creates human-readable explanations of diagnostic logic
3. **FOL Predicate Extraction** - Converts medical reasoning into First-Order Logic predicates
4. **Predicate Verification** - Validates predicates against patient data for consistency
5. **Clinical Recommendations** - Provides actionable insights based on verification results

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DIAGNOSIS ENGINE                          │
│            (Generates Primary Diagnosis)                     │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│               XAI REASONING ENGINE                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Step 1: LLM-Based Reasoning Generation              │  │
│  │  - Compares diagnosis with patient input             │  │
│  │  - Identifies supporting evidence                    │  │
│  │  - Identifies contradicting evidence                 │  │
│  │  - Generates transparent medical reasoning           │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Step 2: FOL Predicate Extraction                    │  │
│  │  - Extracts predicates from XAI reasoning            │  │
│  │  - Creates structured logic statements               │  │
│  │  - Types: has_symptom, has_condition, etc.           │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Step 3: FOL Verification                            │  │
│  │  - Verifies each predicate against patient data      │  │
│  │  - Calculates verification success rate              │  │
│  │  - Determines confidence levels                      │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Step 4: Clinical Assessment                         │  │
│  │  - Assesses reasoning quality                        │  │
│  │  - Generates clinical recommendations                │  │
│  │  - Provides explainable output                       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│            XAI VERIFICATION RESULT                           │
│  - Transparent reasoning                                     │
│  - Supporting/contradicting evidence                         │
│  - Verified FOL predicates                                   │
│  - Confidence metrics                                        │
│  - Clinical recommendations                                  │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. XAI Reasoning Engine (`services/xai_reasoning_engine.py`)

**Purpose**: Orchestrates the entire XAI verification process.

**Key Functions**:
- `generate_xai_reasoning()` - Main entry point for XAI analysis
- Coordinates FOL verification and confidence analysis
- Generates comprehensive XAI results

**Output Structure**:
```python
{
    'diagnosis': str,
    'xai_explanation': str,  # Human-readable reasoning
    'supporting_evidence': List[str],
    'contradicting_evidence': List[str],
    'confidence_level': str,  # HIGH, MEDIUM, LOW
    'confidence_score': float,
    'fol_predicates': List[Dict],
    'fol_verification': Dict,
    'clinical_recommendations': List[str],
    'reasoning_quality': str  # EXCELLENT, GOOD, FAIR, POOR
}
```

### 2. FOL Verification V2 (`services/fol_verification_v2.py`)

**Purpose**: Extracts and verifies First-Order Logic predicates with XAI reasoning.

**Enhanced Features**:
- `_generate_xai_reasoning()` - Uses LLMs to verify diagnosis against patient input
- `_extract_predicates()` - Extracts FOL predicates using AI (Gemini/Groq)
- `_verify_predicate()` - Validates predicates against patient data
- `_generate_medical_reasoning()` - Creates clinical reasoning summaries

**Predicate Types**:
- `has_symptom(patient, symptom_name)` - Patient symptoms
- `has_condition(patient, condition_name)` - Medical conditions
- `takes_medication(patient, medication_name)` - Current medications
- `has_vital_sign(patient, vital_type, value)` - Vital signs
- `has_lab_value(patient, lab_test, value)` - Laboratory results

**XAI Reasoning Prompt**:
The system prompts the LLM to:
1. Verify how diagnosis matches patient symptoms
2. List supporting evidence from patient data
3. Identify contradicting evidence
4. Note missing critical information
5. Explain clinical reasoning
6. Assess confidence level

### 3. Enhanced Confidence Engine (`ai_models/confidence_engine.py`)

**Purpose**: Calculates confidence metrics with XAI explanations.

**XAI Enhancements**:
- `_hybrid_xai_confidence_analysis()` - Generates explainable confidence scores
- Provides transparent reasoning for confidence levels
- Identifies evidence quality issues
- Highlights contradictory evidence

**Confidence Metrics**:
- `symptom_match_score` (0.0-1.0)
- `evidence_quality_score` (0.0-1.0)
- `medical_literature_score` (0.0-1.0)
- `uncertainty_score` (0.0-1.0)
- `overall_confidence` (0.0-1.0)

## Integration in Main Diagnosis Flow

### Location: `core/app.py` - `run_comprehensive_diagnosis()`

```python
# Step 1: Generate diagnosis (existing)
diagnosis_result = await processor.generate_dynamic_diagnosis(patient_input)

# Step 2: Run XAI-enhanced FOL verification (NEW)
xai_engine = XAIReasoningEngine()
xai_result = await xai_engine.generate_xai_reasoning(
    diagnosis=diagnosis_result.primary_diagnosis,
    patient_data=patient_data,
    clinical_context=explanation_text,
    reasoning_paths=diagnosis_result.reasoning_paths
)

# Step 3: Store XAI results in session
diagnosis_sessions[session_id]['xai_reasoning'] = {
    'xai_explanation': xai_result.xai_explanation,
    'supporting_evidence': xai_result.supporting_evidence,
    'contradicting_evidence': xai_result.contradicting_evidence,
    'confidence_level': xai_result.confidence_level,
    'reasoning_quality': xai_result.reasoning_quality,
    'clinical_recommendations': xai_result.clinical_recommendations
}
```

## Usage Examples

### Example 1: Basic XAI Reasoning

```python
from services.xai_reasoning_engine import XAIReasoningEngine

# Initialize engine
xai_engine = XAIReasoningEngine()

# Generate XAI reasoning
xai_result = await xai_engine.generate_xai_reasoning(
    diagnosis="Myxofibrosarcoma",
    patient_data={
        'symptoms': ['chest pain', 'mass', 'swelling'],
        'medical_history': ['previous surgery'],
        'chief_complaint': 'Patient reports painful mass in chest area'
    },
    clinical_context="Patient presents with painful chest mass...",
    reasoning_paths=["Imaging shows soft tissue mass", "Biopsy confirms sarcoma"]
)

# Access results
print(f"XAI Explanation: {xai_result.xai_explanation}")
print(f"Confidence: {xai_result.confidence_level}")
print(f"Supporting Evidence: {xai_result.supporting_evidence}")
print(f"Recommendations: {xai_result.clinical_recommendations}")
```

### Example 2: Direct FOL Verification with XAI

```python
from services.fol_verification_v2 import FOLVerificationV2

fol_verifier = FOLVerificationV2()

# Verify medical explanation with XAI
fol_result = await fol_verifier.verify_medical_explanation(
    explanation_text="Patient has chest pain and mass consistent with sarcoma",
    patient_data={'symptoms': ['chest pain', 'mass']},
    diagnosis="Myxofibrosarcoma"
)

# XAI reasoning is included
print(f"XAI Reasoning: {fol_result['xai_reasoning']}")
print(f"Verified Predicates: {fol_result['verified_predicates']}/{fol_result['total_predicates']}")
```

## XAI Output Example

```json
{
  "diagnosis": "Myxofibrosarcoma",
  "xai_explanation": "XAI Verification Analysis: The diagnosis of 'Myxofibrosarcoma' is supported by patient's reported symptoms including chest pain and palpable mass. Patient medical history includes previous surgical intervention. Clinical findings show high consistency with patient data (8/10 predicates verified, 80.0% success rate). Key symptoms confirmed: chest pain, mass, swelling. The verified findings provide strong support for the diagnosis of Myxofibrosarcoma. Confidence level: HIGH based on available patient input data.",
  
  "supporting_evidence": [
    "Has Symptom: chest pain - Found in symptoms: chest pain",
    "Has Symptom: mass - Found in symptoms: mass",
    "Has Condition: myxofibrosarcoma - Matches primary diagnosis",
    "Patient reported 3 relevant symptoms",
    "Strong FOL predicate verification supports diagnosis"
  ],
  
  "contradicting_evidence": [],
  
  "confidence_level": "HIGH",
  "confidence_score": 0.85,
  
  "fol_predicates": [
    {
      "fol_string": "has_symptom(patient, chest pain)",
      "type": "has_symptom",
      "object": "chest pain",
      "confidence": 0.9,
      "verified": true,
      "evidence": ["Found in symptoms: chest pain"]
    },
    {
      "fol_string": "has_symptom(patient, mass)",
      "type": "has_symptom",
      "object": "mass",
      "confidence": 0.85,
      "verified": true,
      "evidence": ["Found in symptoms: mass"]
    }
  ],
  
  "clinical_recommendations": [
    "✅ XAI analysis shows high confidence with strong evidence base",
    "Clinical findings are well-documented and consistent with diagnosis",
    "Continue with current clinical management plan"
  ],
  
  "reasoning_quality": "EXCELLENT"
}
```

## Benefits

1. **Transparency**: Clinicians can see exactly why the AI made its diagnosis
2. **Verification**: Predicates are validated against actual patient data
3. **Trust**: Explainable reasoning builds confidence in AI recommendations
4. **Clinical Value**: Identifies gaps in patient data and suggests next steps
5. **Quality Assessment**: Automatically evaluates reasoning quality
6. **Error Detection**: Highlights contradicting evidence and inconsistencies

## AI Models Used

- **Primary**: Google Gemini 1.5 Flash (fast, reliable)
- **Fallback**: Groq Llama 3.3 70B (high-quality reasoning)
- **Tertiary**: Rule-based extraction (always available)

## Performance

- **Average Processing Time**: 2-5 seconds
- **Predicate Extraction**: 5-15 predicates per diagnosis
- **Verification Success Rate**: Typically 60-90%
- **Confidence Calculation**: < 1 second

## Configuration

Environment variables (`.env`):
```bash
# AI Services
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key

# FOL Configuration
FOL_CACHE_TTL=1800
FOL_MAX_PREDICATES=50
FOL_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
VERBOSE_LOGS=1  # Enable detailed XAI logs
```

## Future Enhancements

1. **Knowledge Graph Integration**: Link predicates to medical ontologies
2. **Multi-Modal XAI**: Explain image-based diagnoses with visual reasoning
3. **Comparative XAI**: Compare multiple diagnostic hypotheses
4. **Interactive XAI**: Allow clinicians to query specific reasoning steps
5. **Learning from Feedback**: Improve XAI based on clinician corrections

## Troubleshooting

### XAI Reasoning Generation Fails

**Symptom**: No XAI explanation generated
**Solution**: 
- Check API keys for Gemini/Groq
- System falls back to rule-based reasoning
- Review logs for specific error messages

### Low Verification Success Rate

**Symptom**: < 40% predicates verified
**Solution**:
- Check patient data completeness
- Ensure symptoms and history are properly documented
- Review clinical context for clarity

### Poor Reasoning Quality

**Symptom**: Reasoning quality marked as "POOR"
**Solution**:
- Provide more detailed patient information
- Include relevant medical history
- Add vital signs and lab results
- Use structured clinical notes

## References

- First-Order Logic in Medical AI: [Research Paper Link]
- Explainable AI in Healthcare: [Review Article]
- CortexMD Documentation: [Main README]

---

**Last Updated**: 2025-01-01
**Version**: 2.0 (XAI-Enhanced)
**Maintainer**: CortexMD Development Team
