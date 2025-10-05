# XAI System Quick Start Guide

## Overview

The XAI (Explainable AI) System in CortexMD provides transparent, verifiable medical reasoning. This guide shows you how to use it.

## Prerequisites

1. **API Keys** (add to `.env` file):
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

2. **Python Dependencies** (already in `requirements.txt`):
- google-generativeai
- groq
- asyncio (built-in)

## How It Works (Simple Explanation)

```
Your Diagnosis Request
        ↓
AI Generates Diagnosis
        ↓
XAI System Activates:
  1. LLM verifies diagnosis vs your input
  2. Extracts medical logic (FOL predicates)
  3. Verifies logic against patient data
  4. Generates transparent explanation
        ↓
You get explainable results!
```

## Usage in Your Backend

### Automatic Integration (Already Done!)

The XAI system is **automatically activated** for every diagnosis in the main `/diagnose` endpoint. No extra code needed!

When you call:
```bash
POST /diagnose
{
  "clinical_text": "Patient has chest pain and mass",
  "patient_id": "PT-001"
}
```

You automatically get XAI results in the response:
```json
{
  "session_id": "...",
  "diagnosis": {
    "primary_diagnosis": "Myxofibrosarcoma",
    "confidence_score": 0.85
  },
  "xai_reasoning": {
    "xai_explanation": "The diagnosis is supported by...",
    "supporting_evidence": [...],
    "confidence_level": "HIGH",
    "reasoning_quality": "EXCELLENT"
  },
  "fol_verification": {
    "total_predicates": 10,
    "verified_predicates": 8,
    "success_rate": 0.80
  }
}
```

### Manual Usage (Advanced)

If you want to use XAI independently:

```python
from services.xai_reasoning_engine import XAIReasoningEngine

# Initialize
xai_engine = XAIReasoningEngine()

# Generate XAI reasoning
xai_result = await xai_engine.generate_xai_reasoning(
    diagnosis="Myxofibrosarcoma",
    patient_data={
        'symptoms': ['chest pain', 'mass'],
        'medical_history': ['surgery'],
        'chief_complaint': 'Patient reports painful chest mass'
    },
    clinical_context="Patient presents with chest mass...",
    reasoning_paths=["Clinical exam shows mass", "Imaging consistent with sarcoma"],
    patient_id="PT-001"
)

# Access results
print(f"XAI Explanation: {xai_result.xai_explanation}")
print(f"Confidence: {xai_result.confidence_level}")
print(f"Quality: {xai_result.reasoning_quality}")
print(f"Recommendations: {xai_result.clinical_recommendations}")
```

## Testing the System

### Quick Test

Run the test suite:
```bash
cd backend
python tests/test_xai_system.py
```

Expected output:
```
🚀 Starting XAI System Tests...

📋 Step 1: Initializing XAI Reasoning Engine...
✅ XAI Engine initialized successfully

🧠 Step 3: Generating XAI Reasoning...
✅ XAI Reasoning Generated Successfully!

📊 CONFIDENCE METRICS:
   Confidence Level: HIGH
   Confidence Score: 85.00%
   Reasoning Quality: EXCELLENT

✅ TEST COMPLETED SUCCESSFULLY!
```

### Test with curl

Start the backend:
```bash
cd backend
python app.py
```

Send a test request:
```bash
curl -X POST http://localhost:5000/diagnose \
  -F "clinical_text=Patient has chest pain and mass" \
  -F "patient_id=TEST-001"
```

Check results:
```bash
curl http://localhost:5000/status/{session_id}
```

## Understanding the Results

### XAI Explanation
Human-readable explanation of the diagnosis reasoning:
```
"The diagnosis of Myxofibrosarcoma is supported by patient's 
reported symptoms including chest pain and palpable mass. 
Clinical findings show high consistency with patient data 
(8/10 predicates verified, 80% success rate). Confidence: HIGH."
```

### Supporting Evidence
List of facts that support the diagnosis:
```
[
  "Has Symptom: chest pain - Found in symptoms",
  "Has Symptom: mass - Found in symptoms",
  "Has Condition: myxofibrosarcoma - Matches diagnosis",
  "Patient reported 3 relevant symptoms"
]
```

### FOL Predicates
Structured logic statements:
```
has_symptom(patient, chest_pain) ✅ VERIFIED
has_symptom(patient, mass) ✅ VERIFIED
has_condition(patient, myxofibrosarcoma) ✅ VERIFIED
```

### Confidence Level
- **HIGH** (80-100%): Strong evidence, well-verified
- **MEDIUM** (60-80%): Moderate evidence, mostly verified
- **LOW** (<60%): Weak evidence, limited verification

### Reasoning Quality
- **EXCELLENT**: Comprehensive, well-supported reasoning
- **GOOD**: Solid reasoning with good evidence
- **FAIR**: Acceptable but could be improved
- **POOR**: Insufficient evidence or weak reasoning

## Troubleshooting

### Issue: No XAI results in response

**Solution:**
1. Check API keys in `.env` file
2. Verify internet connection (for Gemini/Groq)
3. Check backend logs for errors
4. System should fallback to basic FOL if AI unavailable

### Issue: Low verification success rate (<40%)

**Solution:**
1. Provide more detailed patient information
2. Include relevant symptoms and history
3. Use structured clinical notes
4. Add vital signs if available

### Issue: "Error in XAI reasoning generation"

**Solution:**
1. Check API key validity
2. Verify API quota not exceeded
3. Check logs for specific error:
   ```bash
   tail -f backend/logs/cortexmd.log
   ```
4. System will use fallback reasoning

## Checking Logs

Enable verbose logging in `.env`:
```bash
VERBOSE_LOGS=1
LOG_LEVEL=INFO
```

View XAI logs:
```bash
# In backend directory
tail -f logs/cortexmd.log | grep "XAI"
```

Look for:
- `🧠 Starting XAI-ENHANCED FOL verification`
- `✅ XAI reasoning generated using Gemini`
- `🔬 Verified predicates: X/Y`

## Configuration Options

In `.env` file:

```bash
# Required API Keys
GOOGLE_API_KEY=your_key
GROQ_API_KEY=your_key

# FOL Configuration (optional)
FOL_CACHE_TTL=1800           # Cache TTL in seconds
FOL_MAX_PREDICATES=50        # Max predicates to extract
FOL_TIMEOUT=30               # Timeout in seconds

# Logging (optional)
LOG_LEVEL=INFO               # DEBUG, INFO, WARNING, ERROR
VERBOSE_LOGS=1               # Enable detailed logs

# Performance (optional)
SPEED_MODE=0                 # 0 = Full XAI, 1 = Fast mode
```

## Best Practices

### 1. Provide Complete Patient Data
```python
patient_data = {
    'symptoms': ['chest pain', 'mass', 'swelling'],
    'medical_history': ['surgery', 'trauma'],
    'current_medications': ['pain meds'],
    'chief_complaint': 'Detailed complaint...',
    'vitals': {'bp': '130/85', 'hr': '72'},
    'lab_results': {...}
}
```

### 2. Use Structured Clinical Notes
Good ✅:
```
"Patient presents with chest pain and palpable mass. 
History of trauma 2 years ago. Mass has grown over 6 months.
Physical exam reveals 5cm firm, mobile mass."
```

Bad ❌:
```
"patient has pain"
```

### 3. Review XAI Recommendations
Always check:
- Supporting evidence
- Contradicting evidence
- Clinical recommendations
- Reasoning quality

### 4. Validate Critical Diagnoses
For serious conditions:
- Review FOL verification details
- Check predicate success rate
- Verify confidence level
- Follow clinical recommendations

## API Response Structure

Complete response structure:
```json
{
  "session_id": "abc123",
  "status": "completed",
  "progress": 100,
  
  "diagnosis": {
    "primary_diagnosis": "Myxofibrosarcoma",
    "confidence_score": 0.85,
    "reasoning_paths": [...]
  },
  
  "xai_reasoning": {
    "xai_explanation": "Detailed explanation...",
    "supporting_evidence": ["Evidence 1", "Evidence 2"],
    "contradicting_evidence": [],
    "confidence_level": "HIGH",
    "confidence_score": 0.85,
    "reasoning_quality": "EXCELLENT",
    "clinical_recommendations": ["Rec 1", "Rec 2"],
    "fol_predicates_count": 10,
    "timestamp": "2025-01-01T12:00:00"
  },
  
  "fol_verification": {
    "status": "COMPLETED_XAI_ENHANCED",
    "total_predicates": 10,
    "verified_predicates": 8,
    "success_rate": 0.80,
    "confidence_level": "HIGH",
    "clinical_assessment": "HIGHLY_CONSISTENT",
    "medical_reasoning_summary": "Summary...",
    "predicates": [
      {
        "fol_string": "has_symptom(patient, chest_pain)",
        "type": "has_symptom",
        "object": "chest pain",
        "verified": true,
        "confidence": 0.9,
        "evidence": ["Found in symptoms"]
      }
    ],
    "xai_reasoning": "XAI reasoning text...",
    "xai_enabled": true
  }
}
```

## Next Steps

1. **Test the system** with real patient data
2. **Integrate frontend** to display XAI results
3. **Collect feedback** from users
4. **Monitor performance** via logs
5. **Adjust prompts** if needed

## Support

- Documentation: `backend/docs/XAI_SYSTEM_DOCUMENTATION.md`
- Architecture: `backend/docs/XAI_ARCHITECTURE_DIAGRAMS.md`
- Tests: `backend/tests/test_xai_system.py`
- Issues: Check backend logs

## Summary

✅ **XAI is automatically enabled** for all diagnoses
✅ **No additional code needed** to use it
✅ **Results included** in standard API response
✅ **Transparent reasoning** for every diagnosis
✅ **Verification against patient data**
✅ **Clinical recommendations** included

**You're ready to use XAI!** 🚀
