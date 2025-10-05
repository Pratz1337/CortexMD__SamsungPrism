# XAI System Architecture and Flow

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CORTEXMD BACKEND                              │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    DIAGNOSIS ENGINE                         │   │
│  │  - Medical Image Analysis                                   │   │
│  │  - Clinical Text Processing                                 │   │
│  │  - FHIR Data Integration                                    │   │
│  │  → Outputs: Primary Diagnosis + Reasoning Paths            │   │
│  └──────────────────┬─────────────────────────────────────────┘   │
│                     │                                              │
│                     ▼                                              │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              XAI REASONING ENGINE (NEW!)                    │   │
│  │                                                              │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │ 1️⃣ LLM Reasoning Verification                        │  │   │
│  │  │    ├─ Gemini 1.5 Flash (Primary)                     │  │   │
│  │  │    ├─ Groq Llama 3.3 70B (Fallback)                  │  │   │
│  │  │    └─ Compares diagnosis with patient input          │  │   │
│  │  │    └─ Generates transparent reasoning                 │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                     │                                        │   │
│  │                     ▼                                        │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │ 2️⃣ FOL Predicate Extraction                          │  │   │
│  │  │    ├─ Extracts from XAI reasoning                    │  │   │
│  │  │    ├─ has_symptom(patient, X)                        │  │   │
│  │  │    ├─ has_condition(patient, Y)                      │  │   │
│  │  │    ├─ takes_medication(patient, Z)                   │  │   │
│  │  │    └─ has_vital_sign(patient, V, value)             │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                     │                                        │   │
│  │                     ▼                                        │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │ 3️⃣ Predicate Verification                            │  │   │
│  │  │    ├─ Validate against patient_data                  │  │   │
│  │  │    ├─ Fuzzy matching for medical terms               │  │   │
│  │  │    ├─ Track evidence for each predicate              │  │   │
│  │  │    └─ Calculate success rate                         │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                     │                                        │   │
│  │                     ▼                                        │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │ 4️⃣ Clinical Assessment                               │  │   │
│  │  │    ├─ Reasoning quality: EXCELLENT/GOOD/FAIR/POOR    │  │   │
│  │  │    ├─ Confidence level: HIGH/MEDIUM/LOW              │  │   │
│  │  │    ├─ Clinical recommendations                        │  │   │
│  │  │    └─ Supporting/contradicting evidence              │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                     │                                        │   │
│  └─────────────────────┼────────────────────────────────────────┘   │
│                        │                                            │
│                        ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                  XAI RESULTS STORAGE                        │   │
│  │  - diagnosis_sessions[session_id]['xai_reasoning']         │   │
│  │  - diagnosis_sessions[session_id]['fol_verification']      │   │
│  └────────────────────────────────────────────────────────────┘   │
│                        │                                            │
└────────────────────────┼────────────────────────────────────────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │  API Response │
                 │  to Frontend  │
                 └───────────────┘
```

## Detailed Process Flow

### Phase 1: Initialization
```
User submits diagnosis request
    ↓
Backend receives: clinical_text, images, FHIR data
    ↓
Session created with unique session_id
    ↓
Diagnosis engine processes inputs
    ↓
Primary diagnosis generated
```

### Phase 2: XAI Reasoning (NEW!)
```
┌─────────────────────────────────────────────┐
│  XAI Reasoning Engine Activated             │
└─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│  Step 1: Generate XAI Reasoning             │
│                                             │
│  Input:                                     │
│  - Diagnosis: "Myxofibrosarcoma"           │
│  - Patient Data:                            │
│    • symptoms: [chest pain, mass]          │
│    • history: [previous surgery]           │
│    • medications: [pain meds]              │
│                                             │
│  LLM Prompt:                                │
│  "Verify how this diagnosis relates to     │
│   patient's reported symptoms and input"    │
│                                             │
│  Output: Transparent reasoning text         │
│  "The diagnosis is supported by patient's   │
│   reported symptoms including chest pain    │
│   and palpable mass. Clinical findings      │
│   show consistency with patient data..."    │
└─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│  Step 2: Extract FOL Predicates             │
│                                             │
│  From XAI reasoning + original text:        │
│  - has_symptom(patient, chest_pain)        │
│  - has_symptom(patient, mass)              │
│  - has_condition(patient, myxofibrosarcoma)│
│  - takes_medication(patient, pain_meds)    │
│                                             │
│  Extraction method: AI (Gemini/Groq)       │
│  Fallback: Regex pattern matching           │
└─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│  Step 3: Verify Each Predicate              │
│                                             │
│  For has_symptom(patient, chest_pain):     │
│  ✓ Check patient_data['symptoms']          │
│  ✓ Found: 'chest pain'                     │
│  ✓ Verified: TRUE                          │
│  ✓ Confidence: 0.9                         │
│  ✓ Evidence: "Found in symptoms"           │
│                                             │
│  For has_symptom(patient, mass):           │
│  ✓ Check patient_data['symptoms']          │
│  ✓ Found: 'mass'                           │
│  ✓ Verified: TRUE                          │
│  ✓ Confidence: 0.85                        │
│                                             │
│  Success Rate: 8/10 = 80%                  │
└─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│  Step 4: Calculate Metrics                  │
│                                             │
│  Confidence Metrics:                        │
│  - symptom_match_score: 0.85               │
│  - evidence_quality_score: 0.80            │
│  - overall_confidence: 0.85                │
│  - uncertainty_score: 0.15                 │
│                                             │
│  Reasoning Quality:                         │
│  - Predicates verified: 8/10               │
│  - Success rate: 80%                       │
│  - Quality: EXCELLENT                      │
│                                             │
│  Clinical Assessment:                       │
│  - Assessment: HIGHLY_CONSISTENT           │
│  - Confidence Level: HIGH                  │
└─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│  Step 5: Generate Recommendations           │
│                                             │
│  Based on verification results:             │
│  1. ✅ XAI analysis shows high confidence  │
│  2. Clinical findings well-documented       │
│  3. Continue current management plan        │
└─────────────────────────────────────────────┘
```

### Phase 3: Result Storage & Response
```
┌─────────────────────────────────────────────┐
│  Store in Session                           │
│                                             │
│  diagnosis_sessions[session_id] = {         │
│    'xai_reasoning': {                       │
│      'xai_explanation': "...",             │
│      'supporting_evidence': [...],         │
│      'contradicting_evidence': [...],      │
│      'confidence_level': "HIGH",           │
│      'reasoning_quality': "EXCELLENT",     │
│      'clinical_recommendations': [...]     │
│    },                                       │
│    'fol_verification': {                    │
│      'total_predicates': 10,               │
│      'verified_predicates': 8,             │
│      'success_rate': 0.80,                 │
│      'predicates': [...],                  │
│      'xai_reasoning': "...",               │
│      'xai_enabled': true                   │
│    }                                        │
│  }                                          │
└─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│  Return to Frontend                         │
│                                             │
│  GET /status/{session_id}                  │
│                                             │
│  Response includes:                         │
│  - Primary diagnosis                        │
│  - XAI explanation                         │
│  - Supporting evidence                      │
│  - FOL verification results                │
│  - Clinical recommendations                │
└─────────────────────────────────────────────┘
```

## Component Interaction Diagram

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Gemini     │     │    Groq      │     │    Regex     │
│  1.5 Flash   │     │  Llama 3.3   │     │   Fallback   │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                     │
       └────────────────────┴─────────────────────┘
                            │
                  ┌─────────▼──────────┐
                  │  FOL Verification   │
                  │       V2            │
                  │                     │
                  │ - XAI Reasoning     │
                  │ - Predicate Extract │
                  │ - Verification      │
                  └─────────┬───────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
    ┌─────────▼──────────┐   ┌──────────▼────────┐
    │  Confidence Engine  │   │  XAI Reasoning    │
    │                     │   │     Engine        │
    │ - Hybrid AI        │   │                   │
    │ - XAI Confidence   │   │ - Orchestration   │
    │ - Metrics          │   │ - Integration     │
    └────────┬───────────┘   │ - Results         │
             │                └──────────┬────────┘
             │                           │
             └───────────┬───────────────┘
                         │
                ┌────────▼────────┐
                │   Main App      │
                │   (core/app.py) │
                │                 │
                │ - Diagnosis Flow│
                │ - XAI Integration
                │ - Session Mgmt  │
                └─────────────────┘
```

## Data Flow Example

**Input:**
```python
{
  "clinical_text": "Patient reports chest pain and palpable mass",
  "patient_id": "PT-12345",
  "symptoms": ["chest pain", "mass", "swelling"]
}
```

**Diagnosis Engine Output:**
```python
{
  "primary_diagnosis": "Myxofibrosarcoma",
  "confidence_score": 0.82,
  "reasoning_paths": [
    "Clinical exam reveals chest mass",
    "Imaging consistent with soft tissue sarcoma"
  ]
}
```

**XAI Processing:**

1️⃣ **LLM Reasoning:**
```
"The diagnosis of Myxofibrosarcoma is supported by:
- Patient's reported chest pain
- Palpable mass on examination
- Imaging findings consistent with sarcoma
Confidence: HIGH (85%)"
```

2️⃣ **FOL Predicates Extracted:**
```python
[
  has_symptom(patient, chest_pain),
  has_symptom(patient, mass),
  has_condition(patient, myxofibrosarcoma)
]
```

3️⃣ **Verification Results:**
```python
{
  "has_symptom(patient, chest_pain)": {
    "verified": True,
    "confidence": 0.9,
    "evidence": ["Found in symptoms: chest pain"]
  },
  "has_symptom(patient, mass)": {
    "verified": True,
    "confidence": 0.85,
    "evidence": ["Found in symptoms: mass"]
  }
}
```

**Final XAI Output:**
```python
{
  "xai_explanation": "Diagnosis verified with HIGH confidence...",
  "supporting_evidence": [
    "Chest pain verified in patient symptoms",
    "Mass confirmed on examination",
    "8/10 clinical predicates verified"
  ],
  "confidence_level": "HIGH",
  "reasoning_quality": "EXCELLENT",
  "clinical_recommendations": [
    "XAI analysis shows strong evidence base",
    "Continue with treatment plan"
  ]
}
```

---

**Legend:**
- 🧠 = AI/LLM Processing
- ✅ = Verification Step
- 📊 = Metrics Calculation
- 💡 = Recommendation Generation
- 🔬 = FOL Processing
