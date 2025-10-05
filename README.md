# <div align="center">🧠 CortexMD</div>
 ## Submissions
<div align="center">
 
[![Samsung PRISM](https://img.shields.io/badge/Samsung-PRISM%20GenAI%20Hackathon-1428A0?style=for-the-badge&logo=samsung&logoColor=white)](https://cortexmd-samsung.vercel.app/)
[![Theme](https://img.shields.io/badge/Theme-Multimodal%20AI-FF6B6B?style=for-the-badge&logo=ai&logoColor=white)](https://cortexmd-samsung.vercel.app/)
[![Live Demo](https://img.shields.io/badge/🚀-Live%20Demo-00D9FF?style=for-the-badge)](https://cortexmd-samsung.vercel.app/)
[![License](https://img.shields.io/badge/License-Apache_License-green.svg?style=for-the-badge)](LICENSE)

**An Intelligent Clinical AI Workspace with Explainable Multimodal Diagnosis**

[📽️ Video Demo](https://drive.google.com/drive/u/0/folders/1PURn0ijHWcBH2pD8YvwHZ4m7rbngFbYW) •[🌐 Live Demo](https://cortexmd-samsung.vercel.app/) • [📄 Project Report](#-project-report) • [🚀 Quick Start](#-quick-start) • [📖 Documentation](./Windows12Devs_CortexMD_Pitch.pdf)

</div>

---

## 🎯 Overview

<div align="center">

```mermaid
graph TB
    subgraph "Input Modalities"
        A[📝 Text Input] 
        B[🖼️ Medical Imaging]
        C[🎥 Video Analysis]
        D[🎤 Voice Notes]
        E[📊 DICOM Files]
    end
    
    subgraph "CortexMD Core Engine"
        F[🧠 AI Diagnosis Pipeline]
        G[🔍 XAI Verification]
        H[📚 Knowledge Graph]
        I[⚡ Real-time EWS]
    end
    
    subgraph "Intelligent Outputs"
        J[📋 Structured Diagnosis]
        K[💡 Clinical Reasoning]
        L[⚠️ Risk Assessment]
        M[📈 Evidence Sources]
    end
    
    A --> F
    B --> F
    C --> F
    D --> F
    E --> F
    
    F --> G
    F --> H
    F --> I
    
    G --> J
    H --> K
    I --> L
    G --> M
    
    style F fill:#667eea,stroke:#764ba2,stroke-width:3px,color:#fff
    style G fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    style I fill:#4facfe,stroke:#00f2fe,stroke-width:2px,color:#fff
```

</div>

### ✨ Key Highlights

<table>
<tr>
<td width="50%" valign="top">

#### 🎭 **Multimodal Intelligence**
- 📝 Text symptom analysis
- 🖼️ Medical imaging (X-ray, MRI, CT)
- 🎥 Video diagnostic clips
- 🎤 Voice-to-text intake
- 📊 DICOM file support

</td>
<td width="50%" valign="top">

#### 🔬 **Explainable AI (XAI)**
- 🧩 First-Order Logic verification
- 📚 Knowledge graph reasoning
- 🌐 Online medical verification
- 📊 Confidence scoring
- 🔍 Source attribution

</td>
</tr>
<tr>
<td width="50%" valign="top">

#### ⚡ **Real-Time Systems**
- 🚨 CONCERN Early Warning System
- 📡 WebSocket status updates
- 📈 Live severity tracking
- 🔄 Persistent risk monitoring
- ⏱️ Performance optimization

</td>
<td width="50%" valign="top">

#### 🏥 **Clinical-Ready Features**
- 🏷️ UMLS code lookup
- 🧬 Ontology mapping
- 🖥️ NVIDIA Clara integration
- 💾 Session persistence
- 📱 Responsive UI

</td>
</tr>
</table>

---

## 📚 Project Report

<div align="center">

### 📄 **Comprehensive Technical Documentation**

[![PDF Preview](https://img.shields.io/badge/📖-Read%20Full%20Report-FF6B6B?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](./CortexMD_ProjectReport.pdf)

**[Download CortexMD Project Report PDF →](./Windows12Devs_CortexMD_ProjectReport.pdf)**

*Detailed architecture, methodologies, results, and future roadmap*

</div>

---

## 🚀 Quick Start

### 🪟 **Windows (PowerShell)**

```powershell
# Clone and navigate
git clone <repository-url>
Set-Location .\cortexmd

# 🔥 Development Mode (Hot Reload)
make dev

# 🚀 Production Mode
make prod

# 🌐 Open in Browser
start https://localhost:3000
```

### 🍎 **macOS / 🐧 Linux (Bash)**

```bash
# Clone and navigate
git clone <repository-url>
cd cortexmd

# 🔥 Development Mode (Hot Reload)
make dev

# 🚀 Production Mode
make prod

# 🌐 Open in Browser
open https://localhost:3000 2>/dev/null || xdg-open https://localhost:3000
```

> 💡 **No Make?** See [Docker Commands](#option-c-standard-compose-balanced-defaults) below.

---

## 📋 Prerequisites

<table>
<tr>
<td align="center" width="33%">

### 🐳 **Docker**
Docker Desktop 4.x+  
or  
Docker Engine + Compose

</td>
<td align="center" width="33%">

### 🐍 **Python** (Manual Setup)
Python 3.10+  
pip package manager

</td>
<td align="center" width="33%">

### 📦 **Node.js** (Manual Setup)
Node.js 18+  
npm package manager

</td>
</tr>
</table>

---

## 🏗️ Architecture Deep Dive

### 🌐 System Architecture

<div align="center">

```mermaid
flowchart LR
    subgraph Client["🖥️ Client Layer"]
        UI[Next.js Frontend<br/>Port 3000]
    end
    
    subgraph API["🔌 API Gateway"]
        Flask[Flask Backend<br/>Port 5000]
        WS[WebSocket Server]
    end
    
    subgraph Services["⚙️ Core Services"]
        Diag[Diagnosis Pipeline]
        Ver[Verification Stack]
        EWS[CONCERN EWS]
        UMLS[UMLS Lookup]
    end
    
    subgraph AI["🤖 AI/ML Layer"]
        LLM[LLM Services<br/>Groq/Gemini]
        Clara[NVIDIA Clara]
        STT[Audio STT]
    end
    
    subgraph Data["💾 Data Layer"]
        PG[(PostgreSQL<br/>Port 5432)]
        Neo[(Neo4j<br/>Ports 7475/7688)]
        Redis[(Redis<br/>Port 6379)]
    end
    
    UI <-->|REST/WS| Flask
    UI <-->|Real-time| WS
    
    Flask --> Diag
    Flask --> Ver
    Flask --> EWS
    Flask --> UMLS
    
    Diag --> LLM
    Diag --> Clara
    Diag --> STT
    
    Flask --> PG
    Flask --> Neo
    Flask --> Redis
    
    style UI fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
    style Flask fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    style LLM fill:#4facfe,stroke:#00f2fe,stroke-width:2px,color:#fff
    style PG fill:#fa709a,stroke:#fee140,stroke-width:2px,color:#fff
```

</div>

### 🔄 Diagnosis Pipeline Flow

<div align="center">

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant Pipeline
    participant Verification
    participant EWS
    participant DB

    User->>Frontend: Submit Diagnosis
    Frontend->>API: POST /api/diagnose
    
    API->>Pipeline: Initialize Session
    Pipeline->>Pipeline: Process Modalities
    Note over Pipeline: Text, Image, Video, Audio
    
    Pipeline->>Verification: Run XAI Stack
    Verification->>Verification: FOL Verification
    Verification->>Verification: Knowledge Graph
    Verification->>Verification: Online Validation
    
    Verification->>EWS: Calculate Risk
    EWS->>DB: Persist Severity
    
    Pipeline->>API: Build UI Payload
    API->>Frontend: Return Results
    Frontend->>User: Display Analysis
    
    loop Real-time Updates
        API-->>Frontend: WebSocket Status
        Frontend-->>User: Progress Updates
    end
```

</div>

### 🧩 Component Architecture

```mermaid
graph TB
    subgraph Frontend["🎨 Frontend (Next.js)"]
        UI1[Diagnosis Form]
        UI2[Results Dashboard]
        UI3[Patient Monitoring]
        UI4[File Uploads]
    end
    
    subgraph Backend["⚙️ Backend (Flask)"]
        BP1[AR Blueprint]
        BP2[Optimized API v2]
        BP3[CONCERN Routes]
        BP4[UMLS Service]
    end
    
    subgraph Models["🤖 AI Models"]
        M1[Diagnosis Model]
        M2[Verification Model]
        M3[Risk Classifier]
        M4[STT Model]
    end
    
    subgraph Storage["💾 Storage"]
        S1[PostgreSQL]
        S2[Redis Cache]
        S3[Neo4j Graph]
        S4[File System]
    end
    
    UI1 --> BP1
    UI2 --> BP2
    UI3 --> BP3
    UI4 --> BP1
    
    BP1 --> M1
    BP1 --> M2
    BP3 --> M3
    BP1 --> M4
    
    M1 --> S1
    M2 --> S2
    M3 --> S1
    BP1 --> S4
    BP2 --> S2
    BP4 --> S3
    
    style Frontend fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
    style Backend fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    style Models fill:#4facfe,stroke:#00f2fe,stroke-width:2px,color:#fff
    style Storage fill:#fa709a,stroke:#fee140,stroke-width:2px,color:#fff
```

---

## 🐳 Docker Deployment

### Option A: Development (Hot Reload)

```bash
# Start development environment
make dev

# Or manually
docker compose -f docker-compose.dev.yml up --build -d
```

<details>
<summary><b>📍 Service Endpoints</b></summary>

| Service | URL | Credentials |
|---------|-----|-------------|
| 🎨 Frontend | http://localhost:3000 | N/A |
| ⚙️ Backend | http://localhost:5000 | N/A |
| 🗄️ PostgreSQL | localhost:5432 | `cortexmd` / `cortexmd123` |
| 🌐 Neo4j Browser | http://localhost:7475 | `neo4j` / `12345678` |
| ⚡ Neo4j Bolt | bolt://localhost:7688 | `neo4j` / `12345678` |
| 🔴 Redis | localhost:6379 | No password |

</details>

**Management Commands:**

```bash
# View status
make status

# Follow logs
make logs

# Stop all services
make down

# Restart services
make restart
```

### Option B: Production Mode

```bash
# Start production environment
make prod

# Or manually
docker compose -f docker-compose.prod.yml up --build -d
```

<details>
<summary><b>🔒 SSL Configuration</b></summary>

Production mode requires SSL certificates:

**Backend:** `backend/ssl_certs/`
- `cert.pem`
- `key.pem`

**Frontend:** `frontend/ssl_certs/`
- `cert.pem`
- `key.pem`

Generate self-signed certs (development):
```bash
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout key.pem -out cert.pem -days 365
```

</details>

### Option C: Standard Compose (Balanced Defaults)

```bash
docker compose up --build -d
```

### 🛠️ Useful Docker Commands

```bash
# View service status
docker compose ps

# Follow logs (all services)
docker compose logs -f

# Follow logs (specific service)
docker compose logs -f backend

# Rebuild and restart
docker compose up --build -d

# Stop all services
docker compose down

# Remove volumes (clean slate)
docker compose down -v

# Execute commands in container
docker compose exec backend bash
docker compose exec frontend sh
```

---

## 💻 Manual Development Setup

Perfect for local debugging with hot reload!

### 🐍 Backend Setup (Flask)

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Create `.env` file:**

<details>
<summary><b>Windows PowerShell</b></summary>

```powershell
@"
DATABASE_URL=postgresql://cortexmd:cortexmd123@localhost:5432/cortexmd
REDIS_URL=redis://localhost:6379/0
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_AI_API_KEY=your_google_ai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
UMLS_API_KEY=your_umls_api_key_here
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
SPEED_MODE=0
VERBOSE_LOGS=1
"@ | Out-File -Encoding ascii .env
```

</details>

<details>
<summary><b>macOS/Linux</b></summary>

```bash
cat > .env << 'EOF'
DATABASE_URL=postgresql://cortexmd:cortexmd123@localhost:5432/cortexmd
REDIS_URL=redis://localhost:6379/0
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_AI_API_KEY=your_google_ai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
UMLS_API_KEY=your_umls_api_key_here
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
SPEED_MODE=0
VERBOSE_LOGS=1
EOF
```

</details>

**Start backend server:**

```bash
# HTTPS (if certs exist)
python app.py

# Force HTTP only
python run_http_server.py
```

### 📦 Frontend Setup (Next.js)

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install
```

**Create `.env.local` file:**

<details>
<summary><b>Windows PowerShell</b></summary>

```powershell
@"
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_WS_URL=ws://localhost:5000
NODE_ENV=development
"@ | Out-File -Encoding ascii .env.local
```

</details>

<details>
<summary><b>macOS/Linux</b></summary>

```bash
cat > .env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_WS_URL=ws://localhost:5000
NODE_ENV=development
EOF
```

</details>

**Start development server:**

```bash
# HTTP mode
npm run dev

# HTTPS mode (requires certs)
npm run dev-https
```

---

## 🔌 API Reference

### 🏥 Health & Status

```http
GET /api/health
```

<details>
<summary><b>Response</b></summary>

```json
{
  "status": "healthy",
  "timestamp": "2025-10-05T10:30:00Z",
  "features": {
    "umls_enabled": true,
    "clara_enabled": true,
    "neo4j_enabled": true
  },
  "dependencies": {
    "database": "connected",
    "redis": "connected",
    "neo4j": "connected"
  }
}
```

</details>

### 🩺 Diagnosis Endpoints

#### Submit Diagnosis

```http
POST /api/diagnose
Content-Type: application/json

{
  "patient_id": "PATIENT_001",
  "symptoms": ["fever", "cough", "fatigue"],
  "vitals": {
    "temperature": 38.5,
    "heart_rate": 96,
    "blood_pressure": "120/80",
    "spo2": 97
  },
  "history": "No known chronic conditions",
  "medications": ["Ibuprofen"],
  "allergies": []
}
```

<details>
<summary><b>Response</b></summary>

```json
{
  "session_id": "e0d9a3b7-4f2e-4c1a-8d6e-9f5b2c1a7e3d",
  "status": "processing",
  "message": "Diagnosis pipeline initiated"
}
```

</details>

#### Check Status

```http
GET /api/status/{session_id}
```

<details>
<summary><b>Response</b></summary>

```json
{
  "session_id": "e0d9a3b7-4f2e-4c1a-8d6e-9f5b2c1a7e3d",
  "status": "completed",
  "progress": 100,
  "current_step": "verification_complete",
  "logs": [
    "Initialized diagnosis pipeline",
    "Processing patient data",
    "Running AI analysis",
    "Verifying results",
    "Complete"
  ]
}
```

</details>

#### Get Results

```http
GET /api/results/{session_id}
```

<details>
<summary><b>Response Structure</b></summary>

```json
{
  "diagnosis": {
    "primary": "Upper Respiratory Tract Infection",
    "differential": ["Common Cold", "Influenza", "COVID-19"],
    "recommendations": ["Rest", "Hydration", "Monitor symptoms"]
  },
  "explanations": [
    "Patient presents with classic URI symptoms...",
    "Vital signs indicate mild fever response...",
    "No red flags for serious conditions..."
  ],
  "confidenceScores": {
    "overall": 0.87,
    "fol_verification": 0.92,
    "enhanced_verification": 0.85,
    "online_verification": 0.84
  },
  "verificationStatus": {
    "fol": "verified",
    "enhanced": "verified",
    "online": "verified"
  },
  "sources": {
    "textbooks": ["Harrison's Internal Medicine"],
    "online": ["PubMed Article #12345", "CDC Guidelines"]
  },
  "risk_assessment": {
    "level": "low",
    "score": 0.23,
    "recommendations": ["Routine follow-up"]
  }
}
```

</details>

### 📁 File Upload

```http
POST /api/upload
Content-Type: multipart/form-data

file: [binary]
patient_id: PATIENT_001
file_type: image
```

<details>
<summary><b>Supported File Types</b></summary>

| Type | Extensions | Max Size |
|------|-----------|----------|
| 🖼️ Images | `.jpg`, `.jpeg`, `.png`, `.tiff`, `.dcm` | 50 MB |
| 🎥 Videos | `.mp4`, `.avi`, `.mov` | 500 MB |
| 🎤 Audio | `.wav`, `.mp3`, `.ogg`, `.m4a`, `.webm` | 25 MB |

</details>

### ⚠️ CONCERN Early Warning System

#### Current Severity

```http
GET /api/concern/current?patient_id=PATIENT_001
```

<details>
<summary><b>Response</b></summary>

```json
{
  "patient_id": "PATIENT_001",
  "current_severity": "medium",
  "risk_score": 0.65,
  "confidence": 0.88,
  "trend": "stable",
  "last_updated": "2025-10-05T10:30:00Z",
  "recommendations": [
    "Continue monitoring",
    "Schedule follow-up in 48 hours"
  ]
}
```

</details>

#### Severity History

```http
GET /api/concern/history?patient_id=PATIENT_001&days=7
```

#### Risk Trends

```http
GET /api/concern/trends?patient_id=PATIENT_001
```

### 🏷️ UMLS Lookup

```http
GET /api/umls/search?query=fever&type=term
```

### 🔄 Optimized API v2

```http
GET /api/v2/diagnose/fast
POST /api/v2/notes/quick
```

---

## 🎨 Multimodal Features

### 📝 Text Modality

<table>
<tr>
<td width="50%">

**Input Processing**
- Structured symptom intake
- Clinical history parsing
- Medication analysis
- Vital signs interpretation

</td>
<td width="50%">

**AI Analysis**
- LLM-backed diagnosis
- Differential generation
- Clinical reasoning
- Recommendation synthesis

</td>
</tr>
</table>

### 🖼️ Imaging Modality

```mermaid
graph LR
    A[Upload Image] --> B[Format Detection]
    B --> C{Image Type}
    C -->|X-ray| D[Radiological Analysis]
    C -->|MRI| E[Tissue Analysis]
    C -->|CT| F[3D Reconstruction]
    C -->|DICOM| G[Medical Standard]
    
    D --> H[NVIDIA Clara]
    E --> H
    F --> H
    G --> H
    
    H --> I[Heatmap Generation]
    I --> J[Clinical Integration]
    
    style H fill:#76b900,stroke:#5a8800,stroke-width:2px,color:#fff
```

**Supported Formats:**
- 🔹 JPEG/PNG/TIFF
- 🔹 DICOM medical imaging
- 🔹 Multi-frame images
- 🔹 Heatmap overlays

### 🎥 Video Modality

**Processing Pipeline:**
1. 📤 Large file upload (up to 500MB)
2. 🎞️ Frame extraction
3. 🔍 Temporal analysis
4. 📊 Motion detection
5. 🏥 Clinical relevance scoring

**Use Cases:**
- Ultrasound clips
- Endoscopy recordings
- Patient movement analysis
- Surgical procedure videos

### 🎤 Audio Modality (STT)

```mermaid
graph LR
    A[🎤 Audio Input] --> B[Format Validation]
    B --> C[Speech-to-Text]
    C --> D[Transcript Cleanup]
    D --> E[Clinical NLP]
    E --> F[Symptom Extraction]
    F --> G[Integration]
    
    style C fill:#fa709a,stroke:#fee140,stroke-width:2px,color:#fff
```

**Supported Formats:**
- 🔸 WAV (uncompressed)
- 🔸 MP3 (compressed)
- 🔸 OGG/M4A/WEBM

### 📚 Knowledge Modality

<table>
<tr>
<td align="center" width="33%">

**🏷️ UMLS Integration**

Code validation  
Terminology normalization  
Concept mapping

</td>
<td align="center" width="33%">

**🕸️ Knowledge Graph**

Neo4j relationships  
Ontology traversal  
Evidence expansion

</td>
<td align="center" width="33%">

**🔍 Online Verification**

PubMed search  
Clinical guidelines  
Recent research

</td>
</tr>
</table>

---

## 🔬 Explainable AI (XAI) Stack

### 🧩 First-Order Logic (FOL) Verification

```
∀ patient: hasSymptom(patient, fever) ∧ hasSymptom(patient, cough) 
→ possibleDiagnosis(patient, URI)
```

**Confidence Scoring:**
- Logical consistency check
- Predicate validation
- Rule application tracing

### 📚 Enhanced Knowledge Verification

```mermaid
graph TB
    A[Diagnosis Result] --> B{Knowledge Sources}
    B --> C[Medical Textbooks]
    B --> D[Knowledge Graph]
    B --> E[Clinical Guidelines]
    
    C --> F[Cross-Reference]
    D --> F
    E --> F
    
    F --> G{Confidence Level}
    G -->|High| H[✅ Verified]
    G -->|Medium| I[⚠️ Needs Review]
    G -->|Low| J[❌ Uncertain]
    
    style H fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    style I fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
    style J fill:#ef4444,stroke:#dc2626,stroke-width:2px,color:#fff
```

### 🌐 Online Medical Verification

**Real-time Validation:**
1. Query medical databases
2. Search recent publications
3. Check clinical guidelines
4. Aggregate evidence
5. Calculate confidence

**Sources:**
- PubMed Central
- CDC Guidelines
- WHO Resources
- Clinical Trials Database

### 📊 Confidence Aggregation

```python
overall_confidence = (
    fol_score * 0.35 +
    enhanced_score * 0.35 +
    online_score * 0.30
)
```

---

## ⚡ CONCERN Early Warning System

### 🚨 Real-time Risk Monitoring

```mermaid
graph TD
    A[Patient Data] --> B[Risk Calculation]
    B --> C{Risk Level}
    
    C -->|0-0.25| D[🟢 Low Risk]
    C -->|0.26-0.50| E[🟡 Medium Risk]
    C -->|0.51-0.75| F[🟠 High Risk]
    C -->|0.76-1.00| G[🔴 Critical Risk]
    
    D --> H[Routine Monitoring]
    E --> I[Enhanced Surveillance]
    F --> J[Immediate Review]
    G --> K[Emergency Protocol]
    
    B --> L[(PostgreSQL)]
    B --> M[(Redis Cache)]
    
    L --> N[Historical Trends]
    M --> O[Real-time Updates]
    
    style G fill:#ef4444,stroke:#dc2626,stroke-width:3px,color:#fff
    style F fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
    style E fill:#eab308,stroke:#ca8a04,stroke-width:2px,color:#fff
    style D fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
```

### 📈 Persistent Severity Tracking

**Database Schema:**
```sql
CREATE TABLE concern_severity_tracking (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(255) NOT NULL,
    severity_level VARCHAR(50),
    risk_score DECIMAL(5,4),
    confidence DECIMAL(5,4),
    recommendations TEXT[],
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(255),
    
    INDEX idx_patient_timestamp (patient_id, timestamp),
    INDEX idx_severity (severity_level),
    INDEX idx_risk_score (risk_score)
);
```

### 📊 Trend Analysis

- 📅 Historical severity progression
- 📈 Risk score trajectory
- ⚠️ Alert threshold breaches
- 🔄 Pattern recognition
- 📉 Improvement tracking

---

## 🛠️ Makefile Commands

<table>
<tr>
<th>Command</th>
<th>Description</th>
</tr>
<tr>
<td><code>make dev</code></td>
<td>🔥 Start development environment with hot reload</td>
</tr>
<tr>
<td><code>make prod</code></td>
<td>🚀 Start production environment with SSL</td>
</tr>
<tr>
<td><code>make up</code></td>
<td>⬆️ Start services (default compose file)</td>
</tr>
<tr>
<td><code>make down</code></td>
<td>⬇️ Stop all services</td>
</tr>
<tr>
<td><code>make restart</code></td>
<td>🔄 Restart all services</td>
</tr>
<tr>
<td><code>make logs</code></td>
<td>📋 Follow logs from all services</td>
</tr>
<tr>
<td><code>make status</code></td>
<td>📊 Show service status</td>
</tr>
<tr>
<td><code>make install</code></td>
<td>📦 Install dependencies (backend + frontend)</td>
</tr>
<tr>
<td><code>make test</code></td>
<td>🧪 Run test suites</td>
</tr>
<tr>
<td><code>make backup</code></td>
<td>💾 Backup databases</td>
</tr>
<tr>
<td><code>make restore</code></td>
<td>♻️ Restore from backup</td>
</tr>
<tr>
<td><code>make clean</code></td>
<td>🧹 Clean up containers and volumes</td>
</tr>
<tr>
<td><code>make reset</code></td>
<td>🔥 Nuclear option - remove everything






