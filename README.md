# CortexMD

> This is our submission for the Samsung PRISM GenAI Hackathon — Theme: Multimodal AI.
### Our solution is fully Deployed with backend and you can experience it too at- https://cortexmd-samsung.vercel.app/
---
### Multimodality + XAI at the core
- Text, Imaging, and Video inputs with optional DICOM handling and file uploads
- Audio-to-text (STT) intake for voice notes integrated into diagnosis flows
- Real-time streams: status updates, severity trends, and websocket-powered EWS
- Knowledge-enabled reasoning: UMLS code lookup and ontology mapping
- Evidence-backed verification: FOL checks, enhanced knowledge verification, and online medical verification
- Explainable outputs: structured reasoning paths, confidence scores, verification status, and sources (designed for clinicians)
- Performance-aware: speed mode, caching, and patient prefetch for snappy UX

An end‑to‑end clinical AI workspace combining a Flask backend (ML, imaging, and knowledge services) with a Next.js frontend. This README focuses on getting you running quickly on Windows, macOS, and Linux using either Docker (recommended) or a manual dev setup.

### What this project does
- Dynamic AI diagnosis with explainability and verification
- Real‑time CONCERN Early Warning System (risk tracking) with persistence
- UMLS code lookup and ontology mapping
- Knowledge graph integrations (Neo4j optional) and enhanced verifications
- Imaging and video pipeline hooks and NVIDIA Clara integration (real)

### At a glance
- **Frontend (Next.js)**: port `3000`
- **Backend (Flask)**: port `5000`
- **PostgreSQL**: port `5432` (db: `cortexmd`, user: `cortexmd`)
- **Neo4j Browser**: port `7475` (user: `neo4j`)
- **Neo4j Bolt**: port `7688`
- **Redis**: port `6379`


## 📚 Project Report

For detailed Technical Project Report, please refer to:

**[📄 CortexMD ProjectReport PDF](./CortexMD_ProjectReport.pdf)**




## Quick start (the easiest way)

### Windows (PowerShell)
```bash
Set-Location .\\
# Development with hot reload
make dev

# Or production-like
make prod

# Open in browser
start https://localhost:3000
```

### macOS/Linux (Bash)
```bash
cd .
# Development with hot reload
make dev

# Or production-like
make prod

# Open in browser
open https://localhost:3000 2>/dev/null || xdg-open https://localhost:3000
```

If you do not have `make`, see the Docker commands below.


## Prerequisites
- Docker Desktop 4.x (Compose v2) or Docker Engine + Compose plugin
- For manual (non‑Docker) dev:
  - Python 3.10+ and `pip`
  - Node.js 18+ and `npm`


## Backend architecture overview

The Flask app lives in `backend/core/app.py` and wires together services, models, and blueprints:

- Initialization
  - Loads env via `dotenv` and configures logging and CORS
  - Initializes PostgreSQL via `get_database()` and ensures CONCERN severity tracking table exists
  - Optionally initializes NVIDIA Clara , patient cache, and optimized DB fallback
  - Registers blueprints:
    - `api_handlers.ar: ar_bp` (Augmented Reasoning/diagnosis routes)
    - `api_handlers.optimized_endpoints: optimized_bp` mounted at `/api/v2`
  - Real‑time CONCERN EWS engine and optional WebSocket server
  - UMLS code lookup service if `UMLS_API_KEY` is set

- Core concepts
  - Diagnosis sessions tracked in memory and/or DB, with progress logs and artifacts
  - Verification stack: FOL verification, enhanced evidence verification, online verification
  - Integrated LLM services via `utils.ai_key_manager` (Groq first, Gemini fallback)
  - Persistent CONCERN severity tracking table in Postgres with indices and history

- Notable endpoints (selection)
  - `GET /api/health`: health status, feature flags, basic dependency checks
  - `GET /api/images/<filename>`: serves images from `uploads`
  - AR blueprint (`/api/*`): diagnosis submission, status polling, results, explanations, predicate/FOL, chat, uploads
  - Optimized API v2 (`/api/v2/*`): streamlined versions of diagnosis/notes endpoints for faster UI
  - Real‑time CONCERN EWS (`/api/concern/*`): risk scoring, severity history, and trend data
  - UMLS lookup (`/umls-lookup`, API in service layer): code and concept search (enabled with `UMLS_API_KEY`)

Note: Many routes are implemented inside `api_handlers/` blueprints; `core/app.py` orchestrates and performs validation, session management, and finalization (including risk updates).


## Diagnosis pipeline (core/app.py)

High‑level processing in `run_comprehensive_diagnosis`:
1. Create/update diagnosis session; start processing logs
2. Initialize model pipeline via `ai_models.model_config_manager` with key management/load balancing
3. Execute primary diagnosis (LLM‑backed) and collect structured `DiagnosisResult`
4. Run verification layers:
   - FOL (first‑order logic) verification and medical reasoning summary
   - Enhanced evidence verification (textbook/knowledge graph)
   - Online verification (medical web search)
5. Build UI payload via `create_ui_data_structure` (explanations, confidence, verification status, sources)
6. Persist and emit results; update Redis caches for responsiveness
7. Finalize by classifying CONCERN risk using integrated LLM or rule‑based fallback; persist to Postgres and Redis

Performance controls: `SPEED_MODE=1` caps total pipeline time and trims heavy steps; verbose logs via `VERBOSE_LOGS=1`.


## CONCERN Early Warning System (EWS)
- Maintains persistent severity per patient with cumulative metrics and history (`concern_severity_tracking`)
- Automatically updated when a diagnosis completes (`_finalize_concern_after_diagnosis`)
- Risk levels: low, medium, high, critical, with confidence and recommendations
- Accessible via `/api/concern/*` for current status and trend charts used by the frontend


## Run with Docker (recommended)

### Option A: Development (hot reload)
```bash
# Equivalent to: docker compose -f docker-compose.dev.yml up --build -d
make dev
```
Services and defaults are defined in `docker-compose.dev.yml`:
- Frontend: http://localhost:3000 (DEV)
- Backend: http://localhost:5000 (DEV)
- PostgreSQL: localhost:5432 (db `cortexmd`, user `cortexmd`, pass `cortexmd123`)
- Neo4j Browser: http://localhost:7475 (user `neo4j`, pass `12345678`)
- Redis: localhost:6379

Stop and logs:
```bash
make status
make logs
make down
```

### Option B: Production-ish (single machine)
```bash
# Equivalent to: docker compose -f docker-compose.prod.yml up --build -d
make prod
```
Defaults in `docker-compose.prod.yml`:
- Frontend: https://localhost:3000
- Backend: https://localhost:5000
- SSL: self-signed certs expected in `backend/ssl_certs` and `frontend/ssl_certs`

Stop and logs:
```bash
make status
make logs
make down
```

### Option C: Standard compose (balanced defaults)
```bash
docker compose up --build -d
```

### Helpful Docker commands
```bash
# Status / logs
docker compose ps
docker compose logs -f

# Rebuild
docker compose up --build -d

# Stop all
docker compose down
```


## Manual development (without Docker)

Use this when you want local Python/Node processes with hot reload and easy debugging.

### 1) Backend (Flask)
```bash
cd backend
python -m venv .venv
# Windows
./.venv/Scripts/Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt

# Environment (create backend .env)
# Windows PowerShell
@"
DATABASE_URL=postgresql://cortexmd:cortexmd123@localhost:5432/cortexmd
REDIS_URL=redis://localhost:6379/0
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_AI_API_KEY=your_google_ai_api_key_here
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
"@ | Out-File -Encoding ascii .env

# Start server (HTTPS if certs exist, else HTTP)
python app.py
# Or force HTTP only
python run_http_server.py
```

Backend binds `0.0.0.0:5000`. If `backend/ssl_certs/cert.pem` and `key.pem` exist, it starts HTTPS automatically.

### 2) Frontend (Next.js)
```bash
cd frontend
npm install

# Environment (create .env.local)
# Windows PowerShell
@"
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_WS_URL=ws://localhost:5000
NODE_ENV=development
"@ | Out-File -Encoding ascii .env.local

npm run dev   # http://localhost:3000
```

If you prefer HTTPS locally for the frontend, add certs into `frontend/ssl_certs` and use:
```bash
npm run dev-https
```


## Ports and URLs
- Frontend (DEV): http://localhost:3000
- Frontend (PROD-ish): https://localhost:3000
- Backend (DEV): http://localhost:5000
- Backend (PROD-ish): https://localhost:5000
- Neo4j Browser: http://localhost:7475
- Neo4j Bolt: bolt://localhost:7688
- PostgreSQL: localhost:5432
- Redis: localhost:6379


## Environment variables

### Backend `.env`
```env
DATABASE_URL=postgresql://cortexmd:cortexmd123@localhost:5432/cortexmd
REDIS_URL=redis://localhost:6379/0
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_AI_API_KEY=your_google_ai_api_key_here
FLASK_ENV=production
SECRET_KEY=your_secret_key_here
```

### Frontend `.env.local`
```env
NEXT_PUBLIC_API_URL=https://localhost:5000
NEXT_PUBLIC_WS_URL=wss://localhost:5000
NODE_ENV=production
```

In `docker-compose.*.yml`, these are wired for you with sane defaults. Update for your environment/secrets.


## API quick reference (selected)

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | API health and feature flags |
| `/api/images/<filename>` | GET | Serve uploaded images |
| `/api/diagnose` | POST | Submit diagnosis (AR blueprint) |
| `/api/status/<session_id>` | GET | Poll diagnosis status |
| `/api/results/<session_id>` | GET | Fetch diagnosis results |
| `/api/concern/*` | GET/POST | Risk scoring, severity history, trends |
| `/api/v2/*` | mixed | Optimized endpoints for fast UI flows |

Exact request/response shapes are defined in `api_handlers/*` and `core/models.py`. The frontend calls into these via `NEXT_PUBLIC_API_URL`.


## SSL certificates
- Development: place self‑signed certs at `backend/ssl_certs/{cert.pem,key.pem}` and `frontend/ssl_certs/`.
- Production: replace with CA‑issued certs; ensure file mounts in `docker-compose.prod.yml` match your paths.
- To run backend without SSL locally, use `python backend/run_http_server.py`.


## Using the Makefile
Targets are wrappers around Compose and common workflows (see `Makefile`).
```bash
make dev       # docker-compose.dev.yml up --build -d
make prod      # docker-compose.prod.yml up --build -d
make up        # docker compose up -d (default file)
make down      # stop all
make logs      # follow logs
make status    # service status
make install   # pip/npm install (local)
make test      # run backend/frontend tests in containers
make backup    # DB backups
make restore   # DB restore
```


## Common tasks
- Change ports: edit `ports:` in the respective `docker-compose*.yml`.
- Switch API base URL used by frontend: set `NEXT_PUBLIC_API_URL` in `frontend/.env.local`.
- Persist uploads locally: files are mounted at `backend/uploads` when using Docker.


## Troubleshooting

### Ports already in use
Change the host ports in `docker-compose*.yml`, e.g.:
```yaml
services:
  frontend:
    ports:
      - "3001:3000"
```

### Backend won’t start (HTTP 500 or healthcheck fails)
- Check logs: `docker compose logs backend`
- Verify DB is healthy: `docker compose ps postgres`
- Confirm `DATABASE_URL` and DB connectivity
 - If running manually with HTTPS and certs are missing/corrupt, either provide valid certs in `backend/ssl_certs` or run `python backend/run_http_server.py` to force HTTP

### SSL errors locally
- Remove/regenerate dev certs in `backend/ssl_certs` and `frontend/ssl_certs`
- Use HTTP mode temporarily: `python backend/run_http_server.py`

### Out of memory / build fails
- Increase Docker Desktop memory to 8GB+ (especially for ML libs)


## Project layout (high level)
```
backend/           # Flask app, services, models, data pipelines
frontend/          # Next.js app (App Router), UI components
docker-compose*.yml
Makefile
```


## References & Scripts
- `HOW-TO-RUN.md` — one-click options and guided instructions
- `DOCKER_README.md` — deeper Docker deployment guide
- `start-dev.bat` / `start-dev.sh` — convenience starters
- `quick-start.ps1` — PowerShell bootstrap


## License
MIT (see `LICENSE` if present). Replace defaults and credentials before deploying to production.

---

## Multimodal feature deep dive

### Text modality
- Structured intake via the diagnosis form (symptoms, vitals, meds, history)
- LLM‑backed primary diagnosis with differential list and clinical recommendations
- Evidence layers (FOL, enhanced, online) turn free‑text into verifiable reasoning

### Imaging modality
- Image uploads (PNG/JPG/TIFF) flow into the diagnosis session
- Optional Clara integration for advanced imaging processing
- Heatmaps/visual explanations summarized into the UI explanations payload

### Video modality
- Large video files supported (configurable `MAX_UPLOAD_SIZE`), e.g., ultrasound clips
- Processing hooks detect video vs image and route via appropriate pipelines
- Results contribute to the unified diagnosis and verification stack

### Audio modality (STT)
- Audio notes (WAV/MP3/OGG/M4A/WEBM) converted to text via `AudioSTTService`
- Transcripts become part of the patient intake and reasoning context

### Knowledge modality
- UMLS lookup (with `UMLS_API_KEY`) enables code validation and terminology normalization
- Knowledge graph (Neo4j optional) for concept relationships and evidence expansion

### XAI layers (explainability)
- Clinical reasoning paths emitted as UI‑friendly strings
- FOL verification summary and confidence
- Enhanced evidence verification sources and confidence
- Online verification sources and confidence
- Overall verification status synthesized for the clinician


## Modality pipelines (high‑level)

```
Text Intake ─► LLM Diagnosis ─► XAI (FOL/Enhanced/Online) ─► UI Explanations

Image/Video ─► Imaging Processor ─┘                     └─► Confidence + Sources

Audio (STT) ─► Transcript ────────┘                     └─► Concern/EWS Update
```

Key integration points:
- `create_ui_data_structure(...)` builds a single payload the frontend expects
- `_finalize_concern_after_diagnosis(...)` updates persistent risk and Redis cache
- Clara and other heavy processors are optional and fail‑safe


## Endpoint examples

### Submit a diagnosis
```bash
curl -k -X POST "https://localhost:5000/api/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PATIENT_001",
    "symptoms": ["fever", "cough"],
    "vitals": {"hr": 96, "spo2": 97},
    "history": "No known chronic conditions"
  }'
```

Response (truncated):
```json
{
  "session_id": "e0d9...",
  "status": "processing"
}
```

### Poll status
```bash
curl -k "https://localhost:5000/api/status/e0d9..."
```

### Fetch results
```bash
curl -k "https://localhost:5000/api/results/e0d9..."
```

Results include:
- `explanations[]` (formatted clinical analysis)
- `confidenceScores{}` (overall + per‑layer)
- `verificationStatus{}` (FOL/enhanced/online)
- `sources{}` (textbooks, online citations)

### Upload image
```bash
curl -k -X POST "https://localhost:5000/api/upload" \
  -F "file=@/path/to/image.jpg" \
  -F "patient_id=PATIENT_001"
```

### CONCERN EWS current severity
```bash
curl -k "https://localhost:5000/api/concern/current?patient_id=PATIENT_001"
```

### UMLS code lookup (if enabled)
```bash
# Example depends on the service route; consult /umls-lookup UI or service endpoints
```


## Developer workflows

### Backend hot reload (manual)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export FLASK_ENV=development
python app.py
```

### Frontend hot reload
```bash
cd frontend
npm install
npm run dev
```

### Common service tasks
```bash
# Dockerized dev
make dev

# Watch logs for backend only
docker compose logs -f backend

# Database shell
make db-postgres
```

### Data and uploads
- Local dev: mounted at `backend/uploads/`
- Clean by deleting files or changing volume mounts


## Testing

### Backend tests (pytest)
```bash
docker compose exec backend python -m pytest tests/
```

### Frontend checks
```bash
cd frontend
npm run lint
npm run type-check
```

### E2E ideas (not included by default)
- Playwright/Cypress for UI flows (upload image, run diagnosis, verify explanations render)
- API contract tests for `/api/v2` optimized endpoints


## Benchmarks and performance tips
- Set `SPEED_MODE=1` to cap pipeline time and skip heavy steps
- Ensure Docker has 8GB+ RAM for smooth ML builds and faster runs
- Use production images (`make prod`) for faster startup and fewer rebuilds
- Enable Redis for caching; ensure `REDIS_URL` points to a healthy instance
- Prewarm patient cache by visiting a few records at startup if applicable


## Security & compliance
- Replace all default credentials and passwords before any public deployment
- Use CA‑issued SSL certificates in production; never ship self‑signed
- Restrict CORS origins to your trusted domains
- Store API keys as secrets, not in `.env` committed to VCS
- Review logs for PHI or PII; configure redaction if needed
- Consider audit logging and access controls for clinical usage


## UX tour (frontend)
- Diagnosis form with multimodal inputs
- Real‑time progress indicators and logs while the pipeline runs
- Results view with explanations, confidence, source tabs
- Patient severity charts pulling from the CONCERN EWS endpoints
- Optional pages: camera test, test reports (developer tooling)


## Demo scripts (suggested)

### End‑to‑end diagnosis
1. Start dev stack: `make dev`
2. Open the app: https://localhost:3000
3. Enter text symptoms and vitals
4. Upload an image or short video
5. (Optional) Record an audio note and attach
6. Submit and watch real‑time progress
7. Review explanations, sources, and risk level

### Imaging‑heavy 
1. Prepare multiple images/videos for a case
2. Upload sequentially to the same patient session
3. Trigger diagnosis to see image features influence the result


## FAQ

### Why both HTTP and HTTPS locally?
Backend auto‑enables HTTPS if certs exist. Otherwise, it falls back to HTTP for convenience. Frontend can run in either mode.

### Do I need Neo4j?
No. It’s optional. The system runs without it; enable it to explore knowledge‑graph‑powered features.

### What about NVIDIA Clara?
The app attempts real Clara,  and gracefully continues if neither is present.

### How do I change ports?
Edit `docker-compose*.yml` `ports:` mappings and/or frontend `NEXT_PUBLIC_API_URL`.

### Can I disable heavy steps?
Yes. Use `SPEED_MODE=1`. You can also trim verification steps in service layers.

### Where are uploads stored?
When dockerized, mapped to `backend/uploads`. In manual runs, `UPLOAD_FOLDER` defaults to `uploads`.

### How do I reset everything?
```bash
make reset   # destructive; removes volumes/images
```


## Extended troubleshooting

### Database migrations / schema
- The app ensures the CONCERN severity table exists on startup
- If schema drifts, rebuild DB volumes or run explicit migration scripts

### SSL handshake noise in logs
- Werkzeug logging is patched to suppress common client disconnect errors during development

### Large uploads timing out
- Increase `MAX_UPLOAD_SIZE` and ensure reverse proxy/client limits allow large bodies

### Slow first request after rebuild
- ML libraries JIT and caches can cause first calls to be slow. Warm up with a quick diagnosis request.

### GPU support
- Containers do not assume GPU. If you have CUDA, adapt Dockerfiles and base images accordingly.


## Glossary
- AR: Augmented Reasoning blueprint for diagnosis APIs
- EWS: Early Warning System for persistent risk tracking
- FOL: First‑Order Logic verification layer
- STT: Speech‑to‑Text
- UMLS: Unified Medical Language System


## Credits & acknowledgements
- Built for the Samsung PRISM GenAI Hackathon (Multimodal AI theme)
- Thanks to open‑source communities behind Flask, Next.js, Tailwind, and ML/NLP libraries
- Medical verification features inspired by evidence‑based clinical practice


## Roadmap (selected)
- More granular imaging explainability
- Real‑time video stream ingestion
- Pluggable LLM backends and tool routers
- Import/Export for diagnosis sessions and datasets
- FHIR/HL7 compatibility layer for EHR integration






