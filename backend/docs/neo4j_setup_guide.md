# Neo4j Setup and Integration Guide for CortexMD

## 🚀 Quick Installation Steps

### Option 1: Neo4j Desktop (Easiest for Development)

1. **Download Neo4j Desktop**
   ```powershell
   # Open PowerShell as Administrator and run:
   Start-Process "https://neo4j.com/download/"
   ```
   - Click "Download Neo4j Desktop"
   - Choose Windows version
   - Run the installer

2. **Setup Neo4j Desktop**
   - Launch Neo4j Desktop
   - Create a new project called "CortexMD"
   - Add a new database called "medical_knowledge"
   - Set password: `neo4j_cortex_2024`
   - Start the database

### Option 2: Neo4j Community Server (CLI Installation)

```powershell
# Install using Chocolatey (if you have it)
choco install neo4j-community -y

# OR Download manually
Invoke-WebRequest -Uri "https://neo4j.com/artifact.php?name=neo4j-community-5.15.0-windows.zip" -OutFile "neo4j.zip"
Expand-Archive -Path "neo4j.zip" -DestinationPath "C:\neo4j"
```

### Option 3: Docker Installation (Most Flexible)

```powershell
# Pull and run Neo4j container
docker pull neo4j:latest
docker run --name cortexmd-neo4j `
    -p 7474:7474 -p 7687:7687 `
    -d `
    -v ${PWD}/neo4j/data:/data `
    -v ${PWD}/neo4j/logs:/logs `
    -v ${PWD}/neo4j/import:/var/lib/neo4j/import `
    -v ${PWD}/neo4j/plugins:/plugins `
    --env NEO4J_AUTH=neo4j/neo4j_cortex_2024 `
    --env NEO4J_PLUGINS='["apoc", "graph-data-science"]' `
    neo4j:latest
```

## 🔧 Configuration

### 1. Update Neo4j Configuration
Edit `neo4j.conf` (located in Neo4j installation directory):

```properties
# Memory Configuration
server.memory.heap.initial_size=512m
server.memory.heap.max_size=2G
server.memory.pagecache.size=512m

# Network Configuration
server.default_listen_address=0.0.0.0
server.bolt.enabled=true
server.bolt.listen_address=:7687
server.http.enabled=true
server.http.listen_address=:7474

# Security
dbms.security.auth_enabled=true
dbms.security.procedures.unrestricted=apoc.*,gds.*

# APOC Configuration
apoc.export.file.enabled=true
apoc.import.file.enabled=true
apoc.import.file.use_neo4j_config=true
```

### 2. Install APOC and GDS Plugins

```powershell
# Download APOC (Advanced Procedures)
Invoke-WebRequest -Uri "https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/5.15.0/apoc-5.15.0-core.jar" -OutFile "apoc.jar"
Move-Item apoc.jar "C:\neo4j\plugins\"

# Download Graph Data Science
Invoke-WebRequest -Uri "https://graphdatascience.ninja/neo4j-graph-data-science-2.5.6.jar" -OutFile "gds.jar"
Move-Item gds.jar "C:\neo4j\plugins\"

# Restart Neo4j
neo4j restart
```

## 🐍 Python Dependencies

```bash
# Install Neo4j Python driver
pip install neo4j>=5.15.0
pip install py2neo>=2021.2.3
```

## 🔌 Environment Variables

Add to your `.env` file:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_cortex_2024
NEO4J_DATABASE=medical_knowledge
NEO4J_ENABLED=true

# Optional: Neo4j Aura (Cloud) Configuration
# NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=your_aura_password
```

## 📊 Initialize Medical Knowledge Graph

### Create the initial schema and indexes:

```cypher
// Run these in Neo4j Browser (http://localhost:7474)

// Create constraints and indexes for performance
CREATE CONSTRAINT unique_disease_name IF NOT EXISTS
FOR (d:Disease) REQUIRE d.name IS UNIQUE;

CREATE CONSTRAINT unique_symptom_name IF NOT EXISTS
FOR (s:Symptom) REQUIRE s.name IS UNIQUE;

CREATE CONSTRAINT unique_medication_name IF NOT EXISTS
FOR (m:Medication) REQUIRE m.name IS UNIQUE;

CREATE CONSTRAINT unique_anatomy_name IF NOT EXISTS
FOR (a:Anatomy) REQUIRE a.name IS UNIQUE;

CREATE INDEX disease_icd10 IF NOT EXISTS
FOR (d:Disease) ON (d.icd10_code);

CREATE INDEX disease_umls IF NOT EXISTS
FOR (d:Disease) ON (d.umls_cui);

// Create sample medical knowledge nodes
MERGE (sarcoma:Disease {
    name: 'Soft Tissue Sarcoma',
    icd10_code: 'C49.9',
    umls_cui: 'C1261473',
    description: 'Malignant tumor of soft tissue',
    severity: 'high',
    category: 'oncology'
})

MERGE (liposarcoma:Disease {
    name: 'Liposarcoma',
    icd10_code: 'C49.4',
    umls_cui: 'C0023827',
    description: 'Malignant tumor of adipose tissue',
    severity: 'high',
    category: 'oncology',
    subtype_of: 'Soft Tissue Sarcoma'
})

MERGE (myxofibrosarcoma:Disease {
    name: 'Myxofibrosarcoma',
    icd10_code: 'C49.9',
    umls_cui: 'C0334515',
    description: 'Malignant fibroblastic tumor with myxoid stroma',
    severity: 'high',
    category: 'oncology',
    subtype_of: 'Soft Tissue Sarcoma'
})

// Create symptoms
MERGE (mass:Symptom {name: 'Mass', description: 'Palpable lump or swelling'})
MERGE (pain:Symptom {name: 'Pain', description: 'Localized or radiating pain'})
MERGE (swelling:Symptom {name: 'Swelling', description: 'Tissue edema or enlargement'})
MERGE (limping:Symptom {name: 'Limping', description: 'Abnormal gait pattern'})

// Create relationships
MERGE (sarcoma)-[:PRESENTS_WITH {frequency: 0.8}]->(mass)
MERGE (sarcoma)-[:PRESENTS_WITH {frequency: 0.6}]->(pain)
MERGE (sarcoma)-[:PRESENTS_WITH {frequency: 0.7}]->(swelling)
MERGE (liposarcoma)-[:IS_SUBTYPE_OF]->(sarcoma)
MERGE (myxofibrosarcoma)-[:IS_SUBTYPE_OF]->(sarcoma)

// Create anatomical locations
MERGE (thigh:Anatomy {name: 'Thigh', region: 'Lower Extremity'})
MERGE (arm:Anatomy {name: 'Arm', region: 'Upper Extremity'})
MERGE (retroperitoneum:Anatomy {name: 'Retroperitoneum', region: 'Abdomen'})

// Create location relationships
MERGE (sarcoma)-[:OCCURS_IN {frequency: 0.4}]->(thigh)
MERGE (liposarcoma)-[:OCCURS_IN {frequency: 0.3}]->(retroperitoneum)
MERGE (myxofibrosarcoma)-[:OCCURS_IN {frequency: 0.5}]->(thigh)

// Create treatments
MERGE (surgery:Treatment {name: 'Surgical Resection', type: 'surgical'})
MERGE (radiation:Treatment {name: 'Radiation Therapy', type: 'radiation'})
MERGE (chemo:Treatment {name: 'Chemotherapy', type: 'systemic'})

// Create treatment relationships
MERGE (sarcoma)-[:TREATED_WITH {efficacy: 0.8, first_line: true}]->(surgery)
MERGE (sarcoma)-[:TREATED_WITH {efficacy: 0.6, adjuvant: true}]->(radiation)
MERGE (liposarcoma)-[:TREATED_WITH {efficacy: 0.5, for_advanced: true}]->(chemo)
```

## 🔍 Verify Installation

Run this test script to verify Neo4j is working:

```python
# test_neo4j.py
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

class Neo4jConnection:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            auth=(os.getenv('NEO4J_USER', 'neo4j'), 
                  os.getenv('NEO4J_PASSWORD', 'neo4j_cortex_2024'))
        )
    
    def test_connection(self):
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()['count']
            print(f"✅ Neo4j connected! Found {count} nodes in database")
    
    def get_sarcoma_info(self):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Disease {name: 'Soft Tissue Sarcoma'})-[:PRESENTS_WITH]->(s:Symptom)
                RETURN d.name as disease, collect(s.name) as symptoms
            """)
            for record in result:
                print(f"Disease: {record['disease']}")
                print(f"Symptoms: {', '.join(record['symptoms'])}")
    
    def close(self):
        self.driver.close()

if __name__ == "__main__":
    conn = Neo4jConnection()
    conn.test_connection()
    conn.get_sarcoma_info()
    conn.close()
```

## 🎯 Quick Start Commands

```powershell
# Start Neo4j (if installed as service)
neo4j start

# Check status
neo4j status

# Open Neo4j Browser
Start-Process "http://localhost:7474"

# Default credentials
# Username: neo4j
# Password: neo4j_cortex_2024
```

## 🚨 Troubleshooting

### Port Already in Use
```powershell
# Check what's using port 7687
netstat -an | findstr :7687

# Kill the process
taskkill /F /PID <process_id>
```

### Memory Issues
Increase heap size in `neo4j.conf`:
```properties
server.memory.heap.max_size=4G
```

### Can't Connect
1. Check Windows Firewall - allow ports 7474 and 7687
2. Verify service is running: `neo4j status`
3. Check logs: `C:\neo4j\logs\neo4j.log`

## 📚 Next Steps

1. **Import Medical Ontologies**
   - Load UMLS data
   - Import ICD-10 codes
   - Add SNOMED CT relationships

2. **Build FOL Rules**
   - Create logical inference rules
   - Add temporal relationships
   - Build causal chains

3. **Integrate with CortexMD**
   - Update `neo4j_service.py` configuration
   - Enable knowledge graph features
   - Test FOL verification with graph queries

## 🔗 Useful Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/)
- [APOC Procedures](https://neo4j.com/labs/apoc/)
- [Graph Data Science](https://neo4j.com/docs/graph-data-science/)
