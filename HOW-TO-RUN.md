# 🚀 How to Run Both Backend and Frontend - Complete Guide

## ✨ **EASIEST METHOD: One-Click Launcher**

### Option 1: Double-Click to Start Everything
```
🖱️ Double-click: START-CORTEXMD.bat
```
This will automatically:
- Check if Docker is installed
- Start Docker Desktop if needed  
- Wait for Docker to be ready
- Run the complete setup script
- Start both backend and frontend in containers


## 🐳 **DOCKER METHOD: Containerized (Recommended)**

### Step 1: Start Docker Desktop
- **Manual**: Double-click Docker Desktop icon from Start Menu
- **Script**: The setup script will do this automatically

### Step 2: Run Setup Script
```powershell
# Windows PowerShell (Recommended)
.\setup.ps1

# Alternative: Use batch file with menu
.\setup.bat
```

### Step 3: Access Your Application
- **Frontend**: https://localhost:3000
- **Backend API**: https://localhost:5000
- **Database Admin**: http://localhost:7475 (Neo4j)

## 💻 **MANUAL METHOD: Run Without Docker**

If you prefer not to use Docker, you can run both services manually:

### Terminal 1 - Backend:
```powershell
cd backend
pip install -r requirements.txt
python app.py
```

### Terminal 2 - Frontend:
```powershell
cd frontend  
npm install
npm run dev
```

### Terminal 3 - Databases (Optional):
```powershell
# Start databases manually if needed
# PostgreSQL, Neo4j, Redis would need to be installed separately
```

## 🎯 **RECOMMENDED APPROACH**

For the easiest experience, I recommend:

1. **First Time**: Double-click `START-CORTEXMD.bat` - it handles everything
2. **Daily Use**: Use `.\setup.ps1` or the Docker commands
3. **Development**: Use `.\setup.ps1 -Dev` for hot reload

## 🛠️ **Available Scripts Summary**

| Script | Purpose | Best For |
|--------|---------|----------|
| `START-CORTEXMD.bat` | One-click everything | First-time users |
| `start-cortexmd.ps1` | PowerShell launcher | Regular use |
| `setup.ps1` | Main setup script | Developers | 
| `setup.bat` | Menu-driven setup | GUI lovers |
| `setup.sh` | Linux/macOS setup | Non-Windows |

## 🔧 **Docker Commands (Manual)**

If you want to use Docker commands directly:

```powershell
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f

# Stop everything  
docker-compose down

# Rebuild and start
docker-compose up --build -d
```

## ⚡ **Quick Status Check**

To check if everything is running:
```powershell
docker-compose ps
```

You should see:
- ✅ cortexmd-frontend
- ✅ cortexmd-backend  
- ✅ postgres-cortexmd
- ✅ neo4j-cortexmd
- ✅ redis-cortex

---

**🎉 Just run `START-CORTEXMD.bat` and you're ready to go!**