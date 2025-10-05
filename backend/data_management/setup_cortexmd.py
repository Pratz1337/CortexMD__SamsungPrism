#!/usr/bin/env python3
"""
CortexMD Complete Setup Script
Sets up all required services: PostgreSQL, Redis, Neo4j, and dependencies
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                            🩺 CortexMD Setup Script                           ║
║                     Setting up Medical Analysis Platform                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

def run_command(command, description="", shell=True, check=True):
    """Run a shell command with nice output."""
    print(f"🔄 {description if description else command}")
    try:
        result = subprocess.run(command, shell=shell, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"   ✅ Success: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e.stderr.strip() if e.stderr else str(e)}")
        return False

def check_docker():
    """Check if Docker is installed and running."""
    print("\n📋 Checking Docker...")
    
    # Check if Docker is installed
    if not run_command("docker --version", "Checking Docker installation", check=False):
        print("❌ Docker is not installed. Please install Docker Desktop from https://www.docker.com/products/docker-desktop")
        return False
    
    # Check if Docker is running
    if not run_command("docker ps", "Checking if Docker is running", check=False):
        print("❌ Docker is not running. Please start Docker Desktop")
        return False
    
    print("✅ Docker is ready!")
    return True

def setup_environment():
    """Setup environment file."""
    print("\n📋 Setting up environment...")
    
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your API keys before running the application")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("❌ No .env.example file found")

def install_dependencies():
    """Install Python dependencies."""
    print("\n📋 Installing Python dependencies...")
    
    if run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("✅ Dependencies installed successfully!")
        return True
    else:
        print("❌ Failed to install dependencies")
        return False

def start_services():
    """Start all Docker services."""
    print("\n📋 Starting Docker services...")
    
    if run_command("docker-compose up -d", "Starting all services"):
        print("✅ All services started!")
        return True
    else:
        print("❌ Failed to start services")
        return False

def wait_for_services():
    """Wait for services to be ready."""
    print("\n📋 Waiting for services to be ready...")
    
    services = {
        "PostgreSQL": ("localhost", 5432),
        "Redis": ("localhost", 6379),
        "Neo4j": ("localhost", 7474)
    }
    
    for service_name, (host, port) in services.items():
        print(f"🔄 Waiting for {service_name}...")
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                if service_name == "Neo4j":
                    response = requests.get(f"http://{host}:{port}", timeout=5)
                    if response.status_code == 200:
                        print(f"✅ {service_name} is ready!")
                        break
                else:
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    if result == 0:
                        print(f"✅ {service_name} is ready!")
                        break
            except Exception:
                pass
            
            if attempt == max_attempts - 1:
                print(f"❌ {service_name} failed to start after {max_attempts} attempts")
                return False
            
            time.sleep(2)
    
    return True

def setup_database():
    """Setup database schema."""
    print("\n📋 Setting up database...")
    
    if run_command("python setup_database.py", "Setting up database schema"):
        print("✅ Database setup complete!")
        return True
    else:
        print("❌ Database setup failed")
        return False

def show_service_info():
    """Show information about running services."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                🚀 Services Ready!                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📊 Service URLs:
   • Flask Application:    http://localhost:5000
   • PostgreSQL Database:  localhost:5432
   • Redis Cache:          localhost:6379
   • Neo4j Browser:        http://localhost:7474
   • pgAdmin:              http://localhost:5050
   • Redis Commander:      http://localhost:8081

🔑 Default Credentials:
   • PostgreSQL: cortexmd_user / cortexmd_password
   • Neo4j:      neo4j / cortexmd_neo4j_password
   • pgAdmin:    admin@cortexmd.com / admin123

📝 Next Steps:
   1. Edit .env file with your API keys (UMLS, SNOMED, OpenAI, etc.)
   2. Start the Flask application: python app.py
   3. Access the application at http://localhost:5000

🛠️  Useful Commands:
   • Stop services:        docker-compose down
   • View logs:            docker-compose logs
   • Restart services:     docker-compose restart
   • Clean reset:          docker-compose down -v
    """)

def main():
    """Main setup function."""
    print_banner()
    
    # Change to backend directory if needed
    if not os.path.exists("docker-compose.yml"):
        backend_dir = Path("backend")
        if backend_dir.exists():
            os.chdir(backend_dir)
            print(f"📁 Changed to directory: {backend_dir.absolute()}")
    
    # Check prerequisites
    if not check_docker():
        sys.exit(1)
    
    # Setup steps
    setup_environment()
    
    if not install_dependencies():
        print("⚠️  Continuing despite dependency installation issues...")
    
    if not start_services():
        sys.exit(1)
    
    if not wait_for_services():
        print("⚠️  Some services may not be ready. Check docker-compose logs")
    
    # Setup database if possible
    if os.path.exists("setup_database.py"):
        setup_database()
    
    show_service_info()

if __name__ == "__main__":
    main()
