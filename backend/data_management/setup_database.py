#!/usr/bin/env python3
"""
CortexMD Database Setup Script
Sets up PostgreSQL database and Redis for CortexMD
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

def check_prerequisites():
    """Check if PostgreSQL and Redis are installed"""
    print("🔍 Checking prerequisites...")
    
    # Check PostgreSQL
    try:
        result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ PostgreSQL: {result.stdout.strip()}")
        else:
            print("❌ PostgreSQL not found")
            return False
    except FileNotFoundError:
        print("❌ PostgreSQL not found. Please install PostgreSQL 14+")
        return False
    
    # Check Redis
    try:
        result = subprocess.run(['redis-server', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Redis: {result.stdout.strip()}")
        else:
            print("❌ Redis not found")
            return False
    except FileNotFoundError:
        print("❌ Redis not found. Please install Redis 6+")
        return False
    
    return True

def setup_postgresql():
    """Set up PostgreSQL database"""
    print("\n🐘 Setting up PostgreSQL database...")
    
    db_name = "cortexmd_db"
    db_user = "cortexmd_user"
    db_password = "cortexmd_password"
    
    # Commands to run
    commands = [
        f"CREATE DATABASE {db_name};",
        f"CREATE USER {db_user} WITH ENCRYPTED PASSWORD '{db_password}';",
        f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {db_user};",
        f"ALTER USER {db_user} CREATEDB;"
    ]
    
    try:
        for cmd in commands:
            print(f"  Executing: {cmd}")
            result = subprocess.run(
                ['psql', '-U', 'postgres', '-c', cmd],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0 and "already exists" not in result.stderr:
                print(f"  ⚠️  Warning: {result.stderr.strip()}")
            else:
                print(f"  ✅ Success")
        
        print("✅ PostgreSQL database setup completed")
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup PostgreSQL: {e}")
        return False

def setup_redis():
    """Set up Redis (just verify it's running)"""
    print("\n🚀 Setting up Redis...")
    
    try:
        # Check if Redis is running
        result = subprocess.run(['redis-cli', 'ping'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip() == "PONG":
            print("✅ Redis is running and accessible")
            return True
        else:
            print("❌ Redis is not running. Please start Redis server:")
            print("  Windows: redis-server")
            print("  Linux/Mac: sudo systemctl start redis")
            return False
            
    except Exception as e:
        print(f"❌ Failed to check Redis: {e}")
        return False

def install_python_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Python dependencies installed")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

async def test_database_connection():
    """Test database connections"""
    print("\n🧪 Testing database connections...")
    
    try:
        from database_config import initialize_database
        
        db, session_mgr = await initialize_database()
        print("✅ Database connections successful")
        
        # Test session creation
        test_session_id = "test_session_123"
        test_patient_input = {
            "patient_id": "test_patient",
            "text_data": "Test patient data"
        }
        
        success = await session_mgr.create_diagnosis_session(
            test_session_id, test_patient_input, False
        )
        
        if success:
            print("✅ Session management test successful")
            
            # Clean up test session
            await session_mgr.update_diagnosis_session(test_session_id, {
                'status': 'deleted'
            })
        else:
            print("❌ Session management test failed")
            
        await db.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False

def create_systemd_services():
    """Create systemd service files for Linux"""
    print("\n⚙️  Creating systemd service files...")
    
    cortexmd_service = """[Unit]
Description=CortexMD Medical Diagnosis System
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/cortexmd/backend
Environment=PATH=/path/to/cortexmd/venv/bin
ExecStart=/path/to/cortexmd/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    try:
        with open("cortexmd.service", "w") as f:
            f.write(cortexmd_service)
        print("✅ Created cortexmd.service")
        print("  To install: sudo cp cortexmd.service /etc/systemd/system/")
        print("  To enable: sudo systemctl enable cortexmd")
        print("  To start: sudo systemctl start cortexmd")
        
    except Exception as e:
        print(f"❌ Failed to create service file: {e}")

def main():
    """Main setup function"""
    print("🏥 CortexMD Database Setup")
    print("=" * 40)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please install PostgreSQL and Redis.")
        return False
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("\n❌ Failed to install Python dependencies.")
        return False
    
    # Setup PostgreSQL
    if not setup_postgresql():
        print("\n❌ PostgreSQL setup failed.")
        return False
    
    # Setup Redis
    if not setup_redis():
        print("\n❌ Redis setup failed.")
        return False
    
    # Test connections
    success = asyncio.run(test_database_connection())
    if not success:
        print("\n❌ Database connection tests failed.")
        return False
    
    # Create service files
    create_systemd_services()
    
    print("\n🎉 CortexMD database setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Update .env file with your database credentials")
    print("2. Start Redis server if not running")
    print("3. Run: python app.py")
    print("4. Access CortexMD at: http://localhost:5000")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
