#!/usr/bin/env python3
"""
PostgreSQL Setup Script for CortexMD
Creates database, tables, and sample data
"""

import os
import sys
import logging
import subprocess
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_postgresql_installed():
    """Check if PostgreSQL is installed"""
    try:
        result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ PostgreSQL found: {result.stdout.strip()}")
            return True
        else:
            logger.error("❌ PostgreSQL not found")
            return False
    except FileNotFoundError:
        logger.error("❌ PostgreSQL not installed or not in PATH")
        return False

def create_database():
    """Create CortexMD database (Cloud PostgreSQL)"""
    try:
        logger.info("☁️ Using cloud PostgreSQL - database should already exist")
        logger.info("🔗 Connecting to: pgnode305-mum-1.database.excloud.co.in")
        
        # For cloud databases, we typically don't create the database
        # Instead, we test the connection
        from postgresql_database import get_postgresql_database
        
        try:
            db_url = 'postgresql://postgres:xi6REKcZ3g33qwEk@pgnode305-mum-1.database.excloud.co.in:5432/cortexmd'
            db = get_postgresql_database(db_url)
            health = db.health_check()
            
            if health['status'] == 'healthy':
                logger.info("✅ Cloud PostgreSQL database connection successful")
                return True
            else:
                logger.error(f"❌ Cloud PostgreSQL connection failed: {health.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to cloud PostgreSQL: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing cloud database: {e}")
        return False

def test_connection():
    """Test PostgreSQL connection"""
    try:
        from postgresql_database import get_postgresql_database
        
        # Test connection
        db = get_postgresql_database()
        health = db.health_check()
        
        if health['status'] == 'healthy':
            logger.info(f"✅ PostgreSQL connection successful")
            logger.info(f"   📊 Patient count: {health.get('patient_count', 0)}")
            logger.info(f"   🔗 Connection: {health.get('connection_url', 'unknown')}")
            return True
        else:
            logger.error(f"❌ PostgreSQL connection failed: {health.get('error', 'unknown')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        return False

def setup_environment_variables():
    """Setup environment variables for PostgreSQL"""
    env_vars = {
        'DATABASE_URL': 'postgresql://postgres:xi6REKcZ3g33qwEk@pgnode305-mum-1.database.excloud.co.in:5432/cortexmd',
        'POSTGRES_HOST': 'pgnode305-mum-1.database.excloud.co.in',
        'POSTGRES_PORT': '5432',
        'POSTGRES_DB': 'cortexmd',
        'POSTGRES_USER': 'postgres',
        'POSTGRES_PASSWORD': 'xi6REKcZ3g33qwEk'
    }
    
    logger.info("🔧 Setting up environment variables...")
    
    # Create .env file
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    
    with open(env_file, 'w') as f:
        f.write("# PostgreSQL Configuration for CortexMD\n")
        f.write(f"# Generated on {datetime.now().isoformat()}\n\n")
        
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
            os.environ[key] = value
    
    logger.info(f"✅ Environment variables written to {env_file}")

def install_requirements():
    """Install PostgreSQL Python requirements"""
    requirements = [
        'psycopg2-binary>=2.9.0',
        'SQLAlchemy>=1.4.0',
        'alembic>=1.7.0'
    ]
    
    logger.info("📦 Installing PostgreSQL requirements...")
    
    for req in requirements:
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', req
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"   ✅ Installed {req}")
            else:
                logger.warning(f"   ⚠️ Failed to install {req}: {result.stderr}")
        except Exception as e:
            logger.error(f"   ❌ Error installing {req}: {e}")

def main():
    """Main setup function"""
    logger.info("🐘 CortexMD PostgreSQL Setup")
    logger.info("=" * 40)
    
    # Step 1: Install requirements
    install_requirements()
    
    # Step 2: Check PostgreSQL installation
    if not check_postgresql_installed():
        logger.error("❌ Please install PostgreSQL first:")
        logger.error("   Windows: Download from https://www.postgresql.org/download/windows/")
        logger.error("   macOS: brew install postgresql")
        logger.error("   Ubuntu: sudo apt-get install postgresql postgresql-contrib")
        return False
    
    # Step 3: Setup environment variables
    setup_environment_variables()
    
    # Step 4: Create database
    if not create_database():
        logger.error("❌ Database creation failed")
        return False
    
    # Step 5: Test connection and create tables
    if not test_connection():
        logger.error("❌ Connection test failed")
        return False
    
    logger.info("🎉 PostgreSQL setup completed successfully!")
    logger.info("🚀 You can now start the CortexMD application")
    logger.info("   python app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
