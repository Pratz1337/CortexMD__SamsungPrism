"""
Database Manager for CortexMD
PostgreSQL Implementation as Requested
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def get_database():
    """Get database instance - PostgreSQL as requested"""
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Try PostgreSQL first
        database_url = os.getenv('DATABASE_URL')
        
        if database_url and 'postgresql' in database_url:
            logger.info("🐘 Using PostgreSQL database from DATABASE_URL")
            try:
                from .postgresql_database import get_postgresql_database
            except ImportError:
                from core.postgresql_database import get_postgresql_database
            return get_postgresql_database(database_url)
        
        # Check for individual PostgreSQL environment variables
        postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
        postgres_port = os.getenv('POSTGRES_PORT', '5432')
        postgres_db = os.getenv('POSTGRES_DB', 'cortexmd')
        postgres_user = os.getenv('POSTGRES_USER', 'postgres')
        postgres_password = os.getenv('POSTGRES_PASSWORD', 'password')
        
        # Build PostgreSQL URL
        database_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
        logger.info(f"🐘 Using PostgreSQL: {postgres_host}:{postgres_port}/{postgres_db}")
        
        try:
            from .postgresql_database import get_postgresql_database
        except ImportError:
            from core.postgresql_database import get_postgresql_database
        return get_postgresql_database(database_url)
        
    except ImportError as e:
        logger.error(f"❌ PostgreSQL dependencies not installed: {e}")
        logger.error("💡 Please run: python setup_postgresql.py")
        raise Exception("PostgreSQL dependencies missing. Run setup_postgresql.py first.")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize PostgreSQL database: {e}")
        logger.error("💡 Make sure PostgreSQL is running and configured correctly")
        logger.error("💡 Run: python setup_postgresql.py")
        raise Exception(f"PostgreSQL connection failed: {e}")

def get_database_type():
    """Get the type of database being used"""
    return "postgresql"

def setup_database():
    """Setup PostgreSQL database"""
    try:
        import subprocess
        import sys
        
        logger.info("🔧 Setting up PostgreSQL database...")
        
        # Run setup script
        result = subprocess.run([
            sys.executable, 
            os.path.join(os.path.dirname(__file__), 'setup_postgresql.py')
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ PostgreSQL setup completed")
            return True
        else:
            logger.error(f"❌ PostgreSQL setup failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to setup PostgreSQL: {e}")
        return False