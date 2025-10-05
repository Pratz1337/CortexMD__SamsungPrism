#!/usr/bin/env python3
"""
PostgreSQL Setup Script for CortexMD
Creates database, user, and initializes tables
"""

import os
import sys
import subprocess
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command"""
    logger.info(f"Running: {cmd}")
    return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)

def setup_postgresql_database(
    db_name: str = "cortexmd",
    db_user: str = "cortexmd", 
    db_password: str = "cortexmd",
    db_host: str = "localhost",
    db_port: int = 5432
):
    """Set up PostgreSQL database for CortexMD"""
    
    logger.info("🚀 Setting up PostgreSQL database for CortexMD...")
    
    # Check if PostgreSQL is installed
    try:
        result = run_command("psql --version")
        logger.info(f"✅ PostgreSQL found: {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        logger.error("❌ PostgreSQL not found. Please install PostgreSQL first.")
        sys.exit(1)
    
    # Create database and user (as postgres superuser)
    logger.info("Creating database and user...")
    
    create_db_commands = [
        f"CREATE DATABASE {db_name};",
        f"CREATE USER {db_user} WITH PASSWORD '{db_password}';",
        f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {db_user};",
        f"ALTER USER {db_user} CREATEDB;"
    ]
    
    for cmd in create_db_commands:
        try:
            psql_cmd = f'psql -U postgres -c "{cmd}"'
            run_command(psql_cmd, check=False)  # Don't fail if user/db already exists
        except subprocess.CalledProcessError as e:
            logger.warning(f"Command failed (may be expected): {cmd}")
    
    # Set environment variables
    database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    os.environ['DATABASE_URL'] = database_url
    os.environ['DATABASE_TYPE'] = 'postgresql'
    
    logger.info(f"✅ Database URL: {database_url}")
    
    # Initialize database with SQLAlchemy
    try:
        logger.info("Initializing database tables...")
        from postgres_database import initialize_postgres_database
        db = initialize_postgres_database(database_url)
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        sys.exit(1)
    
    # Initialize Alembic (optional)
    try:
        logger.info("Initializing Alembic migrations...")
        run_command("alembic revision --autogenerate -m 'Initial migration'", check=False)
        run_command("alembic upgrade head", check=False)
        logger.info("✅ Alembic migrations initialized")
    except Exception as e:
        logger.warning(f"⚠️ Alembic initialization failed: {e}")
    
    logger.info("🎉 PostgreSQL setup completed successfully!")
    logger.info(f"Database URL: {database_url}")
    logger.info("You can now start the CortexMD application.")

def create_env_file():
    """Create .env file with database configuration"""
    env_content = f"""# CortexMD Environment Configuration
DATABASE_TYPE=postgresql
DATABASE_URL=postgresql://cortexmd:cortexmd@localhost:5432/cortexmd

# Google API Key (required for AI features)
GOOGLE_API_KEY=your_google_api_key_here

# Redis Configuration (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=true
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    logger.info("✅ Created .env file with default configuration")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up PostgreSQL for CortexMD")
    parser.add_argument("--db-name", default="cortexmd", help="Database name")
    parser.add_argument("--db-user", default="cortexmd", help="Database user")
    parser.add_argument("--db-password", default="cortexmd", help="Database password")
    parser.add_argument("--db-host", default="localhost", help="Database host")
    parser.add_argument("--db-port", type=int, default=5432, help="Database port")
    parser.add_argument("--create-env", action="store_true", help="Create .env file")
    
    args = parser.parse_args()
    
    if args.create_env:
        create_env_file()
    
    setup_postgresql_database(
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        db_host=args.db_host,
        db_port=args.db_port
    )
