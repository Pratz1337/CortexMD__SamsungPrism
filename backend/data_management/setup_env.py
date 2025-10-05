#!/usr/bin/env python3
"""
Environment Setup Script for CortexMD
Creates .env file with PostgreSQL configuration
"""

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_env_file():
    """Create .env file with PostgreSQL configuration"""
    
    env_content = """# CortexMD Environment Configuration
# PostgreSQL Database Configuration (Generated)

# =================================
# PostgreSQL Database Configuration
# =================================

# Individual PostgreSQL settings (used by app.py) - Cloud Configuration
POSTGRES_HOST=pgnode305-mum-1.database.excloud.co.in
POSTGRES_PORT=5432
POSTGRES_DB=cortexmd
POSTGRES_USER=postgres
POSTGRES_PASSWORD=xi6REKcZ3g33qwEk

# Complete PostgreSQL URL (alternative to individual settings) - Cloud Configuration
DATABASE_URL=postgresql://postgres:xi6REKcZ3g33qwEk@pgnode305-mum-1.database.excloud.co.in:5432/cortexmd
DATABASE_TYPE=postgresql

# =================================
# AI Service Configuration
# =================================

# Google AI API Key (required for medical diagnosis)
GOOGLE_API_KEY=your_google_gemini_api_key_here

# NVIDIA Clara AI (optional - for advanced medical AI)
NVIDIA_API_KEY=your_nvidia_api_key_here

# =================================
# Redis Configuration
# =================================

# Redis settings (for caching and chat history)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# =================================
# Flask Application Settings
# =================================

# Flask environment
FLASK_ENV=development
FLASK_DEBUG=true
FLASK_SECRET_KEY=cortexmd-secret-key-change-in-production

# Server settings
PORT=5000
HOST=0.0.0.0

# =================================
# Medical API Services
# =================================

# UMLS (Unified Medical Language System) API
UMLS_API_KEY=your_umls_api_key_here

# =================================
# Performance & Monitoring
# =================================

# Enable performance monitoring
ENABLE_PERFORMANCE_MONITORING=true
"""
    
    env_file_path = os.path.join(os.path.dirname(__file__), '.env')
    
    try:
        # Check if .env already exists
        if os.path.exists(env_file_path):
            logger.info("⚠️ .env file already exists")
            response = input("Do you want to overwrite it? (y/N): ").lower()
            if response != 'y':
                logger.info("❌ Setup cancelled")
                return False
        
        # Write .env file
        with open(env_file_path, 'w') as f:
            f.write(env_content)
        
        logger.info(f"✅ Created .env file at {env_file_path}")
        logger.info("🔧 Default PostgreSQL configuration:")
        logger.info("   - Host: localhost:5432")
        logger.info("   - Database: cortexmd")
        logger.info("   - User: postgres")
        logger.info("   - Password: password")
        logger.info("")
        logger.info("💡 Next steps:")
        logger.info("   1. Update .env with your actual PostgreSQL credentials")
        logger.info("   2. Add your Google AI API key to .env")
        logger.info("   3. Run: python setup_postgresql.py")
        logger.info("   4. Run: python app.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create .env file: {e}")
        return False

def validate_env_file():
    """Validate existing .env file"""
    env_file_path = os.path.join(os.path.dirname(__file__), '.env')
    
    if not os.path.exists(env_file_path):
        logger.error("❌ .env file not found")
        return False
    
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file_path)
        
        # Check required variables
        required_vars = [
            'POSTGRES_HOST',
            'POSTGRES_PORT', 
            'POSTGRES_DB',
            'POSTGRES_USER',
            'POSTGRES_PASSWORD',
            'DATABASE_URL'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            return False
        
        logger.info("✅ .env file validation passed")
        logger.info(f"🐘 Database: {os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}")
        logger.info(f"👤 User: {os.getenv('POSTGRES_USER')}")
        logger.info(f"🔑 API Key: {'✅ Set' if os.getenv('GOOGLE_API_KEY') != 'your_google_gemini_api_key_here' else '❌ Not set'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ .env file validation failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🔧 CortexMD Environment Setup")
    logger.info("=" * 30)
    
    # Check if .env exists
    env_file_path = os.path.join(os.path.dirname(__file__), '.env')
    
    if os.path.exists(env_file_path):
        logger.info("📄 Found existing .env file")
        if validate_env_file():
            logger.info("🎉 Environment setup is complete!")
            return True
        else:
            logger.info("🔧 Recreating .env file...")
    else:
        logger.info("📄 No .env file found, creating new one...")
    
    # Create .env file
    if create_env_file():
        logger.info("🎉 Environment setup completed!")
        return True
    else:
        logger.error("❌ Environment setup failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
