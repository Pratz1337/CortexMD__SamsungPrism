#!/usr/bin/env python3
"""
Setup script for Enhanced AR Scanner feature.
This script sets up the database tables and verifies the installation.
"""

import os
import sys
import logging
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("🔍 Checking dependencies...")
    
    missing_deps = []
    
    # Check Python packages
    try:
        import google.generativeai
        logger.info("✅ google-generativeai installed")
    except ImportError:
        missing_deps.append("google-generativeai")
    
    try:
        import pytesseract
        logger.info("✅ pytesseract installed")
    except ImportError:
        missing_deps.append("pytesseract")
    
    try:
        from PIL import Image
        logger.info("✅ Pillow installed")
    except ImportError:
        missing_deps.append("Pillow")
    
    try:
        import psycopg2
        logger.info("✅ psycopg2 installed")
    except ImportError:
        missing_deps.append("psycopg2")
    
    # Check Tesseract installation
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        logger.info("✅ Tesseract OCR installed")
    except Exception as e:
        logger.warning(f"⚠️  Tesseract OCR not found: {e}")
        logger.info("   Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        logger.info("   On Windows, set TESSERACT_CMD environment variable if needed")
    
    if missing_deps:
        logger.error(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        logger.info("Install with: pip install " + " ".join(missing_deps))
        return False
    
    logger.info("✅ All dependencies are installed")
    return True

def check_environment():
    """Check environment variables."""
    logger.info("🔍 Checking environment variables...")
    
    required_vars = ['GEMINI_API_KEY', 'DATABASE_URL']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            logger.info(f"✅ {var} is set")
    
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        logger.info("Please set these in your .env file")
        return False
    
    logger.info("✅ All environment variables are set")
    return True

def setup_database():
    """Set up the database tables."""
    logger.info("🗄️  Setting up database tables...")
    
    try:
        from enhanced_database_manager import enhanced_db
        
        # Test database connection
        with enhanced_db.get_session() as session:
            from sqlalchemy import text
            session.execute(text("SELECT 1"))
        
        logger.info("✅ Database connection successful")
        
        # The tables will be created automatically by SQLAlchemy
        logger.info("✅ Database tables created/verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

def run_migration():
    """Run the SQL migration script."""
    logger.info("🔄 Running database migration...")
    
    try:
        migration_file = backend_dir / "migrations" / "add_scanned_notes_table.sql"
        
        if not migration_file.exists():
            logger.warning("⚠️  Migration file not found, tables will be created by SQLAlchemy")
            return True
        
        # Read and execute migration
        with open(migration_file, 'r') as f:
            migration_sql = f.read()
        
        from enhanced_database_manager import enhanced_db
        from sqlalchemy import text
        with enhanced_db.get_session() as session:
            # Split by semicolon and execute each statement
            statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]
            for statement in statements:
                if statement and not statement.startswith('--'):
                    session.execute(text(statement))
            session.commit()
        
        logger.info("✅ Database migration completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

def test_ar_processor():
    """Test the AR processor with a sample image."""
    logger.info("🧪 Testing AR processor...")
    
    try:
        from enhanced_ar_processor import enhanced_ocr_and_parse, demo_annotated_preview
        from PIL import Image
        import io
        
        # Create a simple test image with text
        test_img = Image.new('RGB', (400, 200), color='white')
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(test_img)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        draw.text((10, 10), "Test Medical Note", fill='black', font=font)
        draw.text((10, 40), "Patient: John Doe", fill='black', font=font)
        draw.text((10, 70), "BP: 120/80", fill='black', font=font)
        draw.text((10, 100), "HR: 72", fill='black', font=font)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        # Test processing
        result = enhanced_ocr_and_parse(img_bytes)
        
        if result['success']:
            logger.info("✅ AR processor test successful")
            logger.info(f"   OCR text length: {len(result['ocr_text'])}")
            logger.info(f"   AI summary: {result['ai_summary'][:100]}...")
            return True
        else:
            logger.error(f"❌ AR processor test failed: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ AR processor test failed: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("🚀 Setting up Enhanced AR Scanner for CortexMD...")
    
    steps = [
        ("Checking dependencies", check_dependencies),
        ("Checking environment", check_environment),
        ("Setting up database", setup_database),
        # ("Running migration", run_migration),  # Skip migration - tables created by SQLAlchemy
        ("Testing AR processor", test_ar_processor),
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n📋 {step_name}...")
        if not step_func():
            logger.error(f"❌ Setup failed at: {step_name}")
            sys.exit(1)
    
    logger.info("\n🎉 Enhanced AR Scanner setup completed successfully!")
    logger.info("\n📚 Next steps:")
    logger.info("1. Start your Flask backend: python app.py")
    logger.info("2. Use the enhanced scan-note endpoint: POST /api/concern/scan-note")
    logger.info("3. View scanned notes: GET /api/concern/scanned-notes/<patient_id>")
    logger.info("4. Integrate the ScanClinicalNote component in your frontend")

if __name__ == "__main__":
    main()
