#!/usr/bin/env python3
"""
Database Reset Script for CortexMD
Removes old database files and creates fresh schema
"""

import os
import logging
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_old_databases():
    """Remove all existing database files"""
    db_patterns = [
        "*.db",
        "*.sqlite",
        "*.sqlite3"
    ]
    
    removed_files = []
    for pattern in db_patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                removed_files.append(file)
                logger.info(f"🗑️ Removed old database: {file}")
            except Exception as e:
                logger.warning(f"Could not remove {file}: {e}")
    
    if removed_files:
        logger.info(f"✅ Removed {len(removed_files)} old database files")
    else:
        logger.info("ℹ️ No old database files found")

def create_fresh_database():
    """Create a fresh database with proper schema"""
    try:
        logger.info("🔨 Creating fresh database...")
        
        # Import the database module to trigger initialization
        from simple_database import CortexMDDatabase
        
        # Create new database instance
        db = CortexMDDatabase("cortexmd_fresh.db")
        
        # Test basic operations
        logger.info("🧪 Testing database operations...")
        
        # Test patient retrieval (should find sample patient)
        patient = db.get_patient("PATIENT_001")
        if patient:
            logger.info(f"✅ Sample patient found: {patient['patient_name']}")
        else:
            logger.warning("⚠️ Sample patient not found")
        
        # Test diagnosis session creation
        test_session_id = "RESET_TEST_001"
        test_input = {
            'patient_id': 'PATIENT_001',
            'symptoms': 'Database reset test',
            'text_data': 'Testing fresh database'
        }
        
        success = db.create_diagnosis_session(test_session_id, "PATIENT_001", test_input)
        if success:
            logger.info("✅ Diagnosis session creation works")
            
            # Test retrieval
            sessions = db.get_patient_diagnosis_sessions("PATIENT_001")
            logger.info(f"✅ Found {len(sessions)} diagnosis sessions")
        else:
            logger.error("❌ Diagnosis session creation failed")
        
        # Test chat functionality
        try:
            db.ensure_chat_session("RESET_CHAT_001", "PATIENT_001")
            db.save_chat_message("RESET_CHAT_001", "PATIENT_001", "user", "Database reset test message")
            messages = db.get_patient_chat_history("PATIENT_001")
            logger.info(f"✅ Chat functionality works, {len(messages)} messages")
        except Exception as e:
            logger.error(f"❌ Chat functionality failed: {e}")
        
        logger.info("🎉 Fresh database created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create fresh database: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main reset function"""
    logger.info("🔄 CortexMD Database Reset")
    logger.info("=" * 30)
    
    # Remove old databases
    remove_old_databases()
    
    # Create fresh database
    if create_fresh_database():
        logger.info("✅ Database reset completed successfully!")
        logger.info("You can now start the application with: python start_app.py")
        return True
    else:
        logger.error("❌ Database reset failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
