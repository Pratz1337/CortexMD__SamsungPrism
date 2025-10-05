#!/usr/bin/env python3
"""
Create Sample Patients for CortexMD
"""

import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_patients():
    """Create sample patients for testing"""
    try:
        from database_manager import get_database
        db = get_database()
        
        # Sample patients to create
        sample_patients = [
            {
                'patient_id': 'PATIENT_001',
                'patient_name': 'John Doe',
                'date_of_birth': '1985-06-15',
                'gender': 'Male',
                'admission_date': datetime.now().isoformat(),
                'current_status': 'active'
            },
            {
                'patient_id': '45678908765',
                'patient_name': 'Sarah Johnson',
                'date_of_birth': '1992-03-22',
                'gender': 'Female',
                'admission_date': datetime.now().isoformat(),
                'current_status': 'active'
            },
            {
                'patient_id': 'PATIENT_003',
                'patient_name': 'Michael Chen',
                'date_of_birth': '1978-11-08',
                'gender': 'Male',
                'admission_date': datetime.now().isoformat(),
                'current_status': 'active'
            },
            {
                'patient_id': 'PATIENT_004',
                'patient_name': 'Emily Davis',
                'date_of_birth': '1995-07-14',
                'gender': 'Female',
                'admission_date': datetime.now().isoformat(),
                'current_status': 'active'
            }
        ]
        
        created_count = 0
        for patient_data in sample_patients:
            # Check if patient already exists
            existing_patient = db.get_patient(patient_data['patient_id'])
            
            if existing_patient:
                logger.info(f"✅ Patient {patient_data['patient_id']} already exists: {existing_patient['patient_name']}")
            else:
                # Create new patient
                if db.create_patient(patient_data):
                    logger.info(f"🆕 Created patient {patient_data['patient_id']}: {patient_data['patient_name']}")
                    created_count += 1
                else:
                    logger.error(f"❌ Failed to create patient {patient_data['patient_id']}")
        
        # Get all patients
        all_patients = db.get_all_patients()
        logger.info(f"📊 Total patients in database: {len(all_patients)}")
        
        for patient in all_patients:
            logger.info(f"   • {patient['patient_id']}: {patient['patient_name']} ({patient['current_status']})")
        
        logger.info(f"🎉 Sample patients setup complete! Created {created_count} new patients.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create sample patients: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_sample_patients()
    exit(0 if success else 1)
