#!/usr/bin/env python3
"""
Build comprehensive severity history with proper JSON handling
"""

from database_manager import get_database
import requests
import urllib3
from datetime import datetime, timedelta
import json

urllib3.disable_warnings()

def build_history():
    print("🔧 BUILDING COMPREHENSIVE SEVERITY HISTORY")
    print("=" * 60)
    
    db = get_database()
    
    # Delete and recreate the TEST_PATIENT severity record
    print("\n🗑️  Removing existing TEST_PATIENT severity record...")
    
    with db.get_session() as session:
        from postgresql_database import ConcernSeverityTracking
        
        # Delete existing record completely
        existing = session.query(ConcernSeverityTracking).filter_by(patient_id='TEST_PATIENT').first()
        if existing:
            session.delete(existing)
            session.commit()
            print("   ✅ Deleted existing record")
        else:
            print("   ✅ No existing record found")
    
    # Now add diagnoses one by one with realistic timestamps
    diagnoses = [
        {
            'diagnosis_id': 'DIAG_001',
            'confidence': 0.65,
            'fol': 0.6,
            'enhanced': 0.5,
            'explain': 0.7,
            'imaging': False,
            'description': 'Initial symptoms assessment'
        },
        {
            'diagnosis_id': 'DIAG_002', 
            'confidence': 0.78,
            'fol': 0.75,
            'enhanced': 0.65,
            'explain': 0.8,
            'imaging': True,
            'description': 'Follow-up with imaging'
        },
        {
            'diagnosis_id': 'DIAG_003',
            'confidence': 0.85,
            'fol': 0.88,
            'enhanced': 0.75,
            'explain': 0.82,
            'imaging': True,
            'description': 'Advanced diagnostics'
        },
        {
            'diagnosis_id': 'DIAG_004',
            'confidence': 0.92,
            'fol': 0.9,
            'enhanced': 0.85,
            'explain': 0.88,
            'imaging': True,
            'description': 'Comprehensive analysis'
        },
        {
            'diagnosis_id': 'DIAG_005',
            'confidence': 0.95,
            'fol': 0.95,
            'enhanced': 0.9,
            'explain': 0.9,
            'imaging': True,
            'description': 'Critical assessment'
        }
    ]
    
    print(f"\n📊 Adding {len(diagnoses)} diagnoses sequentially...")
    
    for i, diag in enumerate(diagnoses):
        print(f"\n   📋 Diagnosis {i+1}: {diag['diagnosis_id']}")
        print(f"       {diag['description']}")
        print(f"       Confidence: {diag['confidence']:.2f}")
        
        result = db.update_patient_severity(
            patient_id='TEST_PATIENT',
            diagnosis_confidence=diag['confidence'],
            fol_verification=diag['fol'],
            enhanced_verification=diag['enhanced'],
            explainability_score=diag['explain'],
            imaging_present=diag['imaging'],
            diagnosis_id=diag['diagnosis_id']
        )
        
        if result:
            print(f"       ✅ Added successfully")
            print(f"          Risk: {result['risk_level']}")
            print(f"          Score: {result['risk_score']:.3f}")
            print(f"          Cumulative: {result['cumulative_severity']:.3f}")
        else:
            print(f"       ❌ Failed to add")
        
        # Brief pause to ensure different timestamps
        import time
        time.sleep(0.1)
    
    # Now test the API
    print(f"\n📊 Testing final API response...")
    
    response = requests.get('https://localhost:5000/api/concern/patient/TEST_PATIENT/severity-history', verify=False)
    
    if response.status_code == 200:
        data = response.json()
        tracking = data['severity_tracking']
        history = tracking['severity_history']
        
        print(f"   ✅ API Success!")
        print(f"   📊 History entries: {len(history)}")
        print(f"   📊 Total diagnoses: {tracking['total_diagnoses']}")
        print(f"   📊 Current risk: {tracking['current_risk_level']}")
        print(f"   📊 Cumulative severity: {tracking['cumulative_severity']:.3f}")
        
        if len(history) > 0:
            print(f"\n📈 Complete severity progression:")
            for i, entry in enumerate(history):
                timestamp = entry['timestamp'][:19].replace('T', ' ')
                severity = entry.get('severity', 0)
                risk = entry.get('risk_level', 'unknown')
                total_dx = entry.get('total_diagnoses', 0)
                print(f"      {i+1}. {timestamp}: {severity:.3f} severity")
                print(f"         Risk: {risk}, Total Diagnoses: {total_dx}")
            
            print(f"\n🎯 Frontend ConcernTrendChart compatibility:")
            print(f"   ✅ {len(history)} data points ready for graph")
            print(f"   📊 Severity range: {min(e.get('severity', 0) for e in history):.3f} - {max(e.get('severity', 0) for e in history):.3f}")
            print(f"   📊 Risk levels: {set(e.get('risk_level', 'unknown') for e in history)}")
            
            # Show example frontend data structure
            frontend_sample = {
                'score': history[-1].get('severity', 0),
                'level': history[-1].get('risk_level', 'unknown'), 
                'timestamp': history[-1]['timestamp']
            }
            print(f"   📊 Sample data point: {frontend_sample}")
            
            print(f"\n✅ CONCERN system is ready for frontend graph integration!")
            print(f"   🔗 The ConcernTrendChart component can now display this data")
            print(f"   📊 PatientDashboard will show the severity progression chart")
            
        else:
            print(f"   ❌ No history entries found in API response")
    else:
        print(f"   ❌ API failed: {response.status_code}")

if __name__ == "__main__":
    build_history()
