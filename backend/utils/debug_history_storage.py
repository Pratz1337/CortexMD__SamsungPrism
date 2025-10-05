#!/usr/bin/env python3
"""
Debug severity history storage and retrieval
"""

from database_manager import get_database
import json

def debug_history():
    print("🔍 DEBUGGING SEVERITY HISTORY STORAGE")
    print("=" * 50)
    
    db = get_database()
    
    # Get direct database access to the severity record
    with db.get_session() as session:
        from postgresql_database import ConcernSeverityTracking
        
        severity_record = session.query(ConcernSeverityTracking).filter_by(patient_id='TEST_PATIENT').first()
        
        if severity_record:
            print(f"✅ Found severity record for TEST_PATIENT")
            print(f"   Total diagnoses: {severity_record.total_diagnoses}")
            print(f"   Cumulative severity: {severity_record.cumulative_severity}")
            print(f"   Current risk: {severity_record.current_risk_level}")
            
            history = severity_record.severity_history
            print(f"   History entries in DB: {len(history) if history else 0}")
            
            if history:
                print("\n📊 Raw history from database:")
                for i, entry in enumerate(history):
                    print(f"   Entry {i+1}: {json.dumps(entry, indent=4)}")
            else:
                print("   ❌ No history entries found!")
        else:
            print("❌ No severity record found for TEST_PATIENT")
    
    # Test the API endpoint
    import requests
    import urllib3
    urllib3.disable_warnings()
    
    print(f"\n📊 Testing API endpoint...")
    response = requests.get('https://localhost:5000/api/concern/patient/TEST_PATIENT/severity-history', verify=False)
    
    if response.status_code == 200:
        data = response.json()
        history = data['severity_tracking']['severity_history']
        print(f"   API returned {len(history)} history entries")
        
        for i, entry in enumerate(history):
            print(f"   API Entry {i+1}: {json.dumps(entry, indent=4)}")
    else:
        print(f"   ❌ API failed: {response.status_code}")

if __name__ == "__main__":
    debug_history()
