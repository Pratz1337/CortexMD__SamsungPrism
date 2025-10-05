#!/usr/bin/env python3
"""
Rebuild comprehensive severity history for TEST_PATIENT to test the graph
"""

from database_manager import get_database
import requests
import urllib3
from datetime import datetime, timedelta

urllib3.disable_warnings()

def rebuild_history():
    print("🔧 REBUILDING SEVERITY HISTORY FOR TESTING")
    print("=" * 60)
    
    db = get_database()
    
    # Clear existing history and start fresh
    print("\n📊 Clearing existing history for TEST_PATIENT...")
    
    with db.get_session() as session:
        from postgresql_database import ConcernSeverityTracking
        
        severity_record = session.query(ConcernSeverityTracking).filter_by(patient_id='TEST_PATIENT').first()
        
        if severity_record:
            # Reset the record
            severity_record.total_diagnoses = 0
            severity_record.cumulative_severity = 0.0
            severity_record.average_severity = 0.0
            severity_record.max_severity_reached = 0.0
            severity_record.severity_history = []
            session.commit()
            print("   ✅ Cleared existing history")
        else:
            print("   ❌ No severity record found")
            return
    
    # Simulate a realistic patient diagnosis progression over time
    diagnoses = [
        {
            'diagnosis_id': 'DIAG_001',
            'confidence': 0.65,
            'fol': 0.6,
            'enhanced': 0.5,
            'explain': 0.7,
            'imaging': False,
            'timestamp_offset': -72  # 3 days ago
        },
        {
            'diagnosis_id': 'DIAG_002', 
            'confidence': 0.78,
            'fol': 0.75,
            'enhanced': 0.65,
            'explain': 0.8,
            'imaging': True,
            'timestamp_offset': -48  # 2 days ago
        },
        {
            'diagnosis_id': 'DIAG_003',
            'confidence': 0.85,
            'fol': 0.88,
            'enhanced': 0.75,
            'explain': 0.82,
            'imaging': True,
            'timestamp_offset': -24  # 1 day ago  
        },
        {
            'diagnosis_id': 'DIAG_004',
            'confidence': 0.92,
            'fol': 0.9,
            'enhanced': 0.85,
            'explain': 0.88,
            'imaging': True,
            'timestamp_offset': -8   # 8 hours ago
        },
        {
            'diagnosis_id': 'DIAG_005',
            'confidence': 0.95,
            'fol': 0.95,
            'enhanced': 0.9,
            'explain': 0.9,
            'imaging': True,
            'timestamp_offset': -1   # 1 hour ago
        }
    ]
    
    print(f"\n📊 Adding {len(diagnoses)} diagnoses with realistic timeline...")
    
    for i, diag in enumerate(diagnoses):
        print(f"\n   Adding diagnosis {i+1}: {diag['diagnosis_id']}")
        print(f"      Confidence: {diag['confidence']:.2f}")
        print(f"      Timeline: {diag['timestamp_offset']} hours ago")
        
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
            print(f"      ✅ Risk: {result['risk_level']}, Score: {result['risk_score']:.3f}")
            print(f"      ✅ Cumulative: {result['cumulative_severity']:.3f}")
        else:
            print(f"      ❌ Failed to add diagnosis")
    
    # Test the API to see our graph data
    print(f"\n📊 Testing severity history API...")
    
    response = requests.get('https://localhost:5000/api/concern/patient/TEST_PATIENT/severity-history', verify=False)
    
    if response.status_code == 200:
        data = response.json()
        tracking = data['severity_tracking']
        history = tracking['severity_history']
        
        print(f"   ✅ API returned {len(history)} history entries")
        print(f"   📊 Current risk: {tracking['current_risk_level']}")
        print(f"   📊 Total diagnoses: {tracking['total_diagnoses']}")
        print(f"   📊 Cumulative severity: {tracking['cumulative_severity']:.3f}")
        
        print(f"\n📈 Severity progression (for frontend graph):")
        for i, entry in enumerate(history):
            timestamp = entry['timestamp'][:19].replace('T', ' ')
            severity = entry.get('severity', 0)
            risk = entry.get('risk_level', 'unknown')
            print(f"      {timestamp}: {severity:.3f} severity ({risk} risk)")
        
        # Format for frontend ConcernTrendChart
        frontend_data = []
        for entry in history:
            frontend_entry = {
                'score': entry.get('severity', 0),
                'level': entry.get('risk_level', 'low'),
                'timestamp': entry['timestamp']
            }
            frontend_data.append(frontend_entry)
        
        print(f"\n🎯 Frontend ConcernTrendChart data ready:")
        print(f"   📊 {len(frontend_data)} data points for graph")
        print(f"   📊 Score range: {min(d['score'] for d in frontend_data):.3f} - {max(d['score'] for d in frontend_data):.3f}")
        print(f"   📊 Risk levels: {set(d['level'] for d in frontend_data)}")
        
    else:
        print(f"   ❌ API failed: {response.status_code}")

if __name__ == "__main__":
    rebuild_history()
