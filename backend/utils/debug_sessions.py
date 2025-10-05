#!/usr/bin/env python3
"""
Debug script to check diagnosis sessions
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from ..core.app import diagnosis_sessions
except ImportError:
    from core.app import diagnosis_sessions

print("=== DIAGNOSIS SESSIONS DEBUG ===")
print(f"Total sessions: {len(diagnosis_sessions)}")

for session_id, session_data in diagnosis_sessions.items():
    print(f"\nSession ID: {session_id}")
    print(f"Status: {session_data.get('status', 'unknown')}")
    print(f"Progress: {session_data.get('progress', 0)}")
    print(f"Created: {session_data.get('created_at', 'unknown')}")

    if session_data.get('status') == 'completed':
        print("✅ COMPLETED SESSION")
        print(f"Has diagnosis_result: {'diagnosis_result' in session_data}")
        print(f"Has patient_input: {'patient_input' in session_data}")
        print(f"Has fol_verification: {'fol_verification' in session_data}")

        # Check diagnosis result structure
        if 'diagnosis_result' in session_data:
            diag_result = session_data['diagnosis_result']
            if hasattr(diag_result, 'primary_diagnosis'):
                print(f"Primary diagnosis: {diag_result.primary_diagnosis}")
            else:
                print(f"Diagnosis result type: {type(diag_result)}")
                if isinstance(diag_result, dict):
                    print(f"Diagnosis result keys: {list(diag_result.keys())}")
    else:
        print(f"❌ NOT COMPLETED: {session_data.get('status')}")

print("\n=== END DEBUG ===")
