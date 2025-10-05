#!/usr/bin/env python3
"""
Debug API response format to understand structure
"""

import requests
import json
import urllib3

urllib3.disable_warnings()

def debug_api():
    print("🔍 DEBUGGING API RESPONSE FORMAT")
    print("=" * 50)
    
    try:
        # Test the severity-history endpoint
        response = requests.get('https://localhost:5000/api/concern/patient/TEST_PATIENT/severity-history', verify=False)
        if response.status_code == 200:
            data = response.json()
            print("✅ Raw API Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ API failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_api()
