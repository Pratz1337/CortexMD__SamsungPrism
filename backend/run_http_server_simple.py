#!/usr/bin/env python3
"""
HTTP Server launcher for CortexMD Backend
For use when SSL certificates are causing issues
"""

import os
import sys
from core.app import app

def main():
    """Run the Flask app with HTTP support"""
    
    print("="*60)
    print("🌐 CortexMD Backend HTTP Server")
    print("="*60)
    print(f"🌐 HTTP Server: http://0.0.0.0:5000")
    print(f"🌍 External Access: http://34.133.136.240:5000")
    print("="*60)
    print("⚠️  WARNING: Running without SSL encryption")
    print("   For production, use HTTPS with valid certificates")
    print("="*60)
    print("Press CTRL+C to stop the server")
    print("="*60)
    
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    except Exception as e:
        print(f"ERROR: Failed to start HTTP server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()