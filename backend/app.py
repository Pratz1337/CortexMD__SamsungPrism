#!/usr/bin/env python3
"""
CortexMD Backend Application Entry Point
Main Flask application with organized folder structure
"""

import sys
import os

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Import the main Flask application from core
from core.app import app

if __name__ == '__main__':
    # Development server configuration with HTTPS
    import os
    
    # Path to SSL certificates
    cert_path = os.path.join(os.path.dirname(__file__), 'ssl_certs', 'cert.pem')
    key_path = os.path.join(os.path.dirname(__file__), 'ssl_certs', 'key.pem')
    
    # Check if SSL certificates exist
    if os.path.exists(cert_path) and os.path.exists(key_path):
        print(f"Starting HTTPS server with SSL certificates...")
        print(f"Certificate: {cert_path}")
        print(f"Key: {key_path}")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            ssl_context=(cert_path, key_path)
        )
    else:
        print("SSL certificates not found. Starting HTTP server...")
        print("Note: Clients trying to connect via HTTPS will fail.")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False
        )