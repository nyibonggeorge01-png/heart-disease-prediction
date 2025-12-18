"""
WSGI config for Heart Disease Prediction App.

This module contains the WSGI application used by Flask's development server.
"""
import os
import sys

# Add the project directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the Flask app
from app import create_app

# Create the Flask application
application = create_app()

if __name__ == "__main__":
    # Run the development server
    application.run(host='0.0.0.0', port=8000, debug=True)
