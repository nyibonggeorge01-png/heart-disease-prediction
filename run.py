import os
import sys
from flask import Flask, jsonify
from app import create_app

# Create necessary directories
os.makedirs('app/static/uploads', exist_ok=True)
os.makedirs('app/models', exist_ok=True)

app = create_app()

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Resource not found',
        'error': str(error)
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'An internal server error occurred',
        'error': str(error)
    }), 500 

@app.errorhandler(Exception)
def handle_exception(error):
    return jsonify({
        'status': 'error',
        'message': 'An unexpected error occurred',
        'error': str(error)
    }), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Heart Disease Prediction App")
    print("="*50)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Models directory exists: {os.path.exists('app/models')}")
    print("="*50 + "\n")
    print("Server running at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop")
    print("="*50 + "\n")
    
    try:
        app.run(host='127.0.0.1', port=5000, debug=True)
    except Exception as e:
        print(f"\nError starting server: {e}")