from flask import Flask, jsonify, request, render_template, send_from_directory, url_for
from flask_cors import CORS
import os
import sys
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import json

# Initialize Flask app with template and static folders
app = Flask(__name__,
            static_folder=os.path.join('app', 'static'),
            template_folder=os.path.join('app', 'templates'))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Enable CORS for all routes
CORS(app, resources={
    r"/api/*": {"origins": "*"}
})

# Configuration
app.config['UPLOAD_FOLDER'] = os.path.join('app', 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_model():
    """Load the machine learning model"""
    try:
        model_path = os.path.join('app', 'models', 'heart_disease_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Load the model when the app starts
model = load_model()

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Handle prediction requests"""
    logger.info("\n" + "="*50)
    logger.info(f"New {request.method} request to /api/predict")
    
    if request.method == 'OPTIONS':
        logger.info("Handling OPTIONS preflight request")
        response = jsonify({'status': 'preflight'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        if not model:
            error_msg = 'Prediction model not loaded. Please contact support.'
            logger.error(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500

        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No input data provided'
            }), 400

        # [Rest of your predict function remains the same...]
        # ... existing predict function code ...

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Error processing your request',
            'details': str(e)
        }), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('app/static', filename)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors by serving the main page"""
    return render_template('index.html'), 200

# This is required for gunicorn
application = app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting server on port {port}...")
    logger.info(f"Model loaded: {model is not None}")
    app.run(host='0.0.0.0', port=port, debug=False)