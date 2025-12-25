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

# Initialize Flask app with template and static folders
app = Flask(__name__,
            static_folder=os.path.join('app', 'static'),
            template_folder=os.path.join('app', 'templates'))

# Enable CORS for all routes
CORS(app, resources={
    r"/api/*": {"origins": "*"}
})

# Configuration
app.config['SECRET_KEY'] = 'heart-disease-prediction-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.join('app', 'static', 'uploads')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model (do this once at startup)
def load_model():
    try:
        model_path = os.path.join('app', 'models', 'heart_disease_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = joblib.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Load the model when the app starts
model = load_model()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    logger.info("\n" + "="*50)
    logger.info(f"New {request.method} request to /api/predict")
    logger.info(f"Headers: {dict(request.headers)}")
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        logger.info("Handling OPTIONS preflight request")
        response = jsonify({'status': 'preflight'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        logger.info(f"Incoming data type: {type(request.get_data())}")
        logger.info(f"Raw data: {request.get_data()}")
        
        if not model:
            error_msg = 'Prediction model not loaded. Please contact support.'
            logger.error(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500

        # Get and validate input data
        data = request.get_json()
        logger.info(f"Parsed JSON data: {data}")
        
        if not data:
            error_msg = 'No input data provided'
            logger.warning(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400

        app.logger.info(f'Received prediction request: {data}')

        # Validate required fields
        required_fields = ['age', 'sex', 'cholesterol', 'cigarettes_per_day', 
                         'family_history', 'chest_pain_type', 'blood_sugar']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            error_msg = f'Missing required fields: {", ".join(missing_fields)}'
            logger.warning(f"{error_msg}. Received fields: {list(data.keys())}")
            return jsonify({
                'success': False,
                'error': error_msg,
                'missing_fields': missing_fields,
                'received_fields': list(data.keys())
            }), 400
            
        logger.info("All required fields present")

        try:
            logger.info("Starting prediction processing")
            # Map chest pain types to numerical values
            chest_pain_map = {
                'typical angina': 0,
                'atypical angina': 1,
                'non-anginal pain': 2,
                'non-anginal': 2,  # Alternative format
                'asymptomatic': 3,
                'none': 3  # Alternative format
            }

            # Get chest pain type with case-insensitive matching
            chest_pain = str(data['chest_pain_type']).strip().lower()
            logger.info(f"Chest pain type: '{chest_pain}' (original: '{data['chest_pain_type']}')")
            
            if chest_pain not in chest_pain_map:
                logger.warning(f'Invalid chest pain type: {chest_pain}. Available types: {list(chest_pain_map.keys())}')
                chest_pain = 'asymptomatic'  # Default value
                logger.info(f'Using default chest pain type: {chest_pain}')

            # Prepare input features as a DataFrame with the correct column names
            try:
                # Create a DataFrame with the exact column names the model was trained on
                input_data = pd.DataFrame({
                    'AGE': [float(data['age'])],
                    'SEX': [1 if str(data['sex']).strip().lower() in ['male', 'm', '1'] else 0],
                    'CHOLESTEROL_MG_DL': [float(data['cholesterol'])],
                    'CIGARETTES_PER_DAY': [float(data['cigarettes_per_day'])],
                    'FAMILY_HISTORY_OF_HEART_DISEASE': [1 if str(data['family_history']).strip().lower() in ['true', 'yes', 'y', '1'] else 0],
                    'CHEST_PAIN_TYPE': [chest_pain_map[chest_pain]],
                    'BLOOD_SUGAR_MG_DL': [float(data['blood_sugar'])]
                })
                logger.info(f"Processed input data: {input_data.to_dict(orient='records')[0]}")
                logger.info(f"Input data columns: {input_data.columns.tolist()}")
                logger.info(f"Input data shape: {input_data.shape}")
                logger.info(f"Input data dtypes: {input_data.dtypes}")
            except (ValueError, KeyError) as ve:
                logger.error(f"Error processing input data: {str(ve)}", exc_info=True)
                return jsonify({
                    'success': False,
                    'error': f'Error processing input data: {str(ve)}',
                    'details': 'Please check that all fields contain valid values.'
                }), 400
            
            # Make prediction
            try:
                logger.info("Making prediction...")
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1]  # Probability of heart disease
                logger.info(f'Prediction result - Class: {prediction[0]}, Probability: {probability:.2f}')
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}", exc_info=True)
                return jsonify({
                    'success': False,
                    'error': 'Error making prediction',
                    'details': str(e)
                }), 500
            
            response_data = {
                'success': True,
                'prediction': int(prediction[0]),
                'probability': float(probability),
                'risk_level': 'High' if prediction[0] == 1 else 'Low',
                'message': 'High risk of heart disease detected' if prediction[0] == 1 
                          else 'Low risk of heart disease',
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"Sending response: {response_data}")
            
            response = jsonify(response_data)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
        except ValueError as ve:
            error_msg = f'Value error in prediction: {str(ve)}'
            logger.error(error_msg, exc_info=True)
            response = jsonify({
                'success': False,
                'error': f'Invalid input format: {str(ve)}',
                'details': 'Please check that all fields contain valid numbers.',
                'timestamp': datetime.now().isoformat()
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 400
            
        except Exception as e:
            error_msg = f'Unexpected error in prediction: {str(e)}'
            logger.error(error_msg, exc_info=True)
            response = jsonify({
                'success': False,
                'error': 'Error processing your request. Please try again.',
                'details': str(e) if app.debug else None,
                'timestamp': datetime.now().isoformat()
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 500
            

    except Exception as e:
        error_msg = f'Server error: {str(e)}'
        logger.critical(error_msg, exc_info=True)
        response = jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e) if app.debug else None,
            'timestamp': datetime.now().isoformat()
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

# Serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('app/static', filename)

# Health check endpoint
@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 200

# This is required for Vercel
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    print(f"Starting server on port {port}...")
    print(f"Model loaded: {model is not None}")
    app.run(host='0.0.0.0', port=port, debug=False)

# Vercel requires a callable application
api = app
