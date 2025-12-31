from flask import Blueprint, render_template, request, jsonify, current_app
import joblib
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

main = Blueprint('main', __name__)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load the trained model
def load_model():
    try:
        # Try multiple possible model locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'models', 'heart_disease_model.pkl'),
            os.path.join(os.getcwd(), 'app', 'models', 'heart_disease_model.pkl'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app', 'models', 'heart_disease_model.pkl'),
            os.path.join(os.getcwd(), 'models', 'heart_disease_model.pkl')
        ]
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    logger.info(f"Attempting to load model from: {path}")
                    model = joblib.load(path)
                    logger.info(f"Successfully loaded model from: {path}")
                    return model
            except Exception as e:
                logger.warning(f"Failed to load model from {path}: {str(e)}")
                continue
        
        error_msg = f"Could not load model from any of the paths: {possible_paths}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise
# Global variable to track if model is loaded
model_loaded = False
model = None

# Load the model when the module is imported
try:
    model = load_model()
    model_loaded = True
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model_loaded = False

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/api/predict', methods=['POST', 'OPTIONS'])
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
        
        if not model_loaded or model is None:
            error_msg = 'Prediction model not loaded. Please contact support.'
            logger.error(error_msg)
            response = jsonify({
                'success': False,
                'error': error_msg
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 500

        # Get and validate input data
        data = request.get_json()
        logger.info(f"Parsed JSON data: {data}")
        
        if not data:
            error_msg = 'No input data provided'
            logger.warning(error_msg)
            response = jsonify({
                'success': False,
                'error': error_msg
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 400

        # Validate required fields
        required_fields = ['age', 'sex', 'cholesterol', 'cigarettes_per_day', 
                         'family_history', 'chest_pain_type', 'blood_sugar']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            error_msg = f'Missing required fields: {", ".join(missing_fields)}'
            logger.warning(f"{error_msg}. Received fields: {list(data.keys())}")
            response = jsonify({
                'success': False,
                'error': error_msg,
                'missing_fields': missing_fields,
                'received_fields': list(data.keys())
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 400

        logger.info("All required fields present")
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
        
        try:
            # Prepare the input data with proper types and structure
            sex = str(data['sex']).strip().lower()
            is_male = sex in ['male', 'm', '1']
            has_family_history = str(data['family_history']).strip().lower() in ['true', 'yes', 'y', '1']
            
            # Create DataFrame with the exact column names and structure expected by the model
            input_data = pd.DataFrame({
                'AGE': [float(data['age'])],
                'CHOLESTEROL_MG_DL': [float(data['cholesterol'])],
                'CIGARETTES_PER_DAY': [float(data['cigarettes_per_day'])],
                'BLOOD_SUGAR_MG_DL': [float(data['blood_sugar'])],
                'SEX': ['MALE' if is_male else 'FEMALE'],
                'FAMILY_HISTORY_OF_HEART_DISEASE': [str(has_family_history)],
                'CHEST_PAIN_TYPE': [chest_pain]
            })
            
            logger.info(f"Input data before preprocessing:\n{input_data}")
            logger.info(f"Data types before preprocessing:\n{input_data.dtypes}")
            
            logger.info(f"Processed input data: {input_data.to_dict(orient='records')[0]}")
            logger.info(f"Input data columns: {input_data.columns.tolist()}")
            logger.info(f"Input data shape: {input_data.shape}")
            logger.info(f"Input data dtypes: {input_data.dtypes}")
            
            # Log the prepared input data
            logger.info(f"Input data before prediction:\n{input_data}")
            
            # Make prediction
            logger.info("Making prediction...")
            try:
                # Get prediction and probability
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1]  # Probability of heart disease
                logger.info(f"Prediction successful. Class: {prediction[0]}, Probability: {probability:.4f}")
                
                # Get feature importances if available
                if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                    importances = model.named_steps['classifier'].feature_importances_
                    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                    logger.info("Feature importances:")
                    for name, importance in zip(feature_names, importances):
                        logger.info(f"  {name}: {importance:.4f}")
                        
            except Exception as pred_error:
                logger.error(f"Prediction failed: {str(pred_error)}", exc_info=True)
                if hasattr(model, 'named_steps'):
                    logger.info(f"Model steps: {list(model.named_steps.keys())}")
                    if 'preprocessor' in model.named_steps:
                        try:
                            logger.info("Preprocessor transformers:")
                            for name, transformer, columns in model.named_steps['preprocessor'].transformers_:
                                logger.info(f"  {name}: {columns}")
                        except Exception as e:
                            logger.error(f"Error getting preprocessor info: {e}")
                raise
            logger.info(f'Prediction result - Class: {prediction[0]}, Probability: {probability:.2f}')

            # Prepare response
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

        except (ValueError, KeyError) as ve:
            error_msg = f'Error processing input data: {str(ve)}'
            logger.error(error_msg, exc_info=True)
            response = jsonify({
                'success': False,
                'error': error_msg,
                'details': 'Please check that all fields contain valid values.',
                'timestamp': datetime.now().isoformat()
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 400
            
        except Exception as e:
            error_msg = f'Prediction error: {str(e)}'
            logger.error(error_msg, exc_info=True)
            response = jsonify({
                'success': False,
                'error': 'Error making prediction',
                'details': str(e) if current_app.debug else None,
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
            'details': str(e) if current_app.debug else None,
            'timestamp': datetime.now().isoformat()
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

@main.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'model_type': str(type(model)) if model is not None else 'None',
        'timestamp': datetime.now().isoformat()
    })
