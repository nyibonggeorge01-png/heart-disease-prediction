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

def load_model():
    """Load the trained model with enhanced error handling and version compatibility."""
    try:
        # Define possible model paths
        possible_paths = [
            os.path.join('app', 'models', 'heart_disease_model.pkl'),
            os.path.join(os.path.dirname(__file__), 'models', 'heart_disease_model.pkl'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'heart_disease_model.pkl'),
            os.path.join(os.getcwd(), 'models', 'heart_disease_model.pkl'),
            'heart_disease_model.pkl'
        ]
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    logger.info(f"🔍 Attempting to load model from: {path}")
                    file_size = os.path.getsize(path)
                    logger.info(f"📦 Model file size: {file_size} bytes")
                    
                    # Try to load the model with specific joblib version
                    try:
                        import joblib
                        model = joblib.load(path)
                        logger.info(f"✅ Successfully loaded object from {path}")
                        logger.info(f"📊 Object type: {type(model).__name__}")
                        
                        # Check if it's a valid model
                        if not hasattr(model, 'predict'):
                            logger.warning("⚠️ Loaded object is not a valid scikit-learn model (missing predict method)")
                            
                            # If it's a dictionary, try to find the model in it
                            if isinstance(model, dict):
                                logger.info("🔍 Found dictionary, searching for model...")
                                for key, value in model.items():
                                    if hasattr(value, 'predict'):
                                        logger.info(f"✅ Found model in dictionary with key: {key}")
                                        model = value
                                        break
                            
                            if not hasattr(model, 'predict'):
                                raise ValueError(f"Object from {path} is not a valid model")
                        
                        # Log model details
                        if hasattr(model, 'named_steps'):
                            logger.info(f"⚙️ Model is a pipeline with steps: {list(model.named_steps.keys())}")
                        
                        logger.info(f"🚀 Successfully loaded model from: {path}")
                        logger.info(f"🔧 Model has predict method: {hasattr(model, 'predict')}")
                        return model
                        
                    except Exception as load_error:
                        logger.error(f"❌ Error loading model from {path}: {str(load_error)}", exc_info=True)
                        continue
                        
            except Exception as e:
                logger.error(f"❌ Error processing path {path}: {str(e)}")
                continue
        
        # If we get here, no model was successfully loaded
        error_msg = f"Could not load model from any of the paths: {possible_paths}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    except Exception as e:
        logger.error(f"❌ Critical error in load_model: {str(e)}", exc_info=True)
        raise

# Global variable to track if model is loaded
model_loaded = False
model = None

# Load the model when the module is imported
try:
    model = load_model()
    model_loaded = True
    logger.info("Model loaded successfully")
    logger.info(f"Model has predict method: {hasattr(model, 'predict')}")
    logger.info(f"Model has predict_proba method: {hasattr(model, 'predict_proba') if model_loaded else 'N/A'}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}", exc_info=True)
    model_loaded = False

@main.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@main.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Handle prediction requests with comprehensive error handling."""
    logger.info("\n" + "="*50)
    logger.info(f"New {request.method} request to /api/predict")
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        logger.info("Handling OPTIONS preflight request")
        response = jsonify({'status': 'preflight'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        # Log request details
        logger.info(f"Headers: {dict(request.headers)}")
        logger.info(f"Content-Type: {request.content_type}")
        
        # Get and validate input data
        if not request.is_json:
            error_msg = 'Request must be JSON'
            logger.error(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400

        data = request.get_json()
        logger.info(f"Parsed JSON data: {data}")
        
        if not data:
            error_msg = 'No input data provided'
            logger.warning(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400

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
                'missing_fields': missing_fields
            }), 400

        logger.info("All required fields present")
        
        # Check if model is loaded
        if not model_loaded or model is None:
            error_msg = 'Prediction model not loaded. Please contact support.'
            logger.error(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500

        # Process input data
        try:
            # Map chest pain types to standardized format
            chest_pain_map = {
                'typical angina': 'typical angina',
                'atypical angina': 'atypical angina',
                'non-anginal pain': 'non-anginal pain',
                'non-anginal': 'non-anginal pain',
                'asymptomatic': 'asymptomatic',
                'none': 'asymptomatic'
            }

            # Process chest pain type
            chest_pain = str(data['chest_pain_type']).strip().lower()
            chest_pain = chest_pain_map.get(chest_pain, 'asymptomatic')
            logger.info(f"Processed chest pain type: {chest_pain}")

            # Process other fields
            try:
                age = float(data['age'])
                cholesterol = float(data['cholesterol'])
                cigarettes_per_day = float(data['cigarettes_per_day'])
                blood_sugar = float(data['blood_sugar'])
                sex = 'MALE' if str(data['sex']).strip().lower() in ['male', 'm', '1'] else 'FEMALE'
                has_family_history = str(data['family_history']).lower() in ['true', 'yes', 'y', '1']
                
                # Create input DataFrame with expected column names
                input_data = pd.DataFrame([{
                    'AGE': age,
                    'CHOLESTEROL': cholesterol,
                    'CIGARETTES_PER_DAY': cigarettes_per_day,
                    'BLOOD_SUGAR': blood_sugar,
                    'SEX': sex,
                    'FAMILY_HISTORY': has_family_history,
                    'CHEST_PAIN_TYPE': chest_pain
                }])
                
                logger.info(f"Processed input data:\n{input_data.to_string()}")
                
            except (ValueError, TypeError) as e:
                error_msg = f"Invalid data format: {str(e)}"
                logger.error(error_msg)
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'details': 'Please check that all fields contain valid values.'
                }), 400

            # Make prediction
            logger.info("Making prediction...")
            try:
                # Get prediction
                prediction = model.predict(input_data)
                prediction = int(prediction[0])  # Convert numpy.int64 to Python int
                
                # Get probability if available
                probability = None
                if hasattr(model, 'predict_proba'):
                    probability = float(model.predict_proba(input_data)[0][1])
                
                logger.info(f"Prediction result - Class: {prediction}, Probability: {probability}")

                # Prepare response
                response_data = {
                    'success': True,
                    'prediction': prediction,
                    'probability': probability,
                    'risk_level': 'High' if prediction == 1 else 'Low',
                    'message': 'High risk of heart disease' if prediction == 1 else 'Low risk of heart disease',
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Prediction successful: {response_data}")
                response = jsonify(response_data)
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response

            except Exception as pred_error:
                error_msg = f"Prediction failed: {str(pred_error)}"
                logger.error(error_msg, exc_info=True)
                return jsonify({
                    'success': False,
                    'error': 'Error making prediction',
                    'details': str(pred_error) if current_app.debug else None
                }), 500

        except Exception as e:
            error_msg = f"Error processing input data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return jsonify({
                'success': False,
                'error': 'Error processing input data',
                'details': str(e) if current_app.debug else None
            }), 400

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.critical(error_msg, exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e) if current_app.debug else None
        }), 500

@main.route('/api/health')
def health_check():
    """Health check endpoint to verify service status."""
    model_info = {
        'loaded': model_loaded,
        'type': str(type(model).__name__) if model_loaded else 'None',
        'has_predict': hasattr(model, 'predict') if model_loaded else False,
        'has_predict_proba': hasattr(model, 'predict_proba') if model_loaded else False
    }
    
    return jsonify({
        'status': 'healthy',
        'model': model_info,
        'timestamp': datetime.now().isoformat()
    })