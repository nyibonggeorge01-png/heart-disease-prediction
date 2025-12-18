from flask import Blueprint, render_template, request, jsonify, current_app
import joblib
import os
import pandas as pd
import numpy as np

main = Blueprint('main', __name__)

# Load the trained model
def load_model():
    # Get the directory where the current script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from the current directory (from app/ to project root)
    project_root = os.path.dirname(current_dir)
    # Construct the full path to the model
    model_path = os.path.join(project_root, 'app', 'models', 'heart_disease_model.pkl')
    
    # Log the path for debugging
    print(f"Looking for model at: {model_path}")
    
    if not os.path.exists(model_path):
        # Try alternative path (relative to current file)
        alt_path = os.path.join(current_dir, 'models', 'heart_disease_model.pkl')
        print(f"Model not found at {model_path}. Trying alternative path: {alt_path}")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model not found at {model_path} or {alt_path}. Please train the model first.")
    
    print(f"Loading model from: {model_path}")
    return joblib.load(model_path)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    try:
        # Log request headers for debugging
        current_app.logger.info(f'Request headers: {dict(request.headers)}')
        
        # Get data from request
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Request must be JSON',
                'received_content_type': request.content_type
            }), 400
            
        data = request.get_json()
        
        # Log received data for debugging
        current_app.logger.info(f'Received data: {data}')
        print(f"\n{'='*50}\nReceived prediction request with data:")
        for key, value in data.items():
            print(f"{key}: {value} (type: {type(value)})")
        print("="*50 + "\n")
        
        # Validate input data
        required_fields = [
            'age', 'sex', 'cholesterol', 'cigarettes_per_day',
            'family_history', 'chest_pain_type', 'blood_sugar'
        ]
        
        for field in required_fields:
            if field not in data:
                error_msg = f'Missing required field: {field}'
                current_app.logger.error(error_msg)
                print(error_msg)
                return jsonify({
                    'status': 'error',
                    'message': error_msg
                }), 400
        
        # Map form data to model's expected format
        sex_mapping = {
            'male': 'MALE',
            'female': 'FEMALE',
            'other': 'OTHER'
        }
        
        print(f"Raw input data: {data}")
        
        # Create input data for prediction with correct column names
        family_history = data['family_history'].lower() in ['true', 'yes', '1', 't', 'y']
        input_dict = {
            'AGE': float(data['age']),
            'SEX': data['sex'].upper(),
            'CHOLESTEROL_MG_DL': float(data['cholesterol']),  # Changed from CHOLESTEROL_MG/DL
            'CIGARETTES_PER_DAY': float(data['cigarettes_per_day']),
            'FAMILY_HISTORY_OF_HEART_DISEASE': family_history,
            'CHEST_PAIN_TYPE': data['chest_pain_type'],
            'BLOOD_SUGAR_MG_DL': float(data['blood_sugar'])  # Changed from BLOOD_SUGER_MG/DL
        }
        
        print(f"Processed input data: {input_dict}")
        
        # Create DataFrame from the single row of data
        try:
            input_data = pd.DataFrame([input_dict])
            print(f"Input data for prediction: \n{input_data}")
            
            # Log the input data being sent to the model
            current_app.logger.info('Input data for prediction:')
            for col, val in input_data.iloc[0].items():
                current_app.logger.info(f"  {col}: {val} (type: {type(val)})")
            
            # Load model and make prediction
            try:
                model = load_model()
                current_app.logger.info("Model loaded successfully")
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}"
                current_app.logger.error(error_msg, exc_info=True)
                return jsonify({
                    'status': 'error',
                    'message': 'Error loading prediction model',
                    'details': str(e)
                }), 500
            
            # Log model features for debugging
            if hasattr(model, 'feature_names_in_'):
                print(f"Model expected features: {model.feature_names_in_}")
                current_app.logger.info(f'Model expected features: {model.feature_names_in_}')
            
            # Make prediction
            print("Making prediction...")
            prediction = model.predict(input_data)
            print(f"Raw prediction: {prediction}")
            
            # Get prediction probabilities
            confidence = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_data)[0]
                confidence = float(max(proba) * 100)
                print(f"Prediction probabilities: {proba}")
            
            result = {
                'status': 'success',
                'prediction': int(prediction[0]),
                'confidence': confidence
            }
            print(f"Prediction result: {result}")
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            print(error_msg)
            current_app.logger.error(error_msg, exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f"Error during prediction: {str(e)}"
            }), 500
        
        current_app.logger.info(f'Prediction result: {result}')
        return jsonify(result)
        
    except Exception as e:
        error_msg = f'Prediction error: {str(e)}'
        current_app.logger.error(error_msg, exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@main.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Heart Disease Prediction API is running'
    })
