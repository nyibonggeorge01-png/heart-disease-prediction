import joblib
import numpy as np
from pathlib import Path

def check_model_file(model_path):
    try:
        print(f"Checking model file: {model_path}")
        
        # Check if file exists
        if not Path(model_path).exists():
            print("Error: Model file not found")
            return False
            
        # Try to load the model
        try:
            model = joblib.load(model_path)
            print(f"Model loaded successfully. Type: {type(model)}")
            print(f"Is numpy array: {isinstance(model, np.ndarray)}")
            
            # Check if it's a scikit-learn model
            if hasattr(model, 'predict'):
                print("✅ Model has 'predict' method")
                return True
            else:
                print("❌ Model is missing 'predict' method")
                if isinstance(model, dict):
                    print("The model appears to be a dictionary. Keys:", list(model.keys()))
                return False
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    model_path = "app/models/heart_disease_model.pkl"
    check_model_file(model_path)
