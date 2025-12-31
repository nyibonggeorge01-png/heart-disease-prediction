import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    """Load and preprocess the heart disease dataset."""
    logger.info("Loading and preprocessing data...")
    
    # Try to find the data file in different locations
    data_paths = [
        "data/heart_disease_data.xlsx",
        "heart_disease_data.xlsx",
        os.path.join(os.path.dirname(__file__), "data", "heart_disease_data.xlsx")
    ]
    
    df = None
    for path in data_paths:
        try:
            logger.info(f"Trying to load data from: {path}")
            df = pd.read_excel(path)
            logger.info("Dataset loaded successfully!")
            break
        except Exception as e:
            logger.warning(f"Failed to load from {path}: {str(e)}")
    
    if df is None:
        raise FileNotFoundError("Could not find the heart disease data file.")
    
    # Standardize column names
    column_mapping = {
        'AGE': ['AGE'],
        'SEX': ['SEX'],
        'CHOLESTEROL': ['CHOLESTEROL_MG/DL', 'CHOLESTEROL (MG/DL)', 'CHOLESTEROL', 'CHOLESTEROL_MG_DL'],
        'CIGARETTES_PER_DAY': ['CIGARETTES_PER_DAY', 'CIGARETTES PER DAY', 'CIGARETTES'],
        'FAMILY_HISTORY': ['FAMILY_HISTORY_OF_HEART_DISEASE', 'FAMILY HISTORY OF HEART DISEASE', 'FAMILY_HISTORY'],
        'CHEST_PAIN_TYPE': ['CHEST_PAIN_TYPE', 'CHEST PAIN TYPE', 'CHEST_PAIN'],
        'BLOOD_SUGAR': ['BLOOD_SUGAR_MG/DL', 'BLOOD SUGER (MG/DL)', 'BLOOD_SUGAR', 'BLOOD SUGAR', 'BLOOD_SUGAR_MG_DL'],
        'HEART_DISEASE': ['HEART_DISEASE', 'HEART DISEASE', 'TARGET']
    }
    
    # Standardize column names
    new_columns = []
    for col in df.columns:
        col_upper = str(col).upper().strip()
        mapped = False
        for std_name, variants in column_mapping.items():
            if any(variant == col_upper or 
                   variant.replace(' ', '_') == col_upper.replace(' ', '_') or 
                   variant.replace('_', ' ') == col_upper.replace('_', ' ')
                   for variant in variants):
                new_columns.append(std_name)
                mapped = True
                break
        if not mapped:
            clean_name = col_upper.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            new_columns.append(clean_name)
    
    df.columns = new_columns
    
    # Ensure all required columns are present
    required_columns = ['AGE', 'SEX', 'CHOLESTEROL', 'CIGARETTES_PER_DAY', 
                       'FAMILY_HISTORY', 'CHEST_PAIN_TYPE', 'BLOOD_SUGAR', 'HEART_DISEASE']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert data types
    df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
    df['CHOLESTEROL'] = pd.to_numeric(df['CHOLESTEROL'], errors='coerce')
    df['CIGARETTES_PER_DAY'] = pd.to_numeric(df['CIGARETTES_PER_DAY'], errors='coerce')
    df['BLOOD_SUGAR'] = pd.to_numeric(df['BLOOD_SUGAR'], errors='coerce')
    
    # Handle missing values
    df = df.dropna()
    
    # Convert boolean columns to 0/1
    if 'FAMILY_HISTORY' in df.columns:
        df['FAMILY_HISTORY'] = df['FAMILY_HISTORY'].astype(int)
    
    # Ensure target variable is binary
    df['HEART_DISEASE'] = df['HEART_DISEASE'].astype(int)
    
    # Define features and target
    X = df[['AGE', 'SEX', 'CHOLESTEROL', 'CIGARETTES_PER_DAY', 
            'FAMILY_HISTORY', 'CHEST_PAIN_TYPE', 'BLOOD_SUGAR']]
    y = df['HEART_DISEASE']
    
    return X, y

def train_model(X, y):
    """Train a RandomForestClassifier with the given data."""
    logger.info("Training model...")
    
    # Define preprocessing for numeric and categorical features
    numeric_features = ['AGE', 'CHOLESTEROL', 'CIGARETTES_PER_DAY', 'BLOOD_SUGAR']
    categorical_features = ['SEX', 'FAMILY_HISTORY', 'CHEST_PAIN_TYPE']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    return model

def save_model(model, filename='heart_disease_model.pkl'):
    """Save the trained model to a file."""
    # Create models directory if it doesn't exist
    os.makedirs('app/models', exist_ok=True)
    
    # Save the model
    model_path = os.path.join('app', 'models', filename)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

def main():
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data()
        
        # Train model
        model = train_model(X, y)
        
        # Save model
        save_model(model)
        
        logger.info("Model retraining completed successfully!")
    except Exception as e:
        logger.error(f"Error during model retraining: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()