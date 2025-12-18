import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

def load_and_prepare_data():
    try:
        # Load the dataset
        df = pd.read_excel("heart_disease_data.xlsx")
        
        # Print column names for debugging
        print("Original columns:", df.columns.tolist())
        
        # Clean column names (remove spaces, special characters, and make uppercase)
        df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        
        # Print cleaned column names
        print("Cleaned columns:", df.columns.tolist())
        
        # Drop rows with missing values
        df = df.dropna()
        
        # Print unique values in each column for debugging
        for col in df.columns:
            print(f"\nColumn: {col}")
            print("Unique values:", df[col].unique())
        
        # Define features and target
        X = df.drop('HEART_DISEASE', axis=1)
        y = df['HEART_DISEASE']
        
        # Convert target to numeric (True/False to 1/0)
        y = y.astype(int)
        
        return X, y
    except Exception as e:
        print(f"Error in load_and_prepare_data: {str(e)}")
        raise

def train_and_save_model():
    try:
        # Load and prepare data
        X, y = load_and_prepare_data()
        
        # Print data types of each column
        print("\nData types:")
        print(X.dtypes)
        
        # Define numeric and categorical features based on actual column names
        # Note: Using forward slashes in column names as they appear in the dataset
        numeric_features = ['AGE', 'CHOLESTEROL_MG/DL', 'CIGARETTES_PER_DAY', 'BLOOD_SUGER_MG/DL']
        categorical_features = ['SEX', 'FAMILY_HISTORY_OF_HEART_DISEASE', 'CHEST_PAIN_TYPE']
        
        # Verify all features exist in the dataframe
        all_features = numeric_features + categorical_features
        missing_features = [f for f in all_features if f not in X.columns]
        if missing_features:
            print(f"Warning: The following features are not found in the dataset: {missing_features}")
            print("Available features:", X.columns.tolist())
            return
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create and train the model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        ])
        
        # Train the model
        print("\nTraining model...")
        model.fit(X, y)
        print("Model training completed successfully!")
        
        # Create models directory if it doesn't exist
        os.makedirs('app/models', exist_ok=True)
        
        # Save the model
        model_path = os.path.join('app', 'models', 'heart_disease_model.pkl')
        joblib.dump(model, model_path)
        print(f"\nModel saved successfully to {os.path.abspath(model_path)}")
        
        # Verify the model file was created
        if os.path.exists(model_path):
            print(f"Model file exists at: {os.path.abspath(model_path)}")
            print(f"File size: {os.path.getsize(model_path) / 1024:.2f} KB")
        else:
            print("Error: Model file was not created!")
            
    except Exception as e:
        print(f"\nAn error occurred during model training/saving: {str(e)}")
        print("\nPlease check the following:")
        print("1. The Excel file 'Project datasheet2.xlsx' exists in the current directory")
        print("2. The Excel file is not open in another program")
        print("3. The Excel file contains the expected columns")
        print("4. There are no missing values in the dataset")
        print("\nError details:", str(e))

if __name__ == "__main__":
    print("Starting model training and saving process...")
    train_and_save_model()
