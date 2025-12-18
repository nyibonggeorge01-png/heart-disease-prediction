
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
try:
    # Read the Excel file
    df = pd.read_excel("heart_disease_data.xlsx")
    print("Loaded dataset successfully!")
    print("\nDataset columns:", df.columns.tolist())
    print("\nFirst few rows of data:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Store original column names for reference
    original_columns = df.columns.tolist()
    print("\nOriginal column names:", original_columns)
    
    # Create a mapping of expected column names to possible variations
    column_mapping = {
        'AGE': ['AGE'],
        'SEX': ['SEX'],
        'CHOLESTEROL_MG_DL': ['CHOLESTEROL_MG/DL', 'CHOLESTEROL (MG/DL)', 'CHOLESTEROL'],
        'CIGARETTES_PER_DAY': ['CIGARETTES_PER_DAY', 'CIGARETTES PER DAY', 'CIGARETTES'],
        'FAMILY_HISTORY_OF_HEART_DISEASE': ['FAMILY_HISTORY_OF_HEART_DISEASE', 'FAMILY HISTORY OF HEART DISEASE', 'FAMILY_HISTORY'],
        'CHEST_PAIN_TYPE': ['CHEST_PAIN_TYPE', 'CHEST PAIN TYPE', 'CHEST_PAIN'],
        'BLOOD_SUGAR_MG_DL': ['BLOOD_SUGAR_MG/DL', 'BLOOD SUGER (MG/DL)', 'BLOOD SUGER (mg/dl)', 'BLOOD_SUGAR', 'BLOOD SUGAR', 'BLOOD SUGER'],
        'HEART_DISEASE': ['HEART_DISEASE', 'HEART DISEASE', 'TARGET']
    }
    
    # Standardize column names
    new_columns = []
    for col in df.columns:
        col_upper = col.upper().strip()
        mapped = False
        for std_name, variants in column_mapping.items():
            if col_upper in variants or col_upper.replace(' ', '_') in variants or col_upper.replace('_', ' ') in variants:
                new_columns.append(std_name)
                mapped = True
                break
        if not mapped:
            # If no mapping found, clean the column name and add it
            clean_name = col.upper().strip().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            new_columns.append(clean_name)
    
    # Apply the new column names
    df.columns = new_columns
    
    # Print cleaned column names for verification
    print("\nStandardized column names:", df.columns.tolist())
    
    # Set the target column name
    target_col = 'HEART_DISEASE'
    
    # Print the first few rows with the cleaned column names
    print("\nFirst few rows with cleaned column names:")
    print(df.head().to_string())
    
    # Check if all expected columns exist
    expected_columns = [
        'AGE', 'SEX', 'CHOLESTEROL_MG_DL', 'CIGARETTES_PER_DAY',
        'FAMILY_HISTORY_OF_HEART_DISEASE', 'CHEST_PAIN_TYPE',
        'BLOOD_SUGAR_MG_DL', 'HEART_DISEASE'
    ]
    
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Handle missing values
    print("\nDropping rows with missing values...")
    initial_rows = len(df)
    df = df.dropna()
    print(f"Dropped {initial_rows - len(df)} rows with missing values.")
    
    # Convert target to binary (0/1) if it's not already
    print("\nUnique values in target column:", df[target_col].unique())
    if df[target_col].dtype == 'object' or df[target_col].dtype == 'bool':
        df[target_col] = df[target_col].astype(int)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Define which columns are numeric and which are categorical
    numeric_features = ['AGE', 'CHOLESTEROL_MG_DL', 'CIGARETTES_PER_DAY', 'BLOOD_SUGAR_MG_DL']
    categorical_features = ['SEX', 'FAMILY_HISTORY_OF_HEART_DISEASE', 'CHEST_PAIN_TYPE']
    
    # Verify all features exist in the dataframe
    all_features = numeric_features + categorical_features
    missing_features = [f for f in all_features if f not in X.columns]
    if missing_features:
        available_columns = X.columns.tolist()
        error_msg = (
            f"The following features are missing from the dataframe: {missing_features}\n"
            f"Available columns: {available_columns}"
        )
        print(f"\nERROR: {error_msg}")
        
        # Try to find similar column names for better error reporting
        import difflib
        suggestions = {}
        for missing in missing_features:
            matches = difflib.get_close_matches(missing, available_columns, n=1, cutoff=0.6)
            if matches:
                suggestions[missing] = matches[0]
        
        if suggestions:
            print("\nDid you mean one of these columns?")
            for missing, suggestion in suggestions.items():
                print(f"  - {missing} -> {suggestion}")
        
        raise ValueError(error_msg)
    
    print("\nFeatures being used:")
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("\nFirst few rows of processed data:")
    print(X[all_features].head())
    
    # Print data types to help with debugging
    print("\nData types:")
    print(X[all_features].dtypes)
    
    # Print unique values in categorical columns
    print("\nUnique values in categorical columns:")
    for col in categorical_features:
        print(f"{col}: {X[col].unique().tolist()}")
    
    # Print basic statistics for numeric columns
    print("\nNumeric columns statistics:")
    print(X[numeric_features].describe())
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create and train the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    print("\nModel Evaluation:")
    print("-" * 50)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Example prediction with proper data formatting
    print("\nExample prediction for a new patient:")
    try:
        # Create a sample patient with the correct column names and data types
        new_patient = pd.DataFrame({
            'AGE': [58],
            'CHOLESTEROL_MG_DL': [250.0],
            'CIGARETTES_PER_DAY': [4.0],
            'BLOOD_SUGAR_MG_DL': [130.0],
            'SEX': ['MALE'],
            'FAMILY_HISTORY_OF_HEART_DISEASE': [True],  # Should be boolean
            'CHEST_PAIN_TYPE': ['typical angina']  # Must match one of the categories
        })
        
        # Ensure the column order matches the training data
        new_patient = new_patient[X.columns]
        
        # Make prediction
        prediction = model.predict(new_patient)
        proba = model.predict_proba(new_patient)[0]
        
        # Map prediction to human-readable output
        prediction_text = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
        confidence = max(proba) * 100
        
        print("\n=== Prediction Result ===")
        print(f"Prediction: {prediction_text}")
        print(f"Confidence: {confidence:.2f}%")
        print("\nFeature Importance (Top 5):")
        
        # Get feature importance if using a model that supports it
        if hasattr(model.named_steps['classifier'], 'coef_'):
            # Get feature names after one-hot encoding
            try:
                # Get the one-hot encoded feature names
                ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
                cat_features = ohe.get_feature_names_out(categorical_features)
                all_features = numeric_features + list(cat_features)
                
                # Get coefficients
                coef = model.named_steps['classifier'].coef_[0]
                
                # Create a DataFrame for feature importance
                feature_importance = pd.DataFrame({
                    'Feature': all_features,
                    'Importance': coef
                })
                
                # Sort by absolute importance and show top 5
                top_features = feature_importance.reindex(
                    feature_importance.Importance.abs().sort_values(ascending=False).index
                ).head()
                
                print(top_features.to_string(index=False))
                
            except Exception as e:
                print("Could not calculate feature importance:", str(e))
        
        print("\n=== New Patient Data ===")
        print(new_patient.to_string(index=False))
        
    except Exception as e:
        print("\nError making prediction. Please check the data format.")
        print(f"Error details: {str(e)}")
        print("\nExpected columns:", X.columns.tolist())
        print("\nExpected data types:")
        print(X.dtypes)
        
    except Exception as e:
        print("\nError making prediction. Please check if the new patient data matches the training data format.")
        print(f"Error details: {str(e)}")
    
except Exception as e:
    print(f"\nAn error occurred: {str(e)}")
    print("\nPlease make sure:")
    print("1. The Excel file 'heart_disease_data.xlsx' is in the same directory as this script")
    print("2. The file is not open in another program")
    print("3. The file contains the expected columns with appropriate data types")
