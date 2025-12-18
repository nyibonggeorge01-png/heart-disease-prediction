import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model():
    print("Loading and preprocessing data...")
    
    # Load the dataset
    try:
        df = pd.read_excel("heart_disease_data.xlsx")
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None
    
    # Standardize column names
    column_mapping = {
        'AGE': ['AGE'],
        'SEX': ['SEX'],
        'CHOLESTEROL_MG_DL': ['CHOLESTEROL_MG/DL', 'CHOLESTEROL (MG/DL)', 'CHOLESTEROL', 'CHOLESTEROL_MG_DL'],
        'CIGARETTES_PER_DAY': ['CIGARETTES_PER_DAY', 'CIGARETTES PER DAY', 'CIGARETTES'],
        'FAMILY_HISTORY_OF_HEART_DISEASE': ['FAMILY_HISTORY_OF_HEART_DISEASE', 'FAMILY HISTORY OF HEART DISEASE', 'FAMILY_HISTORY'],
        'CHEST_PAIN_TYPE': ['CHEST_PAIN_TYPE', 'CHEST PAIN TYPE', 'CHEST_PAIN'],
        'BLOOD_SUGAR_MG_DL': ['BLOOD_SUGAR_MG/DL', 'BLOOD SUGER (MG/DL)', 'BLOOD_SUGAR', 'BLOOD SUGAR', 'BLOOD_SUGAR_MG_DL'],
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
    
    # Set target column
    target_col = 'HEART_DISEASE'
    
    # Clean and convert target to binary (0/1)
    print("\nUnique values in target column before cleaning:", df[target_col].unique())
    
    # Handle NaN values in the target column
    if df[target_col].isna().any():
        print(f"Found {df[target_col].isna().sum()} missing values in the target column. They will be dropped.")
        df = df.dropna(subset=[target_col])
    
    # Convert target to boolean/int
    if df[target_col].dtype == 'object':
        # Handle string representations of booleans/ints
        df[target_col] = df[target_col].astype(str).str.upper().str.strip()
        df[target_col] = df[target_col].map({
            'TRUE': 1, '1': 1, 'YES': 1, 'Y': 1, 'T': 1,
            'FALSE': 0, '0': 0, 'NO': 0, 'N': 0, 'F': 0
        }).fillna(df[target_col])
        
    # Convert to int (this will raise an error if there are still non-numeric values)
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    if df[target_col].isna().any():
        print(f"Warning: {df[target_col].isna().sum()} values could not be converted to numeric in the target column.")
        print("These rows will be dropped.")
        df = df.dropna(subset=[target_col])
    
    df[target_col] = df[target_col].astype(int)
    
    # Ensure binary values (0 or 1)
    df[target_col] = df[target_col].apply(lambda x: 1 if x > 0 else 0)
    
    print("Unique values in target column after cleaning:", df[target_col].unique())
    
    # Define features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Define numeric and categorical features
    numeric_features = ['AGE', 'CHOLESTEROL_MG_DL', 'CIGARETTES_PER_DAY', 'BLOOD_SUGAR_MG_DL']
    categorical_features = ['SEX', 'FAMILY_HISTORY_OF_HEART_DISEASE', 'CHEST_PAIN_TYPE']
    
    # Verify all features exist
    all_features = numeric_features + categorical_features
    missing_features = [f for f in all_features if f not in X.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
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
    print("Training model...")
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nModel Evaluation:")
    print("-" * 50)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def save_model(model, filename='heart_disease_model.pkl'):
    """Save the trained model to a file."""
    # Ensure the models directory exists
    os.makedirs('app/models', exist_ok=True)
    
    # Save the model
    model_path = os.path.join('app', 'models', filename)
    joblib.dump(model, model_path)
    print(f"\nModel saved to {os.path.abspath(model_path)}")

if __name__ == "__main__":
    # Train the model
    model = train_model()
    
    if model is not None:
        # Save the model
        save_model(model)
        print("\nModel training and saving completed successfully!")
