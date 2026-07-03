import os
import joblib
import pandas as pd
import numpy as np

# Resolve models directory relative to this predict.py file (in src/)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
_DEFAULT_MODELS_DIR = os.path.join(_PROJECT_DIR, 'models')

def load_model_and_scaler(models_dir=None):
    """
    Loads the saved model metadata and the fitted StandardScaler.
    """
    if models_dir is None:
        models_dir = _DEFAULT_MODELS_DIR
    model_path = os.path.join(models_dir, 'best_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train_models.py first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Run train_models.py first.")
        
    model_metadata = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model_metadata, scaler

def predict_loan_status(raw_input, models_dir=None):
    """
    Takes a dict of raw input values, preprocesses them, scales them,
    and returns a prediction (Approved/Rejected) and the confidence score.
    
    Expected raw_input keys:
        - no_of_dependents: int
        - education: str ('Graduate', 'Not Graduate')
        - self_employed: str ('Yes', 'No')
        - income_annum: float
        - loan_amount: float
        - loan_term: int (months/years, match dataset)
        - cibil_score: int
        - residential_assets_value: float
        - commercial_assets_value: float
        - luxury_assets_value: float
        - bank_asset_value: float
    """
    # Load assets
    model_metadata, scaler = load_model_and_scaler(models_dir)
    
    # 1. Map categoricals (matching our training encoding)
    education_map = {'Graduate': 0, 'Not Graduate': 1}
    self_employed_map = {'No': 0, 'Yes': 1}
    
    mapped_input = raw_input.copy()
    
    # Sanitize and convert strings
    edu_str = str(mapped_input.get('education', 'Graduate')).strip()
    emp_str = str(mapped_input.get('self_employed', 'No')).strip()
    
    mapped_input['education'] = education_map.get(edu_str, 0)
    mapped_input['self_employed'] = self_employed_map.get(emp_str, 0)
    
    # Convert numerical keys to floats/ints
    num_cols = [
        'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
        'cibil_score', 'residential_assets_value', 'commercial_assets_value',
        'luxury_assets_value', 'bank_asset_value'
    ]
    for col in num_cols:
        mapped_input[col] = float(mapped_input.get(col, 0.0))
        
    # Convert input to DataFrame with the exact features list used in training
    feature_names = model_metadata['features']
    input_df = pd.DataFrame([mapped_input])[feature_names]
    
    # 2. Scale features using the trained scaler
    scaled_features = scaler.transform(input_df)
    
    # 3. Model Inference
    model = model_metadata['model']
    prediction_class = model.predict(scaled_features)[0]  # 0 or 1
    
    # Calculate confidence score
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_features)[0]
        # Class 0 is Approved, Class 1 is Rejected
        # Confidence score is the probability of the predicted class
        confidence = float(probabilities[prediction_class])
    else:
        confidence = 1.0  # Fallback if model doesn't support probability
        
    # Map target index back to status string
    # target mapping: 0 -> Approved, 1 -> Rejected
    status_str = "Approved" if prediction_class == 0 else "Rejected"
    
    return {
        'prediction': status_str,
        'confidence': confidence,
        'probability_approved': float(probabilities[0]) if hasattr(model, "predict_proba") else (1.0 if prediction_class == 0 else 0.0),
        'probability_rejected': float(probabilities[1]) if hasattr(model, "predict_proba") else (1.0 if prediction_class == 1 else 0.0),
        'model_used': model_metadata['model_name']
    }
