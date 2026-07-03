import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def clean_data(df):
    """
    Cleans column names and categorical values by stripping leading/trailing whitespace.
    """
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    
    # Strip spaces from object/string column values
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    return df

def remove_outliers_iqr(df, column):
    """
    Clips outliers using the Interquartile Range (IQR) method.
    Matches the clipping logic of the reference notebook.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df[column] = df[column].clip(lower, upper)
    return df

def preprocess_loan_data(csv_path, save_scaler_path=None):
    """
    Loads, cleans, processes outliers, encodes categories, scales features,
    and returns splits for model training.
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Clean column names and categorical values
    df = clean_data(df)
    
    # Process outliers (clipping values outside 1.5 * IQR)
    num_cols = [
        'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
        'cibil_score', 'residential_assets_value', 'commercial_assets_value',
        'luxury_assets_value', 'bank_asset_value'
    ]
    for col in num_cols:
        df = remove_outliers_iqr(df, col)
        
    # Drop unneeded identifier column
    if 'loan_id' in df.columns:
        df = df.drop(columns=['loan_id'])
        
    # Encode categorical variables using consistent mapping
    # 0 = Graduate, 1 = Not Graduate
    df['education'] = df['education'].map({'Graduate': 0, 'Not Graduate': 1})
    # 0 = No, 1 = Yes
    df['self_employed'] = df['self_employed'].map({'No': 0, 'Yes': 1})
    # Target: 0 = Approved, 1 = Rejected
    df['loan_status'] = df['loan_status'].map({'Approved': 0, 'Rejected': 1})
    
    # Split into features (X) and target (y)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    # Train-test split (80-20, stratify on target, random_state=42 for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled features back to DataFrame with proper column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Save the fitted scaler if path is provided
    if save_scaler_path:
        os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
        joblib.dump(scaler, save_scaler_path)
        print(f"Scaler saved successfully to {save_scaler_path}")
        
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, X_train, X_test
