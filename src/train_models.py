import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from data_preprocessing import preprocess_loan_data

# Resolve paths relative to THIS script file (src/), not the CWD
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

def train_and_evaluate_all(
    csv_path,
    models_dir=None,
    static_img_dir=None
):
    """
    Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost,
    evaluates them, logs results, automatically selects the best, and exports plots.
    """
    if models_dir is None:
        models_dir = os.path.join(_PROJECT_DIR, 'models')
    if static_img_dir is None:
        static_img_dir = os.path.join(_PROJECT_DIR, 'app', 'static', 'images')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_img_dir, exist_ok=True)
    
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    # 1. Preprocess and Split Data (using scaled features for general pipeline safety)
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw = preprocess_loan_data(
        csv_path, save_scaler_path=scaler_path
    )
    
    # 2. Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = {}
    trained_models = {}
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1,
            'ROC-AUC': roc_auc,
            'Confusion Matrix': cm,
            'Classification Report': report
        }
        trained_models[name] = model
        
        print(f"-> {name} Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
        
    # 3. Print Comparison Table
    print("\n=== Model Comparison Summary ===")
    comparison_df = pd.DataFrame({
        m_name: {
            'Accuracy': results[m_name]['Accuracy'],
            'Precision': results[m_name]['Precision'],
            'Recall': results[m_name]['Recall'],
            'F1-score': results[m_name]['F1-score'],
            'ROC-AUC': results[m_name]['ROC-AUC']
        } for m_name in results
    }).T
    print(comparison_df.to_string())
    
    # 4. Automatically select the best model (using F1-score as selection metric)
    best_model_name = max(results, key=lambda k: results[k]['F1-score'])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\n[BEST] Best performing model selected: {best_model_name} (F1-score: {best_metrics['F1-score']:.4f})")
    
    # Save the best model
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    model_metadata = {
        'model_name': best_model_name,
        'model': best_model,
        'features': X_train.columns.tolist(),
        'metrics': {
            'Accuracy': best_metrics['Accuracy'],
            'Precision': best_metrics['Precision'],
            'Recall': best_metrics['Recall'],
            'F1-score': best_metrics['F1-score'],
            'ROC-AUC': best_metrics['ROC-AUC']
        }
    }
    joblib.dump(model_metadata, best_model_path)
    print(f"Model metadata and best model saved successfully to {best_model_path}")
    
    # 5. Generate and save Confusion Matrix Plot for best model
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        best_metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues',
        xticklabels=['Approved', 'Rejected'], yticklabels=['Approved', 'Rejected']
    )
    plt.title(f"Confusion Matrix: {best_model_name}", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Actual Status', fontsize=12)
    plt.xlabel('Predicted Status', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(static_img_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_path}")
    
    # 6. Generate and save Feature Importance Plot
    plt.figure(figsize=(10, 6))
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_names = X_train.columns
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False)
        
        sns.barplot(data=feat_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
        plt.title(f"Feature Importances: {best_model_name}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
    elif best_model_name == 'Logistic Regression':
        # Use absolute value of coefficients
        coefficients = np.abs(best_model.coef_[0])
        feature_names = X_train.columns
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': coefficients})
        feat_df = feat_df.sort_values(by='Importance', ascending=False)
        
        sns.barplot(data=feat_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
        plt.title(f"Feature Coefficients (Abs Value): {best_model_name}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Absolute Coefficient Weight', fontsize=12)
        plt.ylabel('Features', fontsize=12)
    else:
        plt.text(0.5, 0.5, 'Feature Importance Not Available', ha='center', va='center', fontsize=14)
        plt.title(f"Feature Importances: {best_model_name}")
        
    plt.tight_layout()
    fi_path = os.path.join(static_img_dir, 'feature_importance.png')
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to {fi_path}")
    
    # Save a CSV summary of comparison
    comparison_df.to_csv(os.path.join(models_dir, 'model_comparison_metrics.csv'))
    
    return best_model_name, results

if __name__ == '__main__':
    csv_file = 'C:\\Users\\singh\\Downloads\\loan_approval_dataset.csv'
    train_and_evaluate_all(csv_file)
