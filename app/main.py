import os
import sys
from flask import Flask, render_template, request, jsonify

# Include the src directory in the python path for importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from predict import predict_loan_status, load_model_and_scaler
except ImportError:
    # Fallback to absolute/relative paths if sys.path structure differs
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from predict import predict_loan_status, load_model_and_scaler

app = Flask(__name__)

# Configure models directory relative to this file
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))

@app.route('/')
def home():
    """
    Renders the interactive web form and checks if the model exists.
    """
    model_exists = os.path.exists(os.path.join(MODELS_DIR, 'best_model.pkl'))
    model_info = None
    
    if model_exists:
        try:
            model_metadata, _ = load_model_and_scaler(MODELS_DIR)
            model_info = {
                'name': model_metadata['model_name'],
                'accuracy': model_metadata['metrics']['Accuracy'],
                'f1': model_metadata['metrics']['F1-score']
            }
        except Exception as e:
            print(f"Error loading model info: {e}")
            
    return render_template('index.html', model_exists=model_exists, model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint that accepts form or JSON inputs and performs prediction.
    """
    try:
        # Support both form data and JSON data inputs
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()
            
        # Parse inputs
        raw_input = {
            'no_of_dependents': int(data.get('no_of_dependents', 0)),
            'education': data.get('education', 'Graduate'),
            'self_employed': data.get('self_employed', 'No'),
            'income_annum': float(data.get('income_annum', 0.0)),
            'loan_amount': float(data.get('loan_amount', 0.0)),
            'loan_term': int(data.get('loan_term', 12)),
            'cibil_score': int(data.get('cibil_score', 300)),
            'residential_assets_value': float(data.get('residential_assets_value', 0.0)),
            'commercial_assets_value': float(data.get('commercial_assets_value', 0.0)),
            'luxury_assets_value': float(data.get('luxury_assets_value', 0.0)),
            'bank_asset_value': float(data.get('bank_asset_value', 0.0))
        }
        
        # Run prediction
        result = predict_loan_status(raw_input, MODELS_DIR)
        
        # Return results as JSON
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'probability_approved': result['probability_approved'],
            'probability_rejected': result['probability_rejected'],
            'model_used': result['model_used']
        })
        
    except FileNotFoundError as fnf_err:
        return jsonify({
            'success': False,
            'error': 'Model not trained yet. Run model training first.',
            'details': str(fnf_err)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'An error occurred during prediction.',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    # Run locally on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
