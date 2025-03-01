import os
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Enable CORS for React frontend
CORS(app, origins=["http://localhost:3000"])

# Simple mock models for demonstration
class MockIsolationForest:
    def predict(self, X):
        # Return mock predictions
        return np.random.choice([-1, 1], size=len(X) if hasattr(X, '__len__') else 1)
    
    def predict_proba(self, X):
        # Return mock probabilities
        size = len(X) if hasattr(X, '__len__') else 1
        fraud_prob = np.random.uniform(0.1, 0.9, size)
        return np.column_stack([1 - fraud_prob, fraud_prob])

class MockLSTM:
    def predict(self, X):
        # Return mock predictions
        size = len(X) if hasattr(X, '__len__') else 1
        return np.random.uniform(0.0, 1.0, size)

class MockEnsemble:
    def __init__(self):
        self.models = [MockIsolationForest(), MockLSTM()]
    
    def predict_fraud_probability(self, X):
        # Simple average of model predictions
        if_pred = self.models[0].predict_proba(X)
        lstm_pred = self.models[1].predict(X)
        
        if hasattr(if_pred, '__len__') and len(if_pred) > 0:
            fraud_prob = if_pred[0][1] if len(if_pred[0]) > 1 else 0.5
        else:
            fraud_prob = 0.5
            
        return (fraud_prob + lstm_pred[0] if hasattr(lstm_pred, '__len__') else lstm_pred) / 2

# Initialize mock models
ensemble_model = MockEnsemble()

@app.route('/')
def root():
    return {
        'message': 'Fraud Detection System API',
        'version': '1.0.0',
        'status': 'running'
    }

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': str(np.datetime64('now')),
        'services': {
            'api': 'running',
            'models': 'loaded'
        }
    })

@app.route('/api/detect-fraud', methods=['POST'])
def detect_fraud():
    """Main fraud detection endpoint"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided in request'}), 400
        
        # Handle both formats: features array or transaction_data object
        if 'features' in data:
            features = np.array(data['features'])
        elif 'transaction_data' in data:
            # Convert transaction data to features
            transaction = data['transaction_data']
            features = convert_transaction_to_features(transaction)
        else:
            return jsonify({'error': 'Missing features or transaction_data in request'}), 400
        
        # Get predictions from ensemble
        fraud_probability = ensemble_model.predict_fraud_probability(features.reshape(1, -1))
        
        # Convert to scalar if single prediction
        if hasattr(fraud_probability, '__len__'):
            fraud_probability = float(fraud_probability[0]) if len(fraud_probability) > 0 else 0.5
        else:
            fraud_probability = float(fraud_probability)
        
        # Determine if fraud (threshold = 0.5)
        is_fraud = fraud_probability > 0.5
        confidence = fraud_probability if is_fraud else (1 - fraud_probability)
        
        return jsonify({
            'fraud_probability': fraud_probability,
            'is_fraud': is_fraud,
            'confidence': confidence,
            'risk_level': get_risk_level(fraud_probability),
            'timestamp': str(np.datetime64('now'))
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

def get_risk_level(probability):
    """Determine risk level based on fraud probability"""
    if probability > 0.7:
        return 'HIGH'
    elif probability > 0.3:
        return 'MEDIUM'
    else:
        return 'LOW'

def convert_transaction_to_features(transaction):
    """Convert transaction data to feature vector"""
    # Extract features from transaction
    amount = float(transaction.get('amount', 0))
    merchant = transaction.get('merchant', 'Unknown')
    
    # Create simple feature vector (in real system, this would be more sophisticated)
    features = [
        amount,  # Transaction amount
        len(merchant),  # Merchant name length
        hash(merchant) % 1000,  # Simple merchant encoding
        amount / 100.0,  # Normalized amount
        1 if amount > 500 else 0,  # High amount flag
    ]
    
    return np.array(features)

@app.route('/api/stats')
def get_stats():
    """Get mock statistics"""
    return jsonify({
        'total_transactions': 1247,
        'fraud_transactions': 23,
        'normal_transactions': 1224,
        'fraud_rate': 1.8,
        'avg_fraud_probability': 0.12
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
