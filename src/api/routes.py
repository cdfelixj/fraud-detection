import numpy as np
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from src.models.isolation_forest import IsolationForestModel
from src.models.lstm_model import LSTMModel
from src.models.ensemble import EnsembleModel
from src.database.postgres_handler import PostgresHandler
from src.database.redis_cache import RedisCache
from src.utils.config import DATABASE_CONFIG, REDIS_CONFIG, MODEL_CONFIG

api = Blueprint('api', __name__)

# Initialize models with proper configurations
try:
    isolation_forest_model = IsolationForestModel(
        contamination=MODEL_CONFIG['isolation_forest']['contamination'],
        n_estimators=MODEL_CONFIG['isolation_forest']['n_estimators']
    )
    
    lstm_model = LSTMModel(
        input_shape=(MODEL_CONFIG['lstm']['sequence_length'], MODEL_CONFIG['lstm']['features']),
        num_classes=2
    )
    
    # Create ensemble with both models
    ensemble_model = EnsembleModel()
    ensemble_model.add_model(isolation_forest_model, weight=0.6)
    ensemble_model.add_model(lstm_model, weight=0.4)
    
    postgres_handler = PostgresHandler(DATABASE_CONFIG)
    redis_cache = RedisCache(**REDIS_CONFIG)
    
except Exception as e:
    print(f"Warning: Could not initialize all components: {e}")
    # Create dummy instances for development
    isolation_forest_model = IsolationForestModel()
    lstm_model = LSTMModel()
    ensemble_model = EnsembleModel()
    postgres_handler = None
    redis_cache = None

@api.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': str(np.datetime64('now')),
        'models_loaded': ensemble_model.get_model_count()
    })

@api.route('/detect-fraud', methods=['POST'])
@cross_origin()
def detect_fraud():
    """Main fraud detection endpoint"""
    try:
        data = request.json
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        features = np.array(data['features'])
        
        # Get predictions from ensemble
        fraud_probability = ensemble_model.predict_fraud_probability(features)
        
        # Convert to scalar if single prediction
        if hasattr(fraud_probability, '__len__') and len(fraud_probability) == 1:
            fraud_probability = float(fraud_probability[0])
        elif not hasattr(fraud_probability, '__len__'):
            fraud_probability = float(fraud_probability)
        else:
            fraud_probability = fraud_probability.tolist()
        
        # Determine fraud status based on threshold
        threshold = 0.5
        is_fraud = fraud_probability > threshold if isinstance(fraud_probability, (int, float)) else any(p > threshold for p in fraud_probability)
        
        result = {
            'is_fraud': bool(is_fraud),
            'fraud_probability': fraud_probability,
            'confidence': abs(fraud_probability - 0.5) * 2 if isinstance(fraud_probability, (int, float)) else 0.5,
            'timestamp': str(np.datetime64('now'))
        }
        
        # Cache result if Redis is available
        if redis_cache and 'transaction_id' in data:
            redis_cache.set(f"fraud_result:{data['transaction_id']}", result, ex=3600)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@api.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    """Legacy prediction endpoint"""
    try:
        data = request.json
        features = data.get('features', [])
        
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        features = np.array(features)
        
        # Get individual model predictions
        isolation_result = isolation_forest_model.predict(features)
        lstm_result = lstm_model.predict(features)
        ensemble_result = ensemble_model.predict(features)
        
        return jsonify({
            'isolation_forest': isolation_result.tolist() if hasattr(isolation_result, 'tolist') else isolation_result,
            'lstm': lstm_result.tolist() if hasattr(lstm_result, 'tolist') else lstm_result,
            'ensemble': ensemble_result.tolist() if hasattr(ensemble_result, 'tolist') else ensemble_result
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@api.route('/cache', methods=['POST'])
@cross_origin()
def cache_data():
    """Cache data endpoint"""
    try:
        if not redis_cache:
            return jsonify({'error': 'Redis cache not available'}), 503
        
        data = request.json
        if not data or 'key' not in data or 'value' not in data:
            return jsonify({'error': 'Missing key or value'}), 400
        
        redis_cache.set(data['key'], data['value'])
        return jsonify({'status': 'success', 'message': 'Data cached successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Caching failed: {str(e)}'}), 500

@api.route('/cache/<key>', methods=['GET'])
@cross_origin()
def get_cached_data(key):
    """Get cached data endpoint"""
    try:
        if not redis_cache:
            return jsonify({'error': 'Redis cache not available'}), 503
        
        value = redis_cache.get(key)
        if value is None:
            return jsonify({'error': 'Key not found'}), 404
        
        return jsonify({'key': key, 'value': value})
        
    except Exception as e:
        return jsonify({'error': f'Cache retrieval failed: {str(e)}'}), 500

@api.route('/data', methods=['GET'])
@cross_origin()
def get_data():
    """Get data from database endpoint"""
    try:
        if not postgres_handler:
            return jsonify({'error': 'Database not available'}), 503
        
        query = request.args.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        result = postgres_handler.query(query)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Database query failed: {str(e)}'}), 500

def setup_routes(app):
    """Setup API routes"""
    app.register_blueprint(api, url_prefix='/api')