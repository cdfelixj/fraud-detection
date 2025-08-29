import numpy as np
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from src.models.isolation_forest import IsolationForestModel
from src.models.lstm_model import LSTMModel
from src.models.ensemble import EnsembleModel
from src.database.postgres_handler import PostgresHandler
from src.database.redis_cache import RedisCache
from src.utils.config import DATABASE_CONFIG, REDIS_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)

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
    # Check API status (if we're responding, API is up)
    api_status = True
    
    # Check database status
    database_status = False
    if postgres_handler:
        try:
            database_status = postgres_handler.health_check()
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
    
    # Check cache status
    cache_status = False
    if redis_cache:
        try:
            cache_status = redis_cache.health_check()
        except Exception as e:
            logger.warning(f"Cache health check failed: {e}")
    
    return jsonify({
        'api': api_status,
        'database': database_status,
        'cache': cache_status,
        'timestamp': datetime.utcnow().isoformat(),
        'models_loaded': ensemble_model.get_model_count() if ensemble_model else 0
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
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Cache result if Redis is available
        if redis_cache and 'transaction_id' in data:
            redis_cache.set(f"fraud_result:{data['transaction_id']}", result, ex=3600)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@api.route('/stats', methods=['GET'])
@cross_origin()
def get_stats():
    """Get dashboard statistics endpoint"""
    try:
        # Try to get real stats from database if available
        stats = {}
        if postgres_handler:
            try:
                db_stats = postgres_handler.get_fraud_statistics()
                stats.update(db_stats)
            except Exception as e:
                logger.warning(f"Failed to get database stats: {e}")
        
        # Fill in missing stats with mock data for demo
        from datetime import datetime, timedelta
        import random
        
        current_time = datetime.utcnow()
        mock_stats = {
            'totalTransactions': stats.get('total_transactions', random.randint(10000, 50000)),
            'fraudTransactions': stats.get('fraud_transactions', random.randint(50, 200)),
            'avgFraudProbability': stats.get('avg_fraud_probability', round(random.uniform(0.1, 0.5), 3)),
            'highRiskTransactions': stats.get('high_risk_transactions', random.randint(20, 100)),
            'fraudRate': round((stats.get('fraud_transactions', 100) / max(1, stats.get('total_transactions', 10000))) * 100, 2),
            'totalAmount': round(random.uniform(1000000, 5000000), 2),
            'avgTransactionAmount': round(random.uniform(50, 500), 2),
            'modelsActive': ensemble_model.get_model_count() if ensemble_model else 2,
            'systemHealth': 'healthy',
            'lastUpdated': current_time.isoformat(),
            'hourlyData': []
        }
        
        # Generate hourly data for charts
        for i in range(24):
            hour_time = current_time - timedelta(hours=23-i)
            mock_stats['hourlyData'].append({
                'hour': hour_time.strftime('%H:00'),
                'transactions': random.randint(400, 800),
                'fraudulent': random.randint(2, 10),
                'amount': round(random.uniform(40000, 80000), 2)
            })
        
        return jsonify(mock_stats)
        
    except Exception as e:
        logger.error(f"Stats generation failed: {e}")
        return jsonify({'error': f'Stats generation failed: {str(e)}'}), 500

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