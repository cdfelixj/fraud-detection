import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from src.database.postgres_handler import PostgresHandler
from src.database.redis_cache import RedisCache
from src.utils.preprocessing import FraudDataPreprocessor
from src.utils.config import DATABASE_CONFIG, REDIS_CONFIG

logger = logging.getLogger(__name__)

class StreamProcessor:
    def __init__(self, kafka_consumer=None, model=None, database_handler=None, cache=None):
        self.kafka_consumer = kafka_consumer
        self.model = model
        self.database_handler = database_handler or self._init_database()
        self.cache = cache or self._init_cache()
        self.preprocessor = FraudDataPreprocessor()
        self.is_processing = False
        self.processed_count = 0
        self.fraud_detected_count = 0

    def _init_database(self):
        """Initialize database connection"""
        try:
            db_handler = PostgresHandler(DATABASE_CONFIG)
            if db_handler.connect():
                return db_handler
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
        return None

    def _init_cache(self):
        """Initialize Redis cache connection"""
        try:
            cache = RedisCache(**REDIS_CONFIG)
            if cache.connect():
                return cache
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
        return None

    def start_processing(self):
        """Start the stream processing pipeline"""
        if not self.kafka_consumer:
            logger.error("No Kafka consumer provided")
            return False
        
        if not self.model:
            logger.error("No model provided")
            return False

        self.is_processing = True
        logger.info("Starting stream processing...")

        try:
            # Set up message processor for the consumer
            self.kafka_consumer.message_processor = self.process_message
            
            # Start consuming messages
            self.kafka_consumer.start_consuming(async_mode=False)
            
        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
            self.is_processing = False
            return False

        return True

    def stop_processing(self):
        """Stop the stream processing pipeline"""
        self.is_processing = False
        if self.kafka_consumer:
            self.kafka_consumer.stop_consuming()
        logger.info("Stream processing stopped")

    def process_message(self, message):
        """Process a single Kafka message"""
        try:
            if not self.is_processing:
                return

            # Extract data from message
            data = self._extract_data(message)
            if not data:
                logger.warning("Failed to extract data from message")
                return

            # Analyze the transaction
            prediction_result = self.analyze(data)
            
            # Handle the prediction
            self._handle_prediction(data, prediction_result)
            
            self.processed_count += 1
            
            if prediction_result.get('is_fraud', False):
                self.fraud_detected_count += 1

            # Log processing stats periodically
            if self.processed_count % 100 == 0:
                logger.info(f"Processed {self.processed_count} transactions, detected {self.fraud_detected_count} potential frauds")

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze transaction data for fraud"""
        try:
            # Validate input data
            if not self._validate_transaction_data(data):
                return {
                    'is_fraud': False,
                    'fraud_probability': 0.0,
                    'error': 'Invalid transaction data'
                }

            # Prepare features for model
            features = self._prepare_features(data)
            
            if features is None:
                return {
                    'is_fraud': False,
                    'fraud_probability': 0.0,
                    'error': 'Failed to prepare features'
                }

            # Get model prediction
            if hasattr(self.model, 'predict_fraud_probability'):
                fraud_probability = self.model.predict_fraud_probability(features)
            elif hasattr(self.model, 'predict'):
                prediction = self.model.predict(features)
                # Convert prediction to probability
                if isinstance(prediction, (list, np.ndarray)):
                    fraud_probability = float(prediction[0]) if len(prediction) > 0 else 0.0
                else:
                    fraud_probability = float(prediction)
            else:
                logger.error("Model doesn't have predict method")
                return {
                    'is_fraud': False,
                    'fraud_probability': 0.0,
                    'error': 'Model prediction failed'
                }

            # Ensure probability is in valid range
            fraud_probability = max(0.0, min(1.0, fraud_probability))
            
            # Determine fraud status
            is_fraud = fraud_probability > 0.5
            
            # Calculate confidence score
            confidence = abs(fraud_probability - 0.5) * 2

            return {
                'is_fraud': is_fraud,
                'fraud_probability': fraud_probability,
                'confidence': confidence,
                'risk_level': self._get_risk_level(fraud_probability),
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': getattr(self.model, 'version', '1.0')
            }

        except Exception as e:
            logger.error(f"Error in fraud analysis: {e}")
            return {
                'is_fraud': False,
                'fraud_probability': 0.0,
                'error': f'Analysis failed: {str(e)}'
            }

    def _extract_data(self, message) -> Optional[Dict[str, Any]]:
        """Extract transaction data from Kafka message"""
        try:
            if hasattr(message, 'value'):
                data = message.value
            else:
                data = message

            if isinstance(data, str):
                data = json.loads(data)

            # Handle nested event structure
            if isinstance(data, dict):
                if 'transaction_data' in data:
                    return data['transaction_data']
                elif 'event_type' in data and data['event_type'] == 'fraud_detection_request':
                    return data.get('transaction_data', data)
                else:
                    return data

            return None

        except Exception as e:
            logger.error(f"Error extracting data from message: {e}")
            return None

    def _validate_transaction_data(self, data: Dict[str, Any]) -> bool:
        """Validate required fields in transaction data"""
        try:
            required_fields = ['amount']  # Minimum required field
            
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field: {field}")
                    return False

            # Validate amount is numeric and positive
            amount = data.get('amount')
            if not isinstance(amount, (int, float)) or amount < 0:
                logger.warning(f"Invalid amount value: {amount}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating transaction data: {e}")
            return False

    def _prepare_features(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare features for model prediction"""
        try:
            # Basic feature extraction
            features = []
            
            # Amount (log transformed to handle large values)
            amount = float(data.get('amount', 0))
            features.append(np.log1p(amount))
            
            # Hour of day (if timestamp available)
            if 'timestamp' in data:
                try:
                    timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                    features.append(timestamp.hour)
                    features.append(timestamp.weekday())
                except:
                    features.extend([12, 1])  # Default values
            else:
                features.extend([12, 1])  # Default hour and weekday
            
            # Transaction type (encoded)
            tx_type = data.get('transaction_type', 'unknown')
            type_encoding = {'credit': 1, 'debit': 2, 'transfer': 3, 'unknown': 0}
            features.append(type_encoding.get(tx_type, 0))
            
            # Merchant category (if available)
            merchant_category = data.get('merchant_category', 'unknown')
            category_encoding = {'retail': 1, 'restaurant': 2, 'gas': 3, 'online': 4, 'unknown': 0}
            features.append(category_encoding.get(merchant_category, 0))

            # Pad or truncate features to match model input size
            target_size = 5  # Adjust based on your model's expected input size
            if len(features) < target_size:
                features.extend([0] * (target_size - len(features)))
            else:
                features = features[:target_size]

            return np.array(features).reshape(1, -1)

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    def _get_risk_level(self, probability: float) -> str:
        """Determine risk level based on fraud probability"""
        if probability >= 0.8:
            return 'CRITICAL'
        elif probability >= 0.6:
            return 'HIGH'
        elif probability >= 0.4:
            return 'MEDIUM'
        elif probability >= 0.2:
            return 'LOW'
        else:
            return 'MINIMAL'

    def _handle_prediction(self, transaction_data: Dict[str, Any], prediction: Dict[str, Any]):
        """Handle fraud prediction results"""
        try:
            # Store in database if available
            if self.database_handler:
                self._store_transaction(transaction_data, prediction)

            # Cache result if available
            if self.cache:
                self._cache_result(transaction_data, prediction)

            # Generate alerts for high-risk transactions
            if prediction.get('is_fraud', False) or prediction.get('fraud_probability', 0) > 0.7:
                self._generate_alert(transaction_data, prediction)

            # Log significant events
            if prediction.get('is_fraud', False):
                logger.warning(f"FRAUD DETECTED - Transaction {transaction_data.get('transaction_id', 'unknown')}: "
                             f"probability={prediction.get('fraud_probability', 0):.3f}")

        except Exception as e:
            logger.error(f"Error handling prediction: {e}")

    def _store_transaction(self, transaction_data: Dict[str, Any], prediction: Dict[str, Any]):
        """Store transaction and prediction in database"""
        try:
            # Prepare transaction record
            transaction_record = {
                'transaction_id': transaction_data.get('transaction_id', f"tx_{datetime.utcnow().timestamp()}"),
                'amount': transaction_data.get('amount', 0),
                'timestamp': transaction_data.get('timestamp', datetime.utcnow().isoformat()),
                'merchant_id': transaction_data.get('merchant_id'),
                'user_id': transaction_data.get('user_id'),
                'is_fraud': prediction.get('is_fraud', False),
                'fraud_probability': prediction.get('fraud_probability', 0),
                'model_prediction': prediction.get('risk_level', 'UNKNOWN')
            }
            
            self.database_handler.insert_transaction(transaction_record)

        except Exception as e:
            logger.error(f"Error storing transaction: {e}")

    def _cache_result(self, transaction_data: Dict[str, Any], prediction: Dict[str, Any]):
        """Cache fraud detection result"""
        try:
            transaction_id = transaction_data.get('transaction_id')
            if transaction_id:
                result = {
                    'transaction_data': transaction_data,
                    'prediction': prediction,
                    'processed_at': datetime.utcnow().isoformat()
                }
                self.cache.cache_fraud_result(transaction_id, result, expiration=3600)

        except Exception as e:
            logger.error(f"Error caching result: {e}")

    def _generate_alert(self, transaction_data: Dict[str, Any], prediction: Dict[str, Any]):
        """Generate alert for high-risk transactions"""
        try:
            alert_data = {
                'type': 'fraud_alert',
                'transaction_id': transaction_data.get('transaction_id'),
                'fraud_probability': prediction.get('fraud_probability'),
                'risk_level': prediction.get('risk_level'),
                'timestamp': datetime.utcnow().isoformat(),
                'details': {
                    'amount': transaction_data.get('amount'),
                    'merchant': transaction_data.get('merchant_id'),
                    'user': transaction_data.get('user_id')
                }
            }
            
            # Here you would send to alert system, email, SMS, etc.
            logger.info(f"ALERT GENERATED: {json.dumps(alert_data, indent=2)}")

        except Exception as e:
            logger.error(f"Error generating alert: {e}")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'is_processing': self.is_processing,
            'processed_count': self.processed_count,
            'fraud_detected_count': self.fraud_detected_count,
            'fraud_rate': self.fraud_detected_count / max(1, self.processed_count) * 100,
            'database_connected': self.database_handler is not None and self.database_handler.health_check() if self.database_handler else False,
            'cache_connected': self.cache is not None and self.cache.health_check() if self.cache else False
        }