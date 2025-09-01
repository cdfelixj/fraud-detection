"""
Kafka Consumer for Real-time Fraud Detection
Consumes transaction messages and processes them for fraud detection
"""
import json
import logging
import time
import signal
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from kafka import KafkaConsumer
from kafka.errors import KafkaError

# Import our application components
from app import app, db
from models import Transaction, Prediction, FraudAlert
from ml_models import FraudDetectionModels
from kafka_config import kafka_manager, KafkaConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FraudDetectionConsumer:
    """Kafka consumer for real-time fraud detection"""
    
    def __init__(self):
        self.config = KafkaConfig()
        self.ml_models = FraudDetectionModels()
        self.consumer = None
        self.running = False
        
    def start_consuming(self):
        """Start consuming messages from Kafka"""
        try:
            # Create consumer
            self.consumer = kafka_manager.get_consumer(
                group_id='fraud-detection-processor',
                topics=[self.config.topics['transactions']]
            )
            
            logger.info(f"Started consuming from topic: {self.config.topics['transactions']}")
            self.running = True
            
            # Load ML models
            with app.app_context():
                self.ml_models.load_models()
                if not self.ml_models.is_trained:
                    logger.warning("ML models not trained. Predictions may be inaccurate.")
            
            # Start consuming loop
            self._consume_loop()
            
        except Exception as e:
            logger.error(f"Error starting consumer: {e}")
            raise
    
    def _consume_loop(self):
        """Main consumption loop"""
        try:
            for message in self.consumer:
                if not self.running:
                    break
                    
                try:
                    # Process the transaction message
                    self._process_transaction_message(message)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Consumer loop error: {e}")
        finally:
            self._cleanup()
    
    def _process_transaction_message(self, message):
        """Process a single transaction message"""
        try:
            transaction_data = message.value
            logger.debug(f"Processing transaction: {transaction_data.get('transaction_id', 'unknown')}")
            
            # Validate message format
            if not self._validate_transaction_message(transaction_data):
                logger.warning(f"Invalid transaction message format: {transaction_data}")
                return
            
            with app.app_context():
                # Create transaction record
                transaction = self._create_transaction_record(transaction_data)
                if not transaction:
                    return
                
                # Check if predictions are enabled for this transaction
                enable_predictions = transaction_data.get('enable_predictions', True)  # Default to True for backward compatibility
                
                prediction_result = None
                if enable_predictions:
                    # Make fraud prediction
                    prediction_result = self._make_fraud_prediction(transaction, transaction_data)
                    
                    # Handle fraud alerts
                    if prediction_result and prediction_result['prediction'] == 1:
                        self._create_fraud_alert(transaction, prediction_result)
                    
                    # Publish prediction result to Kafka
                    self._publish_prediction_result(transaction_data, prediction_result)
                else:
                    logger.debug(f"Skipping prediction for transaction {transaction_data.get('transaction_id')} - predictions disabled")
                
        except Exception as e:
            logger.error(f"Error processing transaction message: {e}")
    
    def _validate_transaction_message(self, data: Dict[str, Any]) -> bool:
        """Validate transaction message format"""
        required_fields = ['transaction_id', 'time_feature', 'amount']
        v_fields = [f'v{i}' for i in range(1, 29)]
        required_fields.extend(v_fields)
        
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return False
                
        return True
    
    def _create_transaction_record(self, data: Dict[str, Any]) -> Optional[Transaction]:
        """Create transaction record in database"""
        try:
            transaction = Transaction()
            transaction.time_feature = float(data['time_feature'])
            transaction.amount = float(data['amount'])
            
            # Set V1-V28 features
            for i in range(1, 29):
                setattr(transaction, f'v{i}', float(data.get(f'v{i}', 0)))
            
            # Set actual class (required field)
            transaction.actual_class = int(data.get('actual_class', 0))  # Default to 0 (normal) if not provided
            
            # Set metadata
            transaction.created_at = datetime.utcnow()
            
            db.session.add(transaction)
            db.session.commit()
            
            logger.debug(f"Transaction {transaction.id} saved to database")
            return transaction
            
        except Exception as e:
            logger.error(f"Error creating transaction record: {e}")
            db.session.rollback()
            return None
    
    def _make_fraud_prediction(self, transaction: Transaction, original_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make fraud prediction on transaction"""
        try:
            if not self.ml_models.is_trained:
                logger.warning("Models not trained, skipping prediction")
                return None
            
            # Prepare feature vector
            features = [
                transaction.time_feature,
                *[getattr(transaction, f'v{i}') for i in range(1, 29)],
                transaction.amount
            ]
            
            # Scale features using the trained scaler
            import numpy as np
            features_array = np.array([features])  # Convert to 2D array for prediction
            
            # Apply scaling - the models were trained on scaled data
            if hasattr(self.ml_models, 'data_processor') and self.ml_models.data_processor and hasattr(self.ml_models.data_processor.scaler, "scale_"):
                features_scaled = self.ml_models.data_processor.scaler.transform(features_array)
            else:
                # Load data processor with scaler if not available
                from data_processor import DataProcessor
                data_proc = DataProcessor()
                data_proc.load_scaler()
                features_scaled = data_proc.scaler.transform(features_array)
            
            prediction_results = self.ml_models.ensemble_predict(features_scaled)
            
            if prediction_results is None:
                logger.error("Failed to get prediction results")
                return None
                
            # Extract results
            prediction = int(prediction_results['final_predictions'][0])
            confidence = float(prediction_results['confidence_scores'][0])
            model_scores = {
                'isolation_forest': float(prediction_results['isolation_scores'][0]),
                'logistic_regression': float(prediction_results['logistic_probabilities'][0]),
                'ensemble': float(prediction_results['ensemble_scores'][0])
            }
            
            # Save prediction record - using SQLAlchemy model assignment
            pred_record = Prediction()
            pred_record.transaction_id = transaction.id
            pred_record.isolation_forest_score = float(model_scores.get('isolation_forest', 0))
            pred_record.ensemble_prediction = float(confidence)
            pred_record.final_prediction = prediction
            pred_record.confidence_score = float(confidence)
            pred_record.model_version = 'kafka_v1.0'
            pred_record.prediction_time = datetime.utcnow()
            
            db.session.add(pred_record)
            db.session.commit()
            
            result = {
                'transaction_id': original_data['transaction_id'],
                'prediction': prediction,
                'confidence': float(confidence),
                'model_scores': model_scores,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Fraud prediction: {original_data['transaction_id']} -> {prediction} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error making fraud prediction: {e}")
            db.session.rollback()
            return None
    
    def _create_fraud_alert(self, transaction: Transaction, prediction_result: Dict[str, Any]):
        """Create fraud alert for high-risk transactions"""
        try:
            alert = FraudAlert()
            alert.transaction_id = transaction.id
            alert.alert_level = 'HIGH' if prediction_result['confidence'] > 0.8 else 'MEDIUM'
            alert.alert_reason = f"Real-time fraud detection: confidence {prediction_result['confidence']:.3f}"
            alert.created_at = datetime.utcnow()
            alert.acknowledged = False
            
            db.session.add(alert)
            db.session.commit()
            
            # Publish alert to alerts topic
            alert_data = {
                'alert_id': alert.id,
                'transaction_id': prediction_result['transaction_id'],
                'confidence': prediction_result['confidence'],
                'alert_type': 'fraud_detected',
                'timestamp': datetime.utcnow().isoformat(),
                'amount': float(transaction.amount)
            }
            
            kafka_manager.send_message(
                topic=self.config.topics['alerts'],
                value=alert_data,
                key=str(alert.id)
            )
            
            logger.warning(f"FRAUD ALERT: Transaction {prediction_result['transaction_id']} flagged with confidence {prediction_result['confidence']:.3f}")
            
        except Exception as e:
            logger.error(f"Error creating fraud alert: {e}")
            db.session.rollback()
    
    def _publish_prediction_result(self, transaction_data: Dict[str, Any], prediction_result: Optional[Dict[str, Any]]):
        """Publish prediction result to Kafka"""
        if not prediction_result:
            return
            
        try:
            kafka_manager.send_message(
                topic=self.config.topics['predictions'],
                value=prediction_result,
                key=transaction_data['transaction_id']
            )
            
            logger.debug(f"Prediction result published for transaction {transaction_data['transaction_id']}")
            
        except Exception as e:
            logger.error(f"Error publishing prediction result: {e}")
    
    def _cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    global consumer
    if consumer:
        consumer.running = False
    sys.exit(0)

# Global consumer instance
consumer = None

def main():
    """Main consumer process"""
    global consumer
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for Kafka to be ready
    logger.info("Waiting for Kafka to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            if kafka_manager.health_check():
                logger.info("Kafka is ready!")
                break
        except Exception as e:
            logger.warning(f"Kafka not ready (attempt {i+1}/{max_retries}): {e}")
            time.sleep(2)
    else:
        logger.error("Kafka not available after maximum retries")
        sys.exit(1)
    
    # Create topics if they don't exist
    from kafka_config import create_topics_if_not_exist
    create_topics_if_not_exist()
    
    # Start consumer
    try:
        consumer = FraudDetectionConsumer()
        logger.info("Starting Kafka consumer for fraud detection...")
        consumer.start_consuming()
    except Exception as e:
        logger.error(f"Consumer failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
