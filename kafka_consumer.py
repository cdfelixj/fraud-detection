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
    """Kafka consumer for real-time fraud detection with high throughput optimization"""
    
    def __init__(self, batch_size=50):
        self.config = KafkaConfig()
        self.ml_models = FraudDetectionModels()
        self.consumer = None
        self.running = False
        self.batch_size = batch_size
        
        # Performance monitoring
        self.stats = {
            'messages_processed': 0,
            'predictions_made': 0,
            'alerts_created': 0,
            'start_time': None,
            'last_stats_log': None
        }
        
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
            self.stats['start_time'] = time.time()
            self.stats['last_stats_log'] = time.time()
            
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
    
    def start_multiple_consumers(self, num_consumers=2):
        """Start multiple consumer instances for parallel processing"""
        import threading
        
        logger.info(f"Starting {num_consumers} consumer threads for parallel processing")
        
        threads = []
        for i in range(num_consumers):
            consumer = FraudDetectionConsumer()
            thread = threading.Thread(
                target=consumer.start_consuming,
                name=f"FraudConsumer-{i+1}"
            )
            thread.daemon = True
            threads.append(thread)
            thread.start()
            
        return threads
    
    def _consume_loop(self):
        """Main consumption loop with batch processing for higher throughput"""
        try:
            message_batch = []
            last_commit_time = time.time()
            commit_interval = 5.0  # Commit every 5 seconds
            
            for message in self.consumer:
                if not self.running:
                    break
                
                message_batch.append(message)
                
                # Process batch when it reaches target size or timeout
                if len(message_batch) >= self.batch_size or (time.time() - last_commit_time) > commit_interval:
                    try:
                        # Process the batch
                        self._process_message_batch(message_batch)
                        
                        # Commit offsets manually for better control
                        self.consumer.commit()
                        last_commit_time = time.time()
                        
                        # Clear the batch
                        batch_size = len(message_batch)
                        message_batch = []
                        
                        # Update stats
                        self.stats['messages_processed'] += batch_size
                        self._log_performance_stats()
                        
                        logger.debug(f"Processed batch of {batch_size}, committed offsets")
                        
                    except Exception as e:
                        logger.error(f"Error processing message batch: {e}")
                        # Don't commit on error - will reprocess
                        message_batch = []
                        continue
                    
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
            # Process remaining messages in batch
            if message_batch:
                try:
                    self._process_message_batch(message_batch)
                    self.consumer.commit()
                except Exception as e:
                    logger.error(f"Error processing final batch: {e}")
        except Exception as e:
            logger.error(f"Consumer loop error: {e}")
    
    def _process_message_batch(self, messages):
        """Process a batch of messages for better throughput"""
        if not messages:
            return
            
        logger.debug(f"Processing batch of {len(messages)} messages")
        
        # Group messages by whether they need predictions
        prediction_messages = []
        other_messages = []
        
        for message in messages:
            try:
                transaction_data = message.value
                if not self._validate_transaction_message(transaction_data):
                    continue
                    
                # Create transaction record first
                with app.app_context():
                    transaction = self._create_transaction_record(transaction_data)
                    if not transaction:
                        continue
                    
                    # Check if predictions are enabled
                    enable_predictions = transaction_data.get('enable_predictions', True)
                    if enable_predictions:
                        prediction_messages.append((transaction, transaction_data))
                    else:
                        other_messages.append((transaction, transaction_data))
                        
            except Exception as e:
                logger.error(f"Error preparing message for batch processing: {e}")
                continue
        
        # Process predictions in batch if we have any
        if prediction_messages:
            self._process_prediction_batch(prediction_messages)
            
        # Process other messages
        for transaction, transaction_data in other_messages:
            logger.debug(f"Processed transaction {transaction_data.get('transaction_id')} without prediction")
    
    def _process_prediction_batch(self, prediction_messages):
        """Process fraud predictions in batch for better performance"""
        try:
            transactions = []
            transaction_data_list = []
            
            for transaction, transaction_data in prediction_messages:
                transactions.append(transaction)
                transaction_data_list.append(transaction_data)
            
            # Prepare feature vectors for batch prediction
            feature_vectors = []
            for transaction in transactions:
                features = [
                    transaction.time_feature,
                    *[getattr(transaction, f'v{i}') for i in range(1, 29)],
                    transaction.amount
                ]
                feature_vectors.append(features)
            
            if not feature_vectors:
                return
                
            # Scale features
            import numpy as np
            features_array = np.array(feature_vectors)
            
            with app.app_context():
                # Apply scaling
                if hasattr(self.ml_models, 'data_processor') and self.ml_models.data_processor and hasattr(self.ml_models.data_processor.scaler, "scale_"):
                    features_scaled = self.ml_models.data_processor.scaler.transform(features_array)
                else:
                    from data_processor import DataProcessor
                    data_proc = DataProcessor()
                    data_proc.load_scaler()
                    features_scaled = data_proc.scaler.transform(features_array)
                
                # Make batch predictions
                prediction_results = self.ml_models.ensemble_predict(features_scaled)
                
                if prediction_results is None:
                    logger.error("Failed to get batch prediction results")
                    return
                
                # Save predictions and create alerts in batch
                prediction_records = []
                fraud_alerts = []
                
                for i, (transaction, transaction_data) in enumerate(zip(transactions, transaction_data_list)):
                    try:
                        # Extract prediction results
                        prediction = int(prediction_results['final_predictions'][i])
                        confidence = float(prediction_results['confidence_scores'][i])
                        ensemble_score = float(prediction_results['ensemble_scores'][i])
                        isolation_score = float(prediction_results['isolation_scores'][i]) if prediction_results['isolation_scores'] is not None else 0.0
                        logistic_score = float(prediction_results['logistic_probabilities'][i]) if prediction_results['logistic_probabilities'] is not None else 0.0
                        xgboost_score = float(prediction_results['xgboost_probabilities'][i]) if prediction_results['xgboost_probabilities'] is not None else 0.0
                        
                        # Create prediction record
                        pred_record = Prediction()
                        pred_record.transaction_id = transaction.id
                        pred_record.isolation_forest_score = isolation_score
                        pred_record.logistic_regression_score = logistic_score
                        pred_record.xgboost_score = xgboost_score
                        pred_record.ensemble_prediction = ensemble_score
                        pred_record.final_prediction = prediction
                        pred_record.confidence_score = confidence
                        pred_record.model_version = 'kafka_ensemble_v1.0_batch'
                        pred_record.prediction_time = datetime.utcnow()
                        
                        prediction_records.append(pred_record)
                        
                        # Create fraud alert if needed
                        if prediction == 1 and confidence > 0.7:
                            alert = FraudAlert()
                            alert.transaction_id = transaction.id
                            alert.alert_level = 'HIGH' if confidence > 0.8 else 'MEDIUM'
                            alert.alert_reason = f"Batch fraud prediction (confidence: {confidence:.3f})"
                            fraud_alerts.append(alert)
                        
                        # Publish prediction result
                        result = {
                            'transaction_id': transaction_data['transaction_id'],
                            'prediction': prediction,
                            'confidence': confidence,
                            'model_scores': {
                                'isolation_forest': isolation_score,
                                'logistic_regression': logistic_score,
                                'xgboost': xgboost_score,
                                'ensemble': ensemble_score
                            },
                            'timestamp': datetime.utcnow().isoformat()
                        }
                        self._publish_prediction_result(transaction_data, result)
                        
                    except Exception as e:
                        logger.error(f"Error processing prediction for transaction {i}: {e}")
                        continue
                
                # Bulk save to database
                if prediction_records:
                    db.session.bulk_save_objects(prediction_records)
                if fraud_alerts:
                    db.session.bulk_save_objects(fraud_alerts)
                
                db.session.commit()
                
                # Update stats
                self.stats['predictions_made'] += len(prediction_records)
                self.stats['alerts_created'] += len(fraud_alerts)
                
                logger.info(f"Batch processed {len(prediction_records)} predictions, {len(fraud_alerts)} alerts")
                
        except Exception as e:
            logger.error(f"Error in batch prediction processing: {e}")
            db.session.rollback()
    
    def _log_performance_stats(self):
        """Log performance statistics periodically"""
        current_time = time.time()
        
        # Log stats every 30 seconds
        if current_time - self.stats['last_stats_log'] >= 30:
            elapsed_time = current_time - self.stats['start_time']
            messages_per_sec = self.stats['messages_processed'] / elapsed_time if elapsed_time > 0 else 0
            predictions_per_sec = self.stats['predictions_made'] / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"Performance Stats - Messages: {self.stats['messages_processed']} "
                       f"({messages_per_sec:.1f}/sec), Predictions: {self.stats['predictions_made']} "
                       f"({predictions_per_sec:.1f}/sec), Alerts: {self.stats['alerts_created']}")
            
            self.stats['last_stats_log'] = current_time
    
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
            
            # Set transaction_id based on order (use provided transaction_id if available)
            transaction.transaction_id = data.get('transaction_id', str(data.get('order', Transaction.query.count() + 1)))
            
            # Set metadata
            transaction.created_at = datetime.utcnow()
            
            db.session.add(transaction)
            db.session.commit()
            
            logger.debug(f"Transaction {transaction.transaction_id} (ID: {transaction.id}) saved to database")
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
            ensemble_score = float(prediction_results['ensemble_scores'][0])
            isolation_score = float(prediction_results['isolation_scores'][0]) if prediction_results['isolation_scores'] is not None else 0.0
            logistic_score = float(prediction_results['logistic_probabilities'][0]) if prediction_results['logistic_probabilities'] is not None else 0.0
            xgboost_score = float(prediction_results['xgboost_probabilities'][0]) if prediction_results['xgboost_probabilities'] is not None else 0.0
            
            # Save prediction record - using correct column names
            pred_record = Prediction()
            pred_record.transaction_id = transaction.id
            pred_record.isolation_forest_score = isolation_score
            pred_record.logistic_regression_score = logistic_score
            pred_record.xgboost_score = xgboost_score
            pred_record.ensemble_prediction = ensemble_score
            pred_record.final_prediction = prediction
            pred_record.confidence_score = confidence
            pred_record.model_version = 'kafka_ensemble_v1.0'
            pred_record.prediction_time = datetime.utcnow()
            
            db.session.add(pred_record)
            db.session.commit()
            
            result = {
                'transaction_id': original_data['transaction_id'],
                'prediction': prediction,
                'confidence': float(confidence),
                'model_scores': {
                    'isolation_forest': isolation_score,
                    'logistic_regression': logistic_score,
                    'xgboost': xgboost_score,
                    'ensemble': ensemble_score
                },
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

# High-performance consumer entry point
def run_high_performance_consumer(parallel=1, batch_size=50):
    """Run optimized high-performance fraud detection consumer"""
    
    logger.info(f"Starting high-performance fraud detection consumer")
    logger.info(f"Parallel consumers: {parallel}")
    logger.info(f"Batch size: {batch_size}")
    
    consumer = FraudDetectionConsumer()
    
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        consumer.running = False
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if parallel > 1:
            threads = consumer.start_multiple_consumers(parallel)
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        else:
            consumer.start_consuming()
            
    except Exception as e:
        logger.error(f"Consumer error: {e}")
    finally:
        logger.info("Consumer shutdown complete")

if __name__ == "__main__":
    import argparse
    
    # Add command line arguments for high-performance mode
    parser = argparse.ArgumentParser(description='Fraud Detection Consumer')
    parser.add_argument('--high-performance', action='store_true', help='Enable high-performance batch mode')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel consumers (default: 1)')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing (default: 50)')
    
    args = parser.parse_args()
    
    if args.high_performance:
        logger.info("Starting in HIGH-PERFORMANCE mode")
        logger.info(f"Performance Settings: parallel={args.parallel}, batch_size={args.batch_size}")
        run_high_performance_consumer(parallel=args.parallel, batch_size=args.batch_size)
    else:
        logger.info("Starting in standard mode")
        consumer = FraudDetectionConsumer()
        
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            consumer.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            consumer.start_consuming()
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            logger.info("Consumer shutdown complete")
