"""
Kafka Producer Service for Fraud Detection System
Provides high-level API for sending messages to Kafka topics
"""
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from kafka_config import kafka_manager, KafkaConfig

logger = logging.getLogger(__name__)

class FraudDetectionProducer:
    """High-level producer for fraud detection events"""
    
    def __init__(self):
        self.config = KafkaConfig()
        
    def send_transaction(self, transaction_data: Dict[str, Any]) -> bool:
        """Send transaction data to Kafka for processing"""
        try:
            # Add metadata
            enriched_data = {
                **transaction_data,
                'producer_timestamp': datetime.utcnow().isoformat(),
                'source': 'api'
            }
            
            return kafka_manager.send_message(
                topic=self.config.topics['transactions'],
                value=enriched_data,
                key=transaction_data.get('transaction_id')
            )
            
        except Exception as e:
            logger.error(f"Error sending transaction: {e}")
            return False
    
    def send_batch_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Send multiple transactions efficiently"""
        results = {'success': 0, 'failed': 0}
        
        try:
            producer = kafka_manager.get_producer()
            
            # Send all messages asynchronously
            futures = []
            for transaction in transactions:
                enriched_data = {
                    **transaction,
                    'producer_timestamp': datetime.utcnow().isoformat(),
                    'source': 'batch_upload'
                }
                
                future = producer.send(
                    self.config.topics['transactions'],
                    value=enriched_data,
                    key=transaction.get('transaction_id')
                )
                futures.append(future)
            
            # Wait for all messages to be sent
            for future in futures:
                try:
                    future.get(timeout=10)
                    results['success'] += 1
                except Exception as e:
                    logger.error(f"Failed to send transaction: {e}")
                    results['failed'] += 1
            
            # Ensure all messages are sent
            producer.flush()
            
            logger.info(f"Batch sent: {results['success']} success, {results['failed']} failed")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch send: {e}")
            results['failed'] = len(transactions)
            return results
    
    def send_manual_transaction(self, features: List[float], transaction_id: Optional[str] = None) -> bool:
        """Send manually entered transaction for prediction"""
        try:
            if not transaction_id:
                transaction_id = f"manual_{int(time.time() * 1000)}"
            
            # Convert features to transaction format
            transaction_data = {
                'transaction_id': transaction_id,
                'time_feature': features[0],
                'amount': features[-1],  # Amount is last feature
                'source': 'manual_input',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add V1-V28 features
            for i in range(1, 29):
                transaction_data[f'v{i}'] = features[i] if i < len(features) - 1 else 0
            
            return self.send_transaction(transaction_data)
            
        except Exception as e:
            logger.error(f"Error sending manual transaction: {e}")
            return False
    
    def send_feedback(self, prediction_id: int, feedback_data: Dict[str, Any]) -> bool:
        """Send user feedback about prediction accuracy"""
        try:
            feedback_message = {
                'prediction_id': prediction_id,
                'feedback': feedback_data,
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'user_feedback'
            }
            
            return kafka_manager.send_message(
                topic=self.config.topics['feedback'],
                value=feedback_message,
                key=str(prediction_id)
            )
            
        except Exception as e:
            logger.error(f"Error sending feedback: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Check producer health and Kafka connectivity"""
        try:
            kafka_healthy = kafka_manager.health_check()
            
            # Test message send
            test_message = {
                'test': True,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Try to send test message (use a test topic)
            test_send = kafka_manager.send_message(
                topic='health-check',
                value=test_message
            )
            
            return {
                'kafka_connected': kafka_healthy,
                'test_send_success': test_send,
                'bootstrap_servers': self.config.bootstrap_servers,
                'topics_configured': list(self.config.topics.values())
            }
            
        except Exception as e:
            logger.error(f"Producer health check failed: {e}")
            return {
                'kafka_connected': False,
                'error': str(e)
            }

# Global producer instance
fraud_producer = FraudDetectionProducer()
