import json
import logging
from kafka import KafkaProducer as KafkaProducerLib
from kafka.errors import KafkaError
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class KafkaProducer:
    def __init__(self, kafka_config: Dict[str, Any]):
        self.kafka_config = kafka_config
        self.bootstrap_servers = kafka_config.get('bootstrap_servers', 'localhost:9092')
        self.producer = None
        self.is_connected = False

    def connect(self):
        """Establish Kafka producer connection"""
        try:
            self.producer = KafkaProducerLib(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas to acknowledge
                retries=3,
                retry_backoff_ms=100,
                request_timeout_ms=30000,
                max_in_flight_requests_per_connection=1
            )
            
            # Test connection by checking cluster metadata
            metadata = self.producer.bootstrap()
            self.is_connected = True
            logger.info(f"Connected to Kafka cluster: {self.bootstrap_servers}")
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Kafka: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Close Kafka producer connection"""
        if self.producer:
            self.producer.close()
            self.is_connected = False
            logger.info("Disconnected from Kafka")

    def produce(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> bool:
        """Send a message to Kafka topic"""
        try:
            if not self.is_connected:
                if not self.connect():
                    return False
            
            if not self.producer:
                logger.error("Producer not initialized")
                return False
            
            # Send message
            future = self.producer.send(
                topic=topic,
                value=message,
                key=key
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=30)
            
            logger.info(f"Message sent to topic {topic}, partition {record_metadata.partition}, offset {record_metadata.offset}")
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to send message to Kafka: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}")
            return False

    def produce_fraud_detection_event(self, transaction_data: Dict[str, Any]) -> bool:
        """Send fraud detection event to Kafka"""
        topic = self.kafka_config.get('topic', 'fraud_detection_topic')
        transaction_id = transaction_data.get('transaction_id')
        
        event = {
            'event_type': 'fraud_detection_request',
            'timestamp': transaction_data.get('timestamp'),
            'transaction_data': transaction_data
        }
        
        return self.produce(topic, event, key=transaction_id)

    def produce_fraud_result_event(self, result_data: Dict[str, Any]) -> bool:
        """Send fraud detection result to Kafka"""
        topic = self.kafka_config.get('topic', 'fraud_detection_topic')
        transaction_id = result_data.get('transaction_id')
        
        event = {
            'event_type': 'fraud_detection_result',
            'timestamp': result_data.get('timestamp'),
            'result_data': result_data
        }
        
        return self.produce(topic, event, key=transaction_id)

    def flush(self):
        """Flush all buffered messages"""
        if self.producer and self.is_connected:
            self.producer.flush()

    def health_check(self) -> bool:
        """Check Kafka producer health"""
        try:
            if not self.is_connected:
                return self.connect()
            
            if self.producer:
                # Try to get cluster metadata as health check
                metadata = self.producer.bootstrap()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Kafka producer health check failed: {e}")
            return False