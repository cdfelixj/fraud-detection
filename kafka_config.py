"""
Kafka Configuration and Utilities for Fraud Detection System
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from kafka.errors import KafkaError
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaConfig:
    """Kafka configuration management"""
    
    def __init__(self):
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.topics = {
            'transactions': 'fraud-detection-transactions',
            'predictions': 'fraud-detection-predictions', 
            'alerts': 'fraud-detection-alerts',
            'feedback': 'fraud-detection-feedback'
        }
        
    def get_producer_config(self) -> Dict[str, Any]:
        """Get Kafka producer configuration optimized for high throughput"""
        return {
            'bootstrap_servers': self.bootstrap_servers,
            'value_serializer': lambda x: json.dumps(x, default=str).encode('utf-8'),
            'key_serializer': lambda x: str(x).encode('utf-8') if x else None,
            'acks': 1,  # Changed from 'all' to 1 for better throughput
            'retries': 3,
            'batch_size': 32768,  # Increased from 16384 for better batching
            'linger_ms': 5,  # Increased from 1ms to allow more batching
            'buffer_memory': 67108864,  # Increased to 64MB from 32MB
            'compression_type': 'lz4',  # Changed from gzip to lz4 for faster compression
            'max_in_flight_requests_per_connection': 5,  # Increased from 1 for better throughput
            'enable_idempotence': False,  # Disabled for better performance
            'request_timeout_ms': 30000,
            'delivery_timeout_ms': 120000
        }
    
    def get_consumer_config(self, group_id: str) -> Dict[str, Any]:
        """Get Kafka consumer configuration optimized for high throughput"""
        return {
            'bootstrap_servers': self.bootstrap_servers,
            'group_id': group_id,
            'value_deserializer': lambda x: json.loads(x.decode('utf-8')),
            'key_deserializer': lambda x: x.decode('utf-8') if x else None,
            'auto_offset_reset': 'latest',
            'enable_auto_commit': False,  # Manual commit for better control
            'session_timeout_ms': 30000,
            'heartbeat_interval_ms': 3000,
            'max_poll_records': 1000,  # Increased from 500 for better batching
            'max_poll_interval_ms': 300000,
            'fetch_min_bytes': 1024,  # Wait for at least 1KB of data
            'fetch_max_wait_ms': 100,  # Maximum 100ms wait for batching
            'max_partition_fetch_bytes': 1048576,  # 1MB per partition
            'connections_max_idle_ms': 300000,
            'request_timeout_ms': 40000
        }

class KafkaManager:
    """Kafka producer and consumer management"""
    
    def __init__(self):
        self.config = KafkaConfig()
        self._producer = None
        self._consumer = None
        
    def get_producer(self) -> KafkaProducer:
        """Get or create Kafka producer"""
        if self._producer is None:
            try:
                self._producer = KafkaProducer(**self.config.get_producer_config())
                logger.info(f"Kafka producer connected to {self.config.bootstrap_servers}")
            except Exception as e:
                logger.error(f"Failed to create Kafka producer: {e}")
                raise
        return self._producer
    
    def get_consumer(self, group_id: str, topics: list) -> KafkaConsumer:
        """Get or create Kafka consumer"""
        try:
            consumer = KafkaConsumer(
                *topics,
                **self.config.get_consumer_config(group_id)
            )
            logger.info(f"Kafka consumer created for group {group_id}, topics: {topics}")
            return consumer
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {e}")
            raise
    
    def send_message(self, topic: str, value: Dict[str, Any], key: Optional[str] = None) -> bool:
        """Send message to Kafka topic"""
        try:
            producer = self.get_producer()
            future = producer.send(topic, value=value, key=key)
            
            # Wait for message to be sent (blocking call)
            record_metadata = future.get(timeout=10)
            logger.debug(f"Message sent to {record_metadata.topic}:{record_metadata.partition}:{record_metadata.offset}")
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to send message to {topic}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}")
            return False
    
    def close_producer(self):
        """Close Kafka producer"""
        if self._producer:
            self._producer.flush()
            self._producer.close()
            self._producer = None
            logger.info("Kafka producer closed")
    
    def health_check(self) -> bool:
        """Check Kafka connection health"""
        try:
            # Simple health check - just try to create a producer
            producer = self.get_producer()
            if producer is not None:
                logger.info("Kafka health check passed. Producer created successfully")
                return True
            else:
                logger.warning("Kafka producer is None")
                return False
        except Exception as e:
            logger.error(f"Kafka health check failed: {e}")
            return False

# Global Kafka manager instance
kafka_manager = KafkaManager()

def create_topics_if_not_exist():
    """Create Kafka topics if they don't exist"""
    try:
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka.errors import TopicAlreadyExistsError
        
        admin_client = KafkaAdminClient(
            bootstrap_servers=kafka_manager.config.bootstrap_servers,
            client_id='fraud_detection_admin'
        )
        
        # Check existing topics first
        existing_topics = admin_client.list_topics()
        
        # Define topics to create
        topics_to_create = []
        for topic_name in kafka_manager.config.topics.values():
            if topic_name not in existing_topics:
                topics_to_create.append(
                    NewTopic(
                        name=topic_name,
                        num_partitions=3,
                        replication_factor=1
                    )
                )
        
        # Create only non-existing topics
        if topics_to_create:
            try:
                admin_client.create_topics(topics_to_create, validate_only=False)
                logger.info(f"New topics created: {[t.name for t in topics_to_create]}")
            except TopicAlreadyExistsError:
                logger.info("Topics already exist - this is normal on restart")
            except Exception as e:
                if "TopicAlreadyExistsError" in str(e) or "already exists" in str(e):
                    logger.info("Topics already exist - this is normal on restart")
                else:
                    logger.error(f"Error creating topics: {e}")
        else:
            logger.info("All required topics already exist")
                
        admin_client.close()
        
    except Exception as e:
        logger.error(f"Failed to initialize topic creation: {e}")

# Message schemas for validation
TRANSACTION_SCHEMA = {
    'transaction_id': str,
    'time_feature': float,
    'amount': float,
    'timestamp': str,
}

PREDICTION_SCHEMA = {
    'transaction_id': str,
    'prediction': int,
    'confidence': float,
    'model_scores': dict,
    'timestamp': str
}
