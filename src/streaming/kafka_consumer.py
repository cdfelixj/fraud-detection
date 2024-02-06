import json
import logging
import signal
import sys
from kafka import KafkaConsumer as KafkaConsumerLib
from kafka.errors import KafkaError
from typing import Dict, Any, Callable, Optional
import threading

logger = logging.getLogger(__name__)

class KafkaConsumer:
    def __init__(self, kafka_config: Dict[str, Any], message_processor: Optional[Callable] = None):
        self.kafka_config = kafka_config
        self.bootstrap_servers = kafka_config.get('bootstrap_servers', 'localhost:9092')
        self.topic = kafka_config.get('topic', 'fraud_detection_topic')
        self.group_id = kafka_config.get('group_id', 'fraud_detection_group')
        self.consumer = None
        self.is_connected = False
        self.is_consuming = False
        self.consumer_thread = None
        self.message_processor = message_processor or self.default_message_processor

    def connect(self):
        """Establish Kafka consumer connection"""
        try:
            self.consumer = KafkaConsumerLib(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')) if m else None,
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='earliest',  # Start from beginning if no offset
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                consumer_timeout_ms=1000,  # Timeout for polling
                max_poll_records=100,
                session_timeout_ms=30000,
                heartbeat_interval_ms=3000
            )
            
            self.is_connected = True
            logger.info(f"Connected to Kafka consumer for topic: {self.topic}")
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to connect Kafka consumer: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting Kafka consumer: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Close Kafka consumer connection"""
        self.is_consuming = False
        
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=5)
        
        if self.consumer:
            self.consumer.close()
            self.is_connected = False
            logger.info("Disconnected from Kafka consumer")

    def start_consuming(self, async_mode: bool = False):
        """Start consuming messages"""
        if not self.is_connected:
            if not self.connect():
                return False
        
        self.is_consuming = True
        
        if async_mode:
            self.consumer_thread = threading.Thread(target=self._consume_messages)
            self.consumer_thread.daemon = True
            self.consumer_thread.start()
            logger.info("Started consuming messages asynchronously")
        else:
            self._consume_messages()
        
        return True

    def stop_consuming(self):
        """Stop consuming messages"""
        self.is_consuming = False
        logger.info("Stopped consuming messages")

    def _consume_messages(self):
        """Internal method to consume messages"""
        try:
            while self.is_consuming:
                try:
                    # Poll for messages
                    messages = self.consumer.poll(timeout_ms=1000)
                    
                    for topic_partition, records in messages.items():
                        for message in records:
                            if not self.is_consuming:
                                break
                            
                            try:
                                self.process_message(message)
                            except Exception as e:
                                logger.error(f"Error processing message: {e}")
                                # Continue processing other messages
                    
                    if not self.is_consuming:
                        break
                        
                except KafkaError as e:
                    logger.error(f"Kafka error while consuming: {e}")
                    # Try to reconnect
                    if not self.connect():
                        break
                except Exception as e:
                    logger.error(f"Unexpected error while consuming: {e}")
                    break
                    
        finally:
            logger.info("Message consumption stopped")

    def process_message(self, message):
        """Process a single message"""
        try:
            logger.debug(f"Processing message from topic {message.topic}, partition {message.partition}, offset {message.offset}")
            
            # Call the message processor
            self.message_processor(message)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def default_message_processor(self, message):
        """Default message processor - just logs the message"""
        try:
            data = message.value
            logger.info(f"Received message: {data}")
            
            # Basic message routing based on event type
            if isinstance(data, dict):
                event_type = data.get('event_type')
                
                if event_type == 'fraud_detection_request':
                    self.handle_fraud_detection_request(data)
                elif event_type == 'fraud_detection_result':
                    self.handle_fraud_detection_result(data)
                else:
                    logger.warning(f"Unknown event type: {event_type}")
            
        except Exception as e:
            logger.error(f"Error in default message processor: {e}")

    def handle_fraud_detection_request(self, data: Dict[str, Any]):
        """Handle fraud detection request"""
        logger.info(f"Processing fraud detection request: {data.get('transaction_data', {}).get('transaction_id', 'unknown')}")
        # This would trigger the fraud detection pipeline
        # Implementation would depend on your specific requirements

    def handle_fraud_detection_result(self, data: Dict[str, Any]):
        """Handle fraud detection result"""
        logger.info(f"Processing fraud detection result: {data.get('result_data', {})}")
        # This would handle storing/forwarding the result
        # Implementation would depend on your specific requirements

    def health_check(self) -> bool:
        """Check Kafka consumer health"""
        try:
            if not self.is_connected:
                return self.connect()
            
            if self.consumer:
                # Check if consumer is still connected
                partitions = self.consumer.assignment()
                return len(partitions) >= 0
            return False
            
        except Exception as e:
            logger.error(f"Kafka consumer health check failed: {e}")
            return False

    def get_consumer_stats(self) -> Dict[str, Any]:
        """Get consumer statistics"""
        try:
            if not self.consumer:
                return {}
            
            return {
                'is_connected': self.is_connected,
                'is_consuming': self.is_consuming,
                'topic': self.topic,
                'group_id': self.group_id,
                'assigned_partitions': len(self.consumer.assignment()) if self.consumer else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting consumer stats: {e}")
            return {}