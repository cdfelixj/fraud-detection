import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import numpy as np
from src.streaming.kafka_consumer import KafkaConsumer
from src.streaming.kafka_producer import KafkaProducer
from src.streaming.stream_processor import StreamProcessor
from src.utils.config import KAFKA_CONFIG

class TestKafkaConsumer(unittest.TestCase):
    def setUp(self):
        self.kafka_config = KAFKA_CONFIG.copy()
        self.consumer = KafkaConsumer(self.kafka_config)

    @patch('src.streaming.kafka_consumer.KafkaConsumerLib')
    def test_connect(self, mock_kafka_consumer):
        """Test Kafka consumer connection"""
        mock_consumer_instance = Mock()
        mock_kafka_consumer.return_value = mock_consumer_instance
        
        result = self.consumer.connect()
        
        self.assertTrue(result)
        self.assertTrue(self.consumer.is_connected)
        mock_kafka_consumer.assert_called_once()

    def test_default_message_processor(self):
        """Test default message processing"""
        mock_message = Mock()
        mock_message.value = {'event_type': 'fraud_detection_request', 'data': 'test'}
        mock_message.topic = 'test_topic'
        mock_message.partition = 0
        mock_message.offset = 100
        
        # Should not raise exception
        self.consumer.default_message_processor(mock_message)

    def test_health_check_disconnected(self):
        """Test health check when disconnected"""
        result = self.consumer.health_check()
        self.assertFalse(result)

class TestKafkaProducer(unittest.TestCase):
    def setUp(self):
        self.kafka_config = KAFKA_CONFIG.copy()
        self.producer = KafkaProducer(self.kafka_config)

    @patch('src.streaming.kafka_producer.KafkaProducerLib')
    def test_connect(self, mock_kafka_producer):
        """Test Kafka producer connection"""
        mock_producer_instance = Mock()
        mock_producer_instance.bootstrap.return_value = Mock()
        mock_kafka_producer.return_value = mock_producer_instance
        
        result = self.producer.connect()
        
        self.assertTrue(result)
        self.assertTrue(self.producer.is_connected)
        mock_kafka_producer.assert_called_once()

    @patch('src.streaming.kafka_producer.KafkaProducerLib')
    def test_produce_message(self, mock_kafka_producer):
        """Test message production"""
        mock_producer_instance = Mock()
        mock_future = Mock()
        mock_record_metadata = Mock()
        mock_record_metadata.partition = 0
        mock_record_metadata.offset = 100
        mock_future.get.return_value = mock_record_metadata
        mock_producer_instance.send.return_value = mock_future
        mock_producer_instance.bootstrap.return_value = Mock()
        mock_kafka_producer.return_value = mock_producer_instance
        
        self.producer.connect()
        result = self.producer.produce("test_topic", {"test": "message"})
        
        self.assertTrue(result)
        mock_producer_instance.send.assert_called_once()

    def test_produce_fraud_detection_event(self):
        """Test fraud detection event production"""
        transaction_data = {
            'transaction_id': 'T123',
            'amount': 100.0,
            'timestamp': '2023-01-01T12:00:00Z'
        }
        
        with patch.object(self.producer, 'produce') as mock_produce:
            mock_produce.return_value = True
            result = self.producer.produce_fraud_detection_event(transaction_data)
            
            self.assertTrue(result)
            mock_produce.assert_called_once()

class TestStreamProcessor(unittest.TestCase):
    def setUp(self):
        # Create mock dependencies
        self.mock_consumer = Mock()
        self.mock_model = Mock()
        self.mock_model.predict_fraud_probability = Mock(return_value=0.3)
        
        self.processor = StreamProcessor(
            kafka_consumer=self.mock_consumer,
            model=self.mock_model
        )

    def test_analyze_data(self):
        """Test data analysis"""
        test_data = {
            "transaction_id": "T123",
            "amount": 100,
            "transaction_type": "credit",
            "merchant_id": "M123",
            "timestamp": "2023-01-01T12:00:00Z"
        }
        
        result = self.processor.analyze(test_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('fraud_probability', result)
        self.assertIn('is_fraud', result)
        self.assertIn('confidence', result)

    def test_validate_transaction_data(self):
        """Test transaction data validation"""
        valid_data = {
            "transaction_id": "T123",
            "amount": 100.0,
            "timestamp": "2023-01-01T12:00:00Z"
        }
        
        invalid_data = {
            "amount": -50  # Negative amount should be invalid
        }
        
        self.assertTrue(self.processor._validate_transaction_data(valid_data))
        self.assertFalse(self.processor._validate_transaction_data(invalid_data))

    def test_prepare_features(self):
        """Test feature preparation"""
        test_data = {
            "amount": 100.0,
            "transaction_type": "credit",
            "timestamp": "2023-01-01T12:00:00Z"
        }
        
        features = self.processor._prepare_features(test_data)
        
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], 1)  # Single sample

    def test_get_risk_level(self):
        """Test risk level calculation"""
        self.assertEqual(self.processor._get_risk_level(0.9), 'CRITICAL')
        self.assertEqual(self.processor._get_risk_level(0.7), 'HIGH')
        self.assertEqual(self.processor._get_risk_level(0.5), 'MEDIUM')
        self.assertEqual(self.processor._get_risk_level(0.3), 'LOW')
        self.assertEqual(self.processor._get_risk_level(0.1), 'MINIMAL')

    def test_extract_data(self):
        """Test data extraction from message"""
        # Test with nested transaction data
        mock_message = Mock()
        mock_message.value = {
            'event_type': 'fraud_detection_request',
            'transaction_data': {'transaction_id': 'T123', 'amount': 100}
        }
        
        extracted = self.processor._extract_data(mock_message)
        
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted['transaction_id'], 'T123')
        self.assertEqual(extracted['amount'], 100)

    def test_get_processing_stats(self):
        """Test processing statistics"""
        stats = self.processor.get_processing_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('is_processing', stats)
        self.assertIn('processed_count', stats)
        self.assertIn('fraud_detected_count', stats)

if __name__ == '__main__':
    unittest.main()