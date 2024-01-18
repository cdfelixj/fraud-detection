# Configuration settings for the fraud detection system
import os

# Database configuration
DATABASE_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'password'),
    'database': os.getenv('POSTGRES_DB', 'fraud_detection_db')
}

# Kafka configuration
KAFKA_CONFIG = {
    'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
    'topic': os.getenv('KAFKA_TOPIC', 'fraud_detection_topic'),
    'group_id': os.getenv('KAFKA_GROUP_ID', 'fraud_detection_group')
}

# Redis configuration
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'db': int(os.getenv('REDIS_DB', 0))
}

# Model configuration
MODEL_CONFIG = {
    'isolation_forest': {
        'contamination': float(os.getenv('IF_CONTAMINATION', 0.1)),
        'n_estimators': int(os.getenv('IF_N_ESTIMATORS', 100))
    },
    'lstm': {
        'sequence_length': int(os.getenv('LSTM_SEQUENCE_LENGTH', 10)),
        'features': int(os.getenv('LSTM_FEATURES', 5)),
        'epochs': int(os.getenv('LSTM_EPOCHS', 50))
    }
}