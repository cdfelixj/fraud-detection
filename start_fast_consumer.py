#!/usr/bin/env python3
"""
High-Performance Fraud Detection Consumer Launcher
Optimized for maximum throughput with batch processing and parallel consumers
"""

import os
import sys
import logging
from kafka_consumer import run_high_performance_consumer

# Configure logging for high-performance mode
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fraud_consumer_performance.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for high-performance consumer"""
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION - HIGH PERFORMANCE CONSUMER")
    logger.info("=" * 60)
    
    # Performance optimizations
    logger.info("Performance Optimizations Enabled:")
    logger.info("- Batch processing (50 messages per batch)")
    logger.info("- Manual offset commits every 5 seconds")
    logger.info("- Bulk database operations")
    logger.info("- LZ4 compression for faster throughput")
    logger.info("- Increased batch sizes and buffer memory")
    logger.info("- Performance monitoring and logging")
    
    # Environment check
    kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    logger.info(f"Kafka servers: {kafka_servers}")
    
    db_url = os.getenv('DATABASE_URL', 'sqlite:///fraud_detection.db')
    logger.info(f"Database: {db_url}")
    
    # Start the consumer
    try:
        run_high_performance_consumer()
    except KeyboardInterrupt:
        logger.info("Consumer stopped by user")
    except Exception as e:
        logger.error(f"Consumer failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
