"""
Database Streaming Service for Kafka Integration
Streams existing transaction data from PostgreSQL to Kafka on-demand
"""
import os
import json
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import threading
import signal
import sys

# Database imports
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Kafka imports
from kafka_config import KafkaConfig
from kafka_producer import FraudDetectionProducer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseStreamer:
    """Stream existing transaction data from database to Kafka"""
    
    def __init__(self):
        self.kafka_config = KafkaConfig()
        self.producer = None
        self.is_streaming = False
        self.stream_thread = None
        self.session_factory = None
        self.batch_size = int(os.getenv('DB_STREAM_BATCH_SIZE', '100'))
        self.stream_interval = float(os.getenv('DB_STREAM_INTERVAL', '1.0'))  # seconds between batches
        
        # Database connection
        self._setup_database()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            db_url = os.getenv('DATABASE_URL', 'postgresql://fraud_user:fraud_pass@localhost:5432/fraud_detection')
            engine = create_engine(db_url)
            self.session_factory = sessionmaker(bind=engine)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            raise
    
    def _setup_kafka_producer(self):
        """Setup Kafka producer"""
        try:
            self.producer = FraudDetectionProducer()
            logger.info("Kafka producer initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to setup Kafka producer: {e}")
            return False
    
    def start_streaming(self, limit: Optional[int] = None, offset: int = 0):
        """Start streaming data from database to Kafka"""
        if self.is_streaming:
            logger.warning("Streaming is already active")
            return False
        
        if not self._setup_kafka_producer():
            return False
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(
            target=self._stream_worker,
            args=(limit, offset),
            daemon=True
        )
        self.stream_thread.start()
        logger.info(f"Started database streaming (limit={limit}, offset={offset})")
        return True
    
    def stop_streaming(self):
        """Stop streaming data"""
        if not self.is_streaming:
            logger.warning("No active streaming to stop")
            return False
        
        self.is_streaming = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5)
        
        if self.producer:
            self.producer.close()
            self.producer = None
        
        logger.info("Stopped database streaming")
        return True
    
    def _stream_worker(self, limit: Optional[int], offset: int):
        """Worker thread that streams data from database"""
        session = self.session_factory()
        try:
            # Build query
            query = """
                SELECT id, time_feature, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                       v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                       v21, v22, v23, v24, v25, v26, v27, v28, amount, actual_class,
                       created_at
                FROM transactions
                ORDER BY id
                OFFSET :offset
            """
            
            if limit:
                query += " LIMIT :limit"
            
            # Execute query and stream results
            current_offset = offset
            records_streamed = 0
            
            while self.is_streaming:
                # Fetch batch
                if limit:
                    result = session.execute(
                        text(query), 
                        {"offset": current_offset, "limit": self.batch_size}
                    )
                else:
                    result = session.execute(
                        text(query + f" LIMIT {self.batch_size}"), 
                        {"offset": current_offset}
                    )
                
                rows = result.fetchall()
                
                if not rows:
                    logger.info("No more records to stream")
                    break
                
                # Stream each transaction
                for row in rows:
                    if not self.is_streaming:
                        break
                    
                    transaction_data = self._row_to_transaction(row)
                    
                    # Send to Kafka
                    success = self.producer.send_transaction(transaction_data)
                    
                    if success:
                        records_streamed += 1
                        if records_streamed % 100 == 0:
                            logger.info(f"Streamed {records_streamed} records")
                    else:
                        logger.error(f"Failed to stream transaction {row.id}")
                
                current_offset += len(rows)
                
                # Check if we've reached the limit
                if limit and records_streamed >= limit:
                    break
                
                # Wait before next batch
                if self.is_streaming:
                    time.sleep(self.stream_interval)
            
            logger.info(f"Streaming completed. Total records streamed: {records_streamed}")
            
        except Exception as e:
            logger.error(f"Error in streaming worker: {e}")
        finally:
            session.close()
            self.is_streaming = False
    
    def _row_to_transaction(self, row) -> Dict[str, Any]:
        """Convert database row to transaction dictionary"""
        return {
            'transaction_id': row.id,
            'timestamp': row.created_at.isoformat() if row.created_at else datetime.utcnow().isoformat(),
            'amount': float(row.amount),
            'time_feature': float(row.time_feature),
            'features': {
                f'v{i}': float(getattr(row, f'v{i}')) for i in range(1, 29)
            },
            'actual_class': int(row.actual_class),
            'source': 'database_stream'
        }
    
    def get_streaming_status(self) -> Dict[str, Any]:
        """Get current streaming status"""
        return {
            'is_streaming': self.is_streaming,
            'has_producer': self.producer is not None,
            'thread_alive': self.stream_thread.is_alive() if self.stream_thread else False
        }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        session = self.session_factory()
        try:
            # Total transactions
            total_result = session.execute(text("SELECT COUNT(*) as count FROM transactions"))
            total_transactions = total_result.scalar()
            
            # Fraud vs Normal
            fraud_result = session.execute(text("SELECT actual_class, COUNT(*) as count FROM transactions GROUP BY actual_class"))
            class_counts = {row.actual_class: row.count for row in fraud_result}
            
            return {
                'total_transactions': total_transactions,
                'normal_transactions': class_counts.get(0, 0),
                'fraud_transactions': class_counts.get(1, 0),
                'fraud_percentage': round((class_counts.get(1, 0) / total_transactions * 100), 2) if total_transactions > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                'total_transactions': 0,
                'normal_transactions': 0,
                'fraud_transactions': 0,
                'fraud_percentage': 0
            }
        finally:
            session.close()
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal")
        self.stop_streaming()
        sys.exit(0)

def main():
    """Main function for running as standalone service"""
    logger.info("Starting Database Streaming Service")
    
    streamer = DatabaseStreamer()
    
    # For standalone mode, start streaming all records
    streamer.start_streaming()
    
    try:
        # Keep the main thread alive
        while streamer.is_streaming:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        streamer.stop_streaming()

if __name__ == '__main__':
    main()
