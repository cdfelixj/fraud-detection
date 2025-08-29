import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PostgresHandler:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None

    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.connection.autocommit = False
            logger.info("Database connection established.")
            return True
        except Exception as e:
            logger.error(f"Error connecting to the database: {e}")
            return False

    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed.")

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor"""
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            cursor.close()

    def create_tables(self):
        """Create necessary tables for fraud detection"""
        create_transactions_table = """
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            transaction_id VARCHAR(255) UNIQUE NOT NULL,
            amount DECIMAL(10, 2) NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            merchant_id VARCHAR(255),
            user_id VARCHAR(255),
            is_fraud BOOLEAN DEFAULT FALSE,
            fraud_probability DECIMAL(5, 4),
            model_prediction VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        create_predictions_table = """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            transaction_id VARCHAR(255) REFERENCES transactions(transaction_id),
            model_name VARCHAR(100) NOT NULL,
            prediction_value DECIMAL(5, 4),
            confidence_score DECIMAL(5, 4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            with self.get_cursor() as cursor:
                cursor.execute(create_transactions_table)
                cursor.execute(create_predictions_table)
            logger.info("Tables created successfully.")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")

    def insert(self, table, data):
        """Insert data into table"""
        try:
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['%s'] * len(data))
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            
            with self.get_cursor() as cursor:
                cursor.execute(query, list(data.values()))
            
            logger.info(f"Data inserted successfully into {table}.")
            return True
        except Exception as e:
            logger.error(f"Error inserting data into {table}: {e}")
            return False

    def insert_transaction(self, transaction_data):
        """Insert transaction with fraud prediction"""
        return self.insert('transactions', transaction_data)

    def insert_prediction(self, prediction_data):
        """Insert model prediction"""
        return self.insert('predictions', prediction_data)

    def query(self, query, params=None):
        """Execute a query and return results"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                # Convert RealDictRow to regular dict
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None

    def get_recent_transactions(self, limit=100):
        """Get recent transactions"""
        query = """
        SELECT * FROM transactions 
        ORDER BY timestamp DESC 
        LIMIT %s
        """
        return self.query(query, (limit,))

    def get_fraud_statistics(self):
        """Get fraud detection statistics"""
        query = """
        SELECT 
            COUNT(*) as total_transactions,
            COUNT(CASE WHEN is_fraud THEN 1 END) as fraud_transactions,
            AVG(fraud_probability) as avg_fraud_probability,
            COUNT(CASE WHEN fraud_probability > 0.5 THEN 1 END) as high_risk_transactions
        FROM transactions
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
        """
        result = self.query(query)
        if result and len(result) > 0:
            return result[0]
        return {
            'total_transactions': 0,
            'fraud_transactions': 0,
            'avg_fraud_probability': 0.0,
            'high_risk_transactions': 0
        }

    def health_check(self):
        """Check database connectivity"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False