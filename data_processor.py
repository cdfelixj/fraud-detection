import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models import Transaction
from app import db
import os

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [f'v{i}' for i in range(1, 29)] + ['time_feature', 'amount']
        
    def load_csv_data(self, filepath):
        """Load and validate CSV data"""
        try:
            if not os.path.exists(filepath):
                logging.error(f"CSV file not found: {filepath}")
                return None
                
            df = pd.read_csv(filepath)
            logging.info(f"Loaded CSV with shape: {df.shape}")
            
            # Validate required columns
            required_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logging.error(f"Missing required columns: {missing_cols}")
                return None
                
            return df
            
        except Exception as e:
            logging.error(f"Error loading CSV: {str(e)}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess the transaction data"""
        try:
            # Create a copy to avoid modifying original
            processed_df = df.copy()
            
            # Rename columns to match our model
            column_mapping = {'Time': 'time_feature', 'Amount': 'amount', 'Class': 'actual_class'}
            for i in range(1, 29):
                column_mapping[f'V{i}'] = f'v{i}'
                
            processed_df = processed_df.rename(columns=column_mapping)
            
            # Handle any missing values
            processed_df = processed_df.fillna(0)
            
            # Log class distribution
            class_counts = processed_df['actual_class'].value_counts()
            logging.info(f"Class distribution: Normal={class_counts.get(0, 0)}, Fraud={class_counts.get(1, 0)}")
            
            return processed_df
            
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            return None
    
    def save_to_database(self, df):
        """Save processed data to database"""
        try:
            # Clear existing data
            Transaction.query.delete()
            db.session.commit()
            
            # Convert DataFrame to Transaction records
            transactions = []
            for _, row in df.iterrows():
                transaction = Transaction()
                transaction.time_feature = float(row['time_feature'])
                transaction.v1 = float(row['v1'])
                transaction.v2 = float(row['v2'])
                transaction.v3 = float(row['v3'])
                transaction.v4 = float(row['v4'])
                transaction.v5 = float(row['v5'])
                transaction.v6 = float(row['v6'])
                transaction.v7 = float(row['v7'])
                transaction.v8 = float(row['v8'])
                transaction.v9 = float(row['v9'])
                transaction.v10 = float(row['v10'])
                transaction.v11 = float(row['v11'])
                transaction.v12 = float(row['v12'])
                transaction.v13 = float(row['v13'])
                transaction.v14 = float(row['v14'])
                transaction.v15 = float(row['v15'])
                transaction.v16 = float(row['v16'])
                transaction.v17 = float(row['v17'])
                transaction.v18 = float(row['v18'])
                transaction.v19 = float(row['v19'])
                transaction.v20 = float(row['v20'])
                transaction.v21 = float(row['v21'])
                transaction.v22 = float(row['v22'])
                transaction.v23 = float(row['v23'])
                transaction.v24 = float(row['v24'])
                transaction.v25 = float(row['v25'])
                transaction.v26 = float(row['v26'])
                transaction.v27 = float(row['v27'])
                transaction.v28 = float(row['v28'])
                transaction.amount = float(row['amount'])
                transaction.actual_class = int(row['actual_class'])
                transactions.append(transaction)
            
            # Bulk insert
            db.session.bulk_save_objects(transactions)
            db.session.commit()
            
            logging.info(f"Saved {len(transactions)} transactions to database")
            return True
            
        except Exception as e:
            logging.error(f"Error saving to database: {str(e)}")
            db.session.rollback()
            return False
    
    def get_feature_matrix(self, transactions=None):
        """Extract feature matrix from transactions"""
        try:
            if transactions is None:
                transactions = Transaction.query.all()
            
            if not transactions:
                return None, None
            
            # Extract features
            features = []
            labels = []
            
            for txn in transactions:
                feature_row = [
                    txn.time_feature,
                    txn.v1, txn.v2, txn.v3, txn.v4, txn.v5, txn.v6, txn.v7,
                    txn.v8, txn.v9, txn.v10, txn.v11, txn.v12, txn.v13, txn.v14,
                    txn.v15, txn.v16, txn.v17, txn.v18, txn.v19, txn.v20, txn.v21,
                    txn.v22, txn.v23, txn.v24, txn.v25, txn.v26, txn.v27, txn.v28,
                    txn.amount
                ]
                features.append(feature_row)
                labels.append(txn.actual_class)
            
            X = np.array(features)
            y = np.array(labels)
            
            return X, y
            
        except Exception as e:
            logging.error(f"Error creating feature matrix: {str(e)}")
            return None, None
    
    def prepare_training_data(self):
        """Prepare data for model training"""
        try:
            X, y = self.get_feature_matrix()
            
            if X is None or y is None:
                return None, None, None, None
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Save the fitted scaler
            self.save_scaler()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logging.info(f"Training data shape: {X_train.shape}")
            logging.info(f"Test data shape: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error preparing training data: {str(e)}")
            return None, None, None, None
    
    def save_scaler(self):
        """Save the fitted scaler"""
        try:
            joblib.dump(self.scaler, 'scaler.pkl')
            logging.info("Scaler saved successfully")
        except Exception as e:
            logging.error(f"Error saving scaler: {str(e)}")
    
    def load_scaler(self):
        """Load the fitted scaler"""
        try:
            if os.path.exists('scaler.pkl'):
                self.scaler = joblib.load('scaler.pkl')
                logging.info("Scaler loaded successfully")
                return True
            else:
                logging.warning("No saved scaler found")
                return False
        except Exception as e:
            logging.error(f"Error loading scaler: {str(e)}")
            return False
    
    def send_batch_to_kafka(self, processed_df):
        """Send processed DataFrame to Kafka for real-time processing"""
        try:
            from kafka_producer import fraud_producer
            from datetime import datetime
            
            # Convert DataFrame to transaction format for Kafka
            transactions_for_kafka = []
            for _, row in processed_df.iterrows():
                transaction_kafka = {
                    'transaction_id': f"upload_{int(datetime.now().timestamp() * 1000)}_{len(transactions_for_kafka)}",
                    'time_feature': float(row['time_feature']),
                    'amount': float(row['amount']),
                    'source': 'csv_upload',
                    'has_ground_truth': 'actual_class' in row and pd.notna(row['actual_class']),
                    **{f'v{i}': float(row[f'v{i}']) for i in range(1, 29)}
                }
                
                if 'actual_class' in row and pd.notna(row['actual_class']):
                    transaction_kafka['actual_class'] = int(row['actual_class'])
                
                transactions_for_kafka.append(transaction_kafka)
            
            # Send batch to Kafka
            kafka_results = fraud_producer.send_batch_transactions(transactions_for_kafka)
            logging.info(f"Kafka batch upload: {kafka_results['success']} sent, {kafka_results['failed']} failed")
            
            return kafka_results
            
        except Exception as e:
            logging.error(f"Error sending batch to Kafka: {str(e)}")
            return {'success': 0, 'failed': len(processed_df)}
    
    def process_single_transaction_for_kafka(self, features):
        """Process single transaction for Kafka streaming"""
        try:
            from kafka_producer import fraud_producer
            from datetime import datetime
            
            # Create transaction dictionary
            transaction_data = {
                'transaction_id': f"single_{int(datetime.now().timestamp() * 1000)}",
                'time_feature': float(features[0]),
                'amount': float(features[-1]),
                'source': 'manual_input',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add V1-V28 features
            for i in range(1, 29):
                idx = i  # features[0] is time_feature, features[1-28] are v1-v28, features[29] is amount
                transaction_data[f'v{i}'] = float(features[idx]) if idx < len(features) - 1 else 0
            
            # Send to Kafka
            success = fraud_producer.send_transaction(transaction_data)
            
            if success:
                logging.info(f"Single transaction sent to Kafka: {transaction_data['transaction_id']}")
                return transaction_data['transaction_id']
            else:
                logging.error("Failed to send transaction to Kafka")
                return None
                
        except Exception as e:
            logging.error(f"Error processing single transaction for Kafka: {str(e)}")
            return None
