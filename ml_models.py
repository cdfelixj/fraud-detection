import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os

class FraudDetectionModels:
    def __init__(self):
        self.isolation_forest = None
        self.logistic_model = None
        self.xgboost_model = None
        self.ensemble_weights = {'isolation': 0.3, 'logistic': 0.3, 'xgboost': 0.4}
        self.is_trained = False
        
        # Cache for database imports
        self._db = None
        self._models = None
        
        # Initialize parameter dictionaries
        self.iso_params = {
            'contamination': 0.002,
            'n_estimators': 200,
            'max_samples': 'auto',
            'random_state': 42
        }
        
        self.log_params = {
            'max_iter': 1000,
            'solver': 'liblinear',
            'penalty': 'l2',
            'random_state': 42
        }
        
        self.xgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        self.training_params = {
            'test_size': 0.2,
            'random_state': 42,
            'cross_validation': False,
            'use_smote': True
        }
    
    def _get_db_imports(self):
        """Lazy import of database models to avoid circular imports"""
        if self._db is None or self._models is None:
            try:
                from app import db
                from models import ModelPerformance, Prediction, FraudAlert, PredictionFeedback, Transaction
                self._db = db
                self._models = {
                    'ModelPerformance': ModelPerformance,
                    'Prediction': Prediction,
                    'FraudAlert': FraudAlert,
                    'PredictionFeedback': PredictionFeedback,
                    'Transaction': Transaction
                }
            except ImportError as e:
                logging.warning(f"Database imports not available: {e}")
                return None, None, None, None, None
        # Return in order: db, Transaction, Prediction, PredictionFeedback, FraudAlert, ModelPerformance
        return (self._db, 
                self._models['Transaction'], 
                self._models['Prediction'], 
                self._models['PredictionFeedback'], 
                self._models['FraudAlert'], 
                self._models['ModelPerformance'])
        
    def train_isolation_forest(self, X_train, contamination=0.002):
        """Train Isolation Forest for anomaly detection"""
        try:
            logging.info("Training Isolation Forest...")
            
            self.isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=200,
                max_samples='auto',
                max_features=1.0
            )
            
            # Fit on normal transactions only (class 0)
            self.isolation_forest.fit(X_train)
            
            logging.info("Isolation Forest training completed")
            return True
            
        except Exception as e:
            logging.error(f"Error training Isolation Forest: {str(e)}")
            return False
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression with class balancing"""
        try:
            logging.info("Training Logistic Regression...")
            
            # Compute class weights to handle imbalance
            classes = np.unique(y_train)
            class_weights = compute_class_weight(
                'balanced',
                classes=classes,
                y=y_train
            )
            class_weight_dict = dict(zip(classes, class_weights))
            
            self.logistic_model = LogisticRegression(
                class_weight=class_weight_dict,
                random_state=42,
                max_iter=1000,
                solver='liblinear'
            )
            
            self.logistic_model.fit(X_train, y_train)
            
            logging.info("Logistic Regression training completed")
            return True
            
        except Exception as e:
            logging.error(f"Error training Logistic Regression: {str(e)}")
            return False
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost with class balancing and SMOTE"""
        try:
            logging.info("Training XGBoost...")
            
            # Apply SMOTE for better class balance if enabled
            if self.training_params.get('use_smote', True):
                logging.info("Applying SMOTE for class balancing...")
                smote = SMOTE(random_state=42, k_neighbors=min(5, sum(y_train == 1) - 1) if sum(y_train == 1) > 1 else 1)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                logging.info(f"SMOTE applied: {X_train.shape} -> {X_train_balanced.shape}")
            else:
                X_train_balanced, y_train_balanced = X_train, y_train
            
            # Calculate scale_pos_weight for imbalanced classes
            neg_count = sum(y_train_balanced == 0)
            pos_count = sum(y_train_balanced == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
            
            # Initialize XGBoost with optimized parameters for fraud detection
            self.xgboost_model = xgb.XGBClassifier(
                n_estimators=self.xgb_params['n_estimators'],
                max_depth=self.xgb_params['max_depth'],
                learning_rate=self.xgb_params['learning_rate'],
                subsample=self.xgb_params['subsample'],
                colsample_bytree=self.xgb_params['colsample_bytree'],
                scale_pos_weight=scale_pos_weight,
                random_state=self.xgb_params['random_state'],
                eval_metric=self.xgb_params['eval_metric'],
                use_label_encoder=self.xgb_params['use_label_encoder'],
                tree_method='hist',  # Faster training
                early_stopping_rounds=10,
                verbosity=0  # Suppress XGBoost warnings
            )
            
            # Train the model
            self.xgboost_model.fit(
                X_train_balanced, 
                y_train_balanced,
                eval_set=[(X_train_balanced, y_train_balanced)],
                verbose=False
            )
            
            logging.info("XGBoost training completed")
            return True
            
        except Exception as e:
            logging.error(f"Error training XGBoost: {str(e)}")
            return False
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate performance"""
        try:
            # Train individual models
            if not self.train_isolation_forest(X_train):
                return False
                
            if not self.train_logistic_regression(X_train, y_train):
                return False
            
            if not self.train_xgboost(X_train, y_train):
                return False
            
            # Evaluate models
            self.evaluate_models(X_test, y_test)
            
            # Save models
            self.save_models()
            
            self.is_trained = True
            logging.info("All models trained successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
            return False
    
    def predict_isolation_forest(self, X):
        """Get Isolation Forest predictions"""
        if self.isolation_forest is None:
            return None
            
        # Get anomaly scores (-1 for outliers, 1 for inliers)
        scores = self.isolation_forest.decision_function(X)
        predictions = self.isolation_forest.predict(X)
        
        # Convert to probability-like scores (0-1 range)
        # Higher scores indicate higher fraud probability
        score_range = scores.max() - scores.min()
        if score_range == 0:
            # All scores are the same, return neutral scores
            normalized_scores = np.full(scores.shape, 0.5)
        else:
            normalized_scores = (1 - (scores - scores.min()) / score_range)
        
        return normalized_scores, predictions
    
    def predict_logistic(self, X):
        """Get Logistic Regression predictions"""
        if self.logistic_model is None:
            return None
            
        probabilities = self.logistic_model.predict_proba(X)[:, 1]  # Probability of fraud
        predictions = self.logistic_model.predict(X)
        
        return probabilities, predictions
    
    def predict_xgboost(self, X):
        """Get XGBoost predictions"""
        if self.xgboost_model is None:
            return None
            
        probabilities = self.xgboost_model.predict_proba(X)[:, 1]  # Probability of fraud
        predictions = self.xgboost_model.predict(X)
        
        return probabilities, predictions
    
    def ensemble_predict(self, X):
        """Combine predictions from multiple models"""
        try:
            if not self.is_trained:
                logging.error("Models not trained yet")
                return None
            
            # Get individual predictions
            iso_result = self.predict_isolation_forest(X)
            log_result = self.predict_logistic(X)
            xgb_result = self.predict_xgboost(X)
            
            if iso_result is None or log_result is None or xgb_result is None:
                logging.error("One or more models failed to generate predictions")
                return None
                
            iso_scores, iso_preds = iso_result
            log_probs, log_preds = log_result
            xgb_probs, xgb_preds = xgb_result
            
            if iso_scores is None or log_probs is None or xgb_probs is None:
                logging.error("Invalid prediction results from models")
                return None
            
            # Ensemble scoring with three models
            ensemble_scores = (
                self.ensemble_weights['isolation'] * iso_scores +
                self.ensemble_weights['logistic'] * log_probs +
                self.ensemble_weights['xgboost'] * xgb_probs
            )
            
            # Final predictions based on threshold
            threshold = 0.5
            final_predictions = (ensemble_scores > threshold).astype(int)
            
            results = {
                'isolation_scores': iso_scores,
                'logistic_probabilities': log_probs,
                'xgboost_probabilities': xgb_probs,
                'ensemble_scores': ensemble_scores,
                'final_predictions': final_predictions,
                'confidence_scores': np.maximum(ensemble_scores, 1 - ensemble_scores)
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in ensemble prediction: {str(e)}")
            return None
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate model performance and save metrics"""
        try:
            # Temporarily set is_trained to True for evaluation
            temp_trained = self.is_trained
            self.is_trained = True
            
            results = self.ensemble_predict(X_test)
            
            # Restore original trained status
            self.is_trained = temp_trained
            
            if results is None:
                logging.error("Could not get predictions for evaluation")
                return
            
            y_pred = results['final_predictions']
            y_scores = results['ensemble_scores']
            
            # Calculate metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            
            try:
                auc = roc_auc_score(y_test, y_scores)
            except ValueError:
                auc = 0.0  # Handle case where only one class in test set
            
            # Log metrics
            logging.info(f"Model Performance:")
            logging.info(f"Precision: {precision:.7f}")
            logging.info(f"Recall: {recall:.7f}")
            logging.info(f"F1-Score: {f1:.7f}")
            logging.info(f"Accuracy: {accuracy:.7f}")
            logging.info(f"AUC: {auc:.4f}")
            
            # Save to database
            try:
                db_imports = self._get_db_imports()
                if db_imports[0]:
                    db, ModelPerformance = db_imports[0], db_imports[5]
                    
                    performance = ModelPerformance()
                    performance.model_name = 'Ensemble'
                    performance.precision_score = precision
                    performance.recall_score = recall
                    performance.f1_score = f1
                    performance.auc_score = auc
                    performance.accuracy_score = accuracy
                    
                    db.session.add(performance)
                    db.session.commit()
                else:
                    logging.warning("Database not available for saving performance metrics")
            except Exception as db_error:
                logging.error(f"Database error saving performance: {db_error}")
            
        except Exception as e:
            logging.error(f"Error evaluating models: {str(e)}")
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs('saved_models', exist_ok=True)
            
            if self.isolation_forest:
                joblib.dump(self.isolation_forest, 'saved_models/isolation_forest.pkl')
                
            if self.logistic_model:
                joblib.dump(self.logistic_model, 'saved_models/logistic_model.pkl')
            
            if self.xgboost_model:
                joblib.dump(self.xgboost_model, 'saved_models/xgboost_model.pkl')
            
            logging.info("Models saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            if os.path.exists('saved_models/isolation_forest.pkl'):
                self.isolation_forest = joblib.load('saved_models/isolation_forest.pkl')
                
            if os.path.exists('saved_models/logistic_model.pkl'):
                self.logistic_model = joblib.load('saved_models/logistic_model.pkl')
            
            if os.path.exists('saved_models/xgboost_model.pkl'):
                self.xgboost_model = joblib.load('saved_models/xgboost_model.pkl')
            
            # Load the data processor to get the scaler
            from data_processor import DataProcessor
            self.data_processor = DataProcessor()
            self.data_processor.load_scaler()
            
            if self.isolation_forest and self.logistic_model and self.xgboost_model:
                self.is_trained = True
                logging.info("Models loaded successfully")
                return True
            else:
                logging.warning("Some models could not be loaded")
                return False
                
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return False
    
    def update_parameters(self, params):
        """Update model parameters from user input"""
        try:
            # Update isolation forest parameters
            if 'contamination' in params:
                self.iso_params['contamination'] = params['contamination']
            if 'n_estimators_iso' in params:
                self.iso_params['n_estimators'] = params['n_estimators_iso']
            if 'max_samples' in params:
                self.iso_params['max_samples'] = params['max_samples'] if params['max_samples'] != 'auto' else 'auto'
            
            # Update logistic regression parameters
            if 'max_iter' in params:
                self.log_params['max_iter'] = params['max_iter']
            if 'solver' in params:
                self.log_params['solver'] = params['solver']
            if 'penalty' in params:
                self.log_params['penalty'] = params['penalty']
            
            # Update XGBoost parameters
            if 'n_estimators_xgb' in params:
                self.xgb_params['n_estimators'] = params['n_estimators_xgb']
            if 'max_depth' in params:
                self.xgb_params['max_depth'] = params['max_depth']
            if 'learning_rate' in params:
                self.xgb_params['learning_rate'] = params['learning_rate']
            if 'subsample' in params:
                self.xgb_params['subsample'] = params['subsample']
            if 'colsample_bytree' in params:
                self.xgb_params['colsample_bytree'] = params['colsample_bytree']
            
            # Update ensemble weights (now with three models)
            if 'iso_weight' in params and 'log_weight' in params and 'xgb_weight' in params:
                total_weight = params['iso_weight'] + params['log_weight'] + params['xgb_weight']
                if total_weight > 0:
                    self.ensemble_weights['isolation'] = params['iso_weight'] / total_weight
                    self.ensemble_weights['logistic'] = params['log_weight'] / total_weight
                    self.ensemble_weights['xgboost'] = params['xgb_weight'] / total_weight
            
            # Update training parameters
            if 'test_size' in params:
                self.training_params['test_size'] = params['test_size']
            if 'random_state' in params:
                self.training_params['random_state'] = params['random_state']
                self.iso_params['random_state'] = params['random_state']
                self.log_params['random_state'] = params['random_state']
                self.xgb_params['random_state'] = params['random_state']
            if 'cross_validation' in params:
                self.training_params['cross_validation'] = params['cross_validation']
            if 'use_smote' in params:
                self.training_params['use_smote'] = params['use_smote']
            
            logging.info("Model parameters updated successfully")
            
        except Exception as e:
            logging.error(f"Error updating parameters: {str(e)}")
    
    def predict_and_save_batch(self, transactions):
        """
        Make predictions for a batch of transactions and save to database
        
        Args:
            transactions: List of transaction dictionaries or DataFrame
        
        Returns:
            List of prediction results
        """
        try:
            import pandas as pd
            # Import database models when needed
            db, Transaction, Prediction = self._get_db_imports()[:3]
            
            # Convert to DataFrame if not already
            if not isinstance(transactions, pd.DataFrame):
                df = pd.DataFrame(transactions)
            else:
                df = transactions.copy()
            
            # Prepare features for prediction using actual Transaction schema
            feature_columns = ['time_feature'] + [f'v{i}' for i in range(1, 29)] + ['amount']
            
            # For existing transactions, extract features from the transaction objects
            X_list = []
            for i, (_, row) in enumerate(df.iterrows()):
                if 'id' in row and row['id']:
                    # Get from existing transaction in database
                    transaction = Transaction.query.get(row['id'])
                    if transaction:
                        features = [transaction.time_feature] + [getattr(transaction, f'v{i}') for i in range(1, 29)] + [transaction.amount]
                        X_list.append(features)
                    else:
                        # Use defaults if transaction not found
                        features = [0.0] + [0.0] * 28 + [row.get('amount', 0.0)]
                        X_list.append(features)
                else:
                    # Use provided features or defaults
                    features = [
                        row.get('time_feature', 0.0)
                    ] + [
                        row.get(f'v{i}', 0.0) for i in range(1, 29)
                    ] + [
                        row.get('amount', 0.0)
                    ]
                    X_list.append(features)
            
            import numpy as np
            X = np.array(X_list)
            
            # Apply scaling - the models were trained on scaled data
            try:
                if hasattr(self, 'data_processor') and self.data_processor and hasattr(self.data_processor, 'scaler'):
                    X_scaled = self.data_processor.scaler.transform(X)
                else:
                    # Load data processor with scaler if not available
                    from data_processor import DataProcessor
                    data_proc = DataProcessor()
                    data_proc.load_scaler()
                    X_scaled = data_proc.scaler.transform(X)
            except Exception as e:
                logging.warning(f"Could not apply scaling: {e}, using unscaled features")
                X_scaled = X
            
            # Make ensemble predictions
            predictions = self.ensemble_predict(X_scaled)
            
            if predictions is None:
                logging.error("Failed to generate ensemble predictions")
                return []
            
            results = []
            
            for i, (_, row) in enumerate(df.iterrows()):
                try:
                    # Create or get transaction record
                    if 'id' in row and row['id']:
                        # Transaction already exists, use the existing one
                        transaction = Transaction.query.get(row['id'])
                        if not transaction:
                            logging.error(f"Transaction with ID {row['id']} not found")
                            continue
                    else:
                        # Create new transaction (rarely used for batch processing)
                        # Get next transaction_id based on existing count
                        next_transaction_id = str(Transaction.query.count() + 1)
                        
                        transaction = Transaction(
                            transaction_id=row.get('transaction_id', next_transaction_id),
                            time_feature=float(row.get('time_feature', 0.0)),
                            amount=float(row['amount']),
                            actual_class=int(row.get('actual_class', 0))
                        )
                        # Set V1-V28 features
                        for v_idx in range(1, 29):
                            setattr(transaction, f'v{v_idx}', float(row.get(f'v{v_idx}', 0.0)))
                        
                        db.session.add(transaction)
                        db.session.flush()  # Get the ID
                    
                    # Extract prediction data for this transaction
                    isolation_score = float(predictions['isolation_scores'][i]) if predictions['isolation_scores'] is not None else 0.0
                    logistic_score = float(predictions['logistic_probabilities'][i]) if predictions['logistic_probabilities'] is not None else 0.0
                    xgboost_score = float(predictions['xgboost_probabilities'][i]) if predictions['xgboost_probabilities'] is not None else 0.0
                    ensemble_score = float(predictions['ensemble_scores'][i]) if predictions['ensemble_scores'] is not None else 0.0
                    is_fraud = bool(predictions['final_predictions'][i]) if predictions['final_predictions'] is not None else False
                    confidence = float(predictions['confidence_scores'][i]) if predictions['confidence_scores'] is not None else 0.5
                    
                    # Create prediction record
                    prediction = Prediction(
                        transaction_id=transaction.id,
                        final_prediction=1 if is_fraud else 0,
                        isolation_forest_score=isolation_score,
                        logistic_regression_score=logistic_score,
                        xgboost_score=xgboost_score,
                        ensemble_prediction=ensemble_score,
                        confidence_score=confidence,
                        model_version='ensemble_v1.0'
                    )
                    db.session.add(prediction)
                    
                    results.append({
                        'transaction_id': transaction.transaction_id or f'txn_{transaction.id}',
                        'prediction': is_fraud,
                        'confidence': confidence,
                        'ensemble_score': ensemble_score,
                        'isolation_score': isolation_score,
                        'logistic_score': logistic_score,
                        'xgboost_score': xgboost_score
                    })
                    
                except Exception as e:
                    logging.error(f"Error processing transaction {i}: {str(e)}")
                    continue
            
            # Commit all changes
            db.session.commit()
            logging.info(f"Successfully processed {len(results)} transactions")
            
            return results
            
        except Exception as e:
            # Import database models when needed
            db = self._get_db_imports()[0]
            db.session.rollback()
            logging.error(f"Error in batch prediction: {str(e)}")
            return []
    
    def get_feedback_statistics(self):
        """Get feedback statistics for the dashboard"""
        try:
            # Import database models when needed
            db_imports = self._get_db_imports()
            db, PredictionFeedback = db_imports[0], db_imports[3]
            
            total_feedback = PredictionFeedback.query.count()
            correct_feedback = PredictionFeedback.query.filter_by(
                user_feedback='correct'
            ).count()
            incorrect_feedback = PredictionFeedback.query.filter_by(
                user_feedback='incorrect'
            ).count()
            
            # Calculate agreement rate
            agreement_rate = 0.0
            if total_feedback > 0:
                agreement_rate = (correct_feedback / total_feedback) * 100
            
            return {
                'total_feedback': total_feedback,
                'correct_feedback': correct_feedback,
                'incorrect_feedback': incorrect_feedback,
                'agreement_rate': round(agreement_rate, 2)
            }
            
        except Exception as e:
            logging.error(f"Error getting feedback statistics: {str(e)}")
            return {
                'total_feedback': 0,
                'correct_feedback': 0,
                'incorrect_feedback': 0,
                'agreement_rate': 0.0
            }
    
    def get_problematic_predictions(self, limit=10):
        """Get predictions that have been marked as incorrect for review"""
        try:
            # Import database models when needed
            db, Transaction, Prediction, PredictionFeedback = self._get_db_imports()
            
            # Query for predictions marked as incorrect
            problematic = db.session.query(
                Prediction, Transaction, PredictionFeedback
            ).join(
                Transaction, Prediction.transaction_id == Transaction.id
            ).join(
                PredictionFeedback, Prediction.id == PredictionFeedback.prediction_id
            ).filter(
                PredictionFeedback.user_feedback == 'incorrect'
            ).order_by(
                PredictionFeedback.created_at.desc()
            ).limit(limit).all()
            
            result = []
            for prediction, transaction, feedback in problematic:
                result.append({
                    'prediction_id': prediction.id,
                    'transaction_id': transaction.id,
                    'amount': float(transaction.amount),
                    'predicted_fraud': prediction.final_prediction,
                    'isolation_score': prediction.isolation_forest_score,
                    'logistic_score': prediction.logistic_regression_score,
                    'xgboost_score': prediction.xgboost_score,
                    'ensemble_score': prediction.ensemble_prediction,
                    'feedback_date': feedback.created_at.isoformat(),
                    'feedback_comment': feedback.comment
                })
            
            return result
            
        except Exception as e:
            logging.error(f"Error getting problematic predictions: {str(e)}")
            return []
    
    def create_fraud_alerts(self, fraud_transactions, prediction_results):
        """
        Create fraud alerts for transactions predicted as fraudulent
        
        Args:
            fraud_transactions: List of transactions predicted as fraud
            prediction_results: Dictionary containing prediction results
        """
        try:
            # Import database models when needed
            db, Transaction, Prediction, FraudAlert = self._get_db_imports()[:4]
            
            alerts_created = 0
            
            for i, transaction in enumerate(fraud_transactions):
                try:
                    # Get the prediction result for this transaction
                    if isinstance(prediction_results, dict):
                        if 'final_predictions' in prediction_results:
                            is_fraud = prediction_results['final_predictions'][i] == 1
                            confidence = prediction_results.get('confidence_scores', [0.5])[i] if 'confidence_scores' in prediction_results else 0.5
                            ensemble_score = prediction_results.get('ensemble_scores', [0.5])[i] if 'ensemble_scores' in prediction_results else 0.5
                        else:
                            continue
                    else:
                        continue
                    
                    # Only create alerts for high-confidence fraud predictions
                    if is_fraud and confidence > 0.7:
                        # For batch processing, transactions will be dictionaries with 'id'
                        # For routes.py calls, they might be Transaction objects
                        transaction_id = None
                        amount = 0
                        
                        if isinstance(transaction, dict):
                            transaction_id = transaction.get('id')
                            amount = float(transaction.get('amount', 0))
                        else:
                            # Assume it's a Transaction object
                            transaction_id = transaction.id
                            amount = float(transaction.amount)
                        
                        if not transaction_id:
                            continue
                            
                        # Check if alert already exists for this transaction
                        existing_alert = FraudAlert.query.filter_by(
                            transaction_id=transaction_id
                        ).first()
                        
                        if not existing_alert:
                            # Determine alert level based on confidence and amount
                            if confidence > 0.95 or amount > 10000:
                                alert_level = 'HIGH'
                            elif confidence > 0.85 or amount > 5000:
                                alert_level = 'MEDIUM'
                            else:
                                alert_level = 'LOW'
                            
                            # Create fraud alert
                            alert = FraudAlert(
                                transaction_id=transaction_id,
                                alert_level=alert_level,
                                alert_reason=f"High-confidence fraud prediction (confidence: {confidence:.2f}, amount: ${amount:.2f})"
                            )
                            db.session.add(alert)
                            alerts_created += 1
                
                except Exception as e:
                    logging.error(f"Error creating alert for transaction {i}: {str(e)}")
                    continue
            
            if alerts_created > 0:
                db.session.commit()
                logging.info(f"Created {alerts_created} fraud alerts")
            
            return alerts_created
            
        except Exception as e:
            # Import database models when needed
            db = self._get_db_imports()[0]
            db.session.rollback()
            logging.error(f"Error creating fraud alerts: {str(e)}")
            return 0
    
    def assign_transaction_ids(self):
        """
        Assign transaction_ids to transactions that don't have them yet.
        Uses the order of creation (based on database ID) as the transaction_id.
        """
        try:
            # Import database models when needed
            db, Transaction = self._get_db_imports()[:2]
            
            # Get transactions without transaction_id, ordered by ID (creation order)
            transactions_without_id = Transaction.query.filter(
                Transaction.transaction_id.is_(None)
            ).order_by(Transaction.id).all()
            
            if not transactions_without_id:
                logging.info("All transactions already have transaction_ids")
                return 0
            
            # Get the highest existing transaction_id (as integer)
            max_existing = db.session.query(Transaction.transaction_id).filter(
                Transaction.transaction_id.isnot(None)
            ).all()
            
            next_id = 1
            if max_existing:
                try:
                    max_id = max([int(tid[0]) for tid in max_existing if tid[0] and tid[0].isdigit()])
                    next_id = max_id + 1
                except (ValueError, TypeError):
                    # If existing transaction_ids are not numeric, start from transaction count
                    next_id = Transaction.query.filter(Transaction.transaction_id.isnot(None)).count() + 1
            
            # Assign transaction_ids in order
            updated_count = 0
            for transaction in transactions_without_id:
                transaction.transaction_id = str(next_id)
                next_id += 1
                updated_count += 1
            
            db.session.commit()
            logging.info(f"Assigned transaction_ids to {updated_count} transactions")
            return updated_count
            
        except Exception as e:
            # Import database models when needed
            db = self._get_db_imports()[0]
            db.session.rollback()
            logging.error(f"Error assigning transaction_ids: {str(e)}")
            return 0
