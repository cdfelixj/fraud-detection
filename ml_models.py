import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
from models import ModelPerformance, Prediction, FraudAlert, PredictionFeedback, Transaction
from app import db

class FraudDetectionModels:
    def __init__(self):
        self.isolation_forest = None
        self.logistic_model = None
        self.ensemble_weights = {'isolation': 0.4, 'logistic': 0.6}
        self.is_trained = False
        
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
        
        self.training_params = {
            'test_size': 0.2,
            'random_state': 42,
            'cross_validation': False
        }
        
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
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate performance"""
        try:
            # Train individual models
            if not self.train_isolation_forest(X_train):
                return False
                
            if not self.train_logistic_regression(X_train, y_train):
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
    
    def ensemble_predict(self, X):
        """Combine predictions from multiple models"""
        try:
            if not self.is_trained:
                logging.error("Models not trained yet")
                return None
            
            # Get individual predictions
            iso_result = self.predict_isolation_forest(X)
            log_result = self.predict_logistic(X)
            
            if iso_result is None or log_result is None:
                logging.error("One or more models failed to generate predictions")
                return None
                
            iso_scores, iso_preds = iso_result
            log_probs, log_preds = log_result
            
            if iso_scores is None or log_probs is None:
                logging.error("Invalid prediction results from models")
                return None
            
            # Ensemble scoring
            ensemble_scores = (
                self.ensemble_weights['isolation'] * iso_scores +
                self.ensemble_weights['logistic'] * log_probs
            )
            
            # Final predictions based on threshold
            threshold = 0.5
            final_predictions = (ensemble_scores > threshold).astype(int)
            
            results = {
                'isolation_scores': iso_scores,
                'logistic_probabilities': log_probs,
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
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"F1-Score: {f1:.4f}")
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"AUC: {auc:.4f}")
            
            # Save to database
            performance = ModelPerformance()
            performance.model_name = 'Ensemble'
            performance.precision_score = precision
            performance.recall_score = recall
            performance.f1_score = f1
            performance.auc_score = auc
            performance.accuracy_score = accuracy
            
            db.session.add(performance)
            db.session.commit()
            
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
            
            # Load the data processor to get the scaler
            from data_processor import DataProcessor
            self.data_processor = DataProcessor()
            self.data_processor.load_scaler()
            
            if self.isolation_forest and self.logistic_model:
                self.is_trained = True
                logging.info("Models loaded successfully")
                return True
            else:
                logging.warning("Some models could not be loaded")
                return False
                
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return False
    
    def predict_and_save_batch(self, transactions):
        """Predict on a batch of transactions and save results"""
        try:
            if not self.is_trained:
                logging.error("Models not trained yet")
                return False
            
            # Prepare features
            features = []
            for txn in transactions:
                feature_row = [
                    txn.time_feature,
                    txn.v1, txn.v2, txn.v3, txn.v4, txn.v5,
                    txn.v6, txn.v7, txn.v8, txn.v9, txn.v10,
                    txn.v11, txn.v12, txn.v13, txn.v14, txn.v15,
                    txn.v16, txn.v17, txn.v18, txn.v19, txn.v20,
                    txn.v21, txn.v22, txn.v23, txn.v24, txn.v25,
                    txn.v26, txn.v27, txn.v28, txn.amount
                ]
                features.append(feature_row)
            
            X = np.array(features)
            
            # Load scaler if needed
            from data_processor import DataProcessor
            data_proc = DataProcessor()
            if not hasattr(data_proc.scaler, "scale_"):
                data_proc.load_scaler()
            
            # Scale features
            X_scaled = data_proc.scaler.transform(X)
            
            # Get predictions
            results = self.ensemble_predict(X_scaled)
            if results is None:
                logging.error("Failed to get ensemble predictions")
                return False
            
            # Get individual model results
            iso_result = self.predict_isolation_forest(X_scaled)
            log_result = self.predict_logistic(X_scaled)
            
            iso_scores = iso_result[0] if iso_result is not None else None
            log_probs = log_result[0] if log_result is not None else None
            
            # Save predictions for each transaction
            predictions_saved = 0
            for i, txn in enumerate(transactions):
                try:
                    # Extract individual prediction results
                    individual_results = {
                        'isolation_scores': [iso_scores[i]] if iso_scores is not None else [0.0],
                        'logistic_probabilities': [log_probs[i]] if log_probs is not None else [0.0],
                        'ensemble_scores': [results['ensemble_scores'][i]],
                        'final_predictions': [results['final_predictions'][i]],
                        'confidence_scores': [results['confidence_scores'][i]]
                    }
                    
                    prediction_id = self.save_prediction(
                        transaction_id=txn.id,
                        prediction_results=individual_results,
                        model_version="1.0"
                    )
                    
                    if prediction_id:
                        predictions_saved += 1
                        
                except Exception as e:
                    logging.error(f"Error saving prediction for transaction {txn.id}: {str(e)}")
                    continue
            
            logging.info(f"Saved {predictions_saved}/{len(transactions)} predictions")
            return predictions_saved > 0
            
        except Exception as e:
            logging.error(f"Error in batch prediction: {str(e)}")
            return False

    def save_prediction(self, transaction_id, prediction_results, iso_scores=None, log_probs=None, model_version="1.0"):
        """Save prediction results to database for later validation"""
        try:
            # Handle both dictionary and direct array predictions
            if isinstance(prediction_results, dict):
                iso_score = float(prediction_results.get('isolation_scores', [0])[0]) if iso_scores is None else float(iso_scores[0])
                log_score = float(prediction_results.get('logistic_probabilities', [0])[0]) if log_probs is None else float(log_probs[0])
                ensemble_score = float(prediction_results.get('ensemble_scores', [0])[0])
                final_pred = int(prediction_results.get('final_predictions', [0])[0])
                confidence = float(prediction_results.get('confidence_scores', [0])[0])
            else:
                # Fallback for direct values
                ensemble_score = float(prediction_results)
                final_pred = 1 if ensemble_score > 0.5 else 0
                confidence = max(ensemble_score, 1 - ensemble_score)
                iso_score = float(iso_scores[0]) if iso_scores is not None and len(iso_scores) > 0 else 0.0
                log_score = float(log_probs[0]) if log_probs is not None and len(log_probs) > 0 else 0.0
            
            # Handle NaN values
            if np.isnan(iso_score):
                iso_score = 0.0
            if np.isnan(log_score):
                log_score = 0.0
            if np.isnan(ensemble_score):
                ensemble_score = 0.5
            if np.isnan(confidence):
                confidence = 0.5
            
            # Create prediction record
            prediction = Prediction()
            prediction.transaction_id = transaction_id
            prediction.isolation_forest_score = iso_score
            prediction.ensemble_prediction = ensemble_score
            prediction.final_prediction = final_pred
            prediction.confidence_score = confidence
            prediction.model_version = model_version
            
            db.session.add(prediction)
            db.session.commit()
            
            logging.info(f"Saved prediction for transaction {transaction_id}: {final_pred} (confidence: {confidence:.3f})")
            return prediction.id
            
        except Exception as e:
            logging.error(f"Error saving prediction: {str(e)}")
            return None

    def create_fraud_alerts(self, transactions, predictions):
        """Create fraud alerts for high-risk transactions"""
        try:
            alerts_created = 0
            
            # Handle both dictionary and direct array predictions
            if isinstance(predictions, dict):
                confidence_scores = predictions.get('ensemble_scores', [])
                final_predictions = predictions.get('final_predictions', [])
            else:
                # Assume predictions is the ensemble_scores array directly
                confidence_scores = predictions
                final_predictions = (np.array(predictions) > 0.5).astype(int)
            
            for i, txn in enumerate(transactions):
                if i >= len(confidence_scores) or i >= len(final_predictions):
                    continue
                    
                confidence = float(confidence_scores[i])
                final_pred = int(final_predictions[i])
                
                if final_pred == 1:  # Fraud detected
                    if confidence > 0.8:
                        alert_level = 'HIGH'
                    elif confidence > 0.6:
                        alert_level = 'MEDIUM'
                    else:
                        alert_level = 'LOW'
                    
                    alert = FraudAlert()
                    alert.transaction_id = txn.id
                    alert.alert_level = alert_level
                    alert.alert_reason = f"Fraud detected with {confidence:.2%} confidence"
                    
                    db.session.add(alert)
                    alerts_created += 1
            
            db.session.commit()
            logging.info(f"Created {alerts_created} fraud alerts")
            
        except Exception as e:
            logging.error(f"Error creating fraud alerts: {str(e)}")
    
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
            
            # Update ensemble weights
            if 'iso_weight' in params and 'log_weight' in params:
                self.ensemble_weights['isolation'] = params['iso_weight']
                self.ensemble_weights['logistic'] = params['log_weight']
            
            # Update training parameters
            if 'test_size' in params:
                self.training_params['test_size'] = params['test_size']
            if 'random_state' in params:
                self.training_params['random_state'] = params['random_state']
                self.iso_params['random_state'] = params['random_state']
                self.log_params['random_state'] = params['random_state']
            if 'cross_validation' in params:
                self.training_params['cross_validation'] = params['cross_validation']
            
            logging.info("Model parameters updated successfully")
            
        except Exception as e:
            logging.error(f"Error updating parameters: {str(e)}")

    def save_prediction_feedback(self, prediction_id, feedback_data):
        """Save user feedback on a prediction"""
        try:
            # Get the prediction and transaction
            prediction = Prediction.query.get(prediction_id)
            if not prediction:
                logging.error(f"Prediction {prediction_id} not found")
                return False
            
            # Create feedback record
            feedback = PredictionFeedback()
            feedback.prediction_id = prediction_id
            feedback.transaction_id = prediction.transaction_id
            feedback.user_feedback = feedback_data.get('feedback', 'uncertain')  # 'correct', 'incorrect', 'uncertain'
            feedback.actual_outcome = feedback_data.get('actual_outcome')  # 0 or 1 if known
            feedback.feedback_reason = feedback_data.get('reason', '')
            feedback.confidence_rating = feedback_data.get('confidence_rating', 3)  # 1-5 scale
            feedback.created_by = feedback_data.get('user_id', 'anonymous')
            
            db.session.add(feedback)
            
            # If actual outcome is provided, update the transaction
            if feedback.actual_outcome is not None:
                transaction = prediction.transaction
                if transaction and transaction.actual_class == -1:  # Only update if unknown
                    transaction.actual_class = feedback.actual_outcome
            
            db.session.commit()
            
            logging.info(f"Saved feedback for prediction {prediction_id}: {feedback.user_feedback}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving feedback: {str(e)}")
            db.session.rollback()
            return False
    
    def get_feedback_statistics(self):
        """Get statistics on user feedback"""
        try:
            # Total feedback count
            total_feedback = PredictionFeedback.query.count()
            
            # Feedback breakdown
            correct_feedback = PredictionFeedback.query.filter_by(user_feedback='correct').count()
            incorrect_feedback = PredictionFeedback.query.filter_by(user_feedback='incorrect').count()
            uncertain_feedback = PredictionFeedback.query.filter_by(user_feedback='uncertain').count()
            
            # Feedback with actual outcomes
            feedback_with_outcomes = PredictionFeedback.query.filter(
                PredictionFeedback.actual_outcome.isnot(None)
            ).count()
            
            # Agreement rate (when user feedback matches actual outcome)
            agreement_query = db.session.query(PredictionFeedback, Prediction).join(
                Prediction, PredictionFeedback.prediction_id == Prediction.id
            ).filter(
                PredictionFeedback.actual_outcome.isnot(None)
            ).all()
            
            total_with_outcome = len(agreement_query)
            user_correct = 0
            model_correct = 0
            both_correct = 0
            
            for feedback, prediction in agreement_query:
                actual = feedback.actual_outcome
                user_said_correct = feedback.user_feedback == 'correct'
                model_predicted = prediction.final_prediction
                
                if user_said_correct and model_predicted == actual:
                    both_correct += 1
                elif user_said_correct:
                    user_correct += 1
                elif model_predicted == actual:
                    model_correct += 1
            
            return {
                'total_feedback': total_feedback,
                'correct_feedback': correct_feedback,
                'incorrect_feedback': incorrect_feedback,
                'uncertain_feedback': uncertain_feedback,
                'feedback_with_outcomes': feedback_with_outcomes,
                'user_model_agreement': both_correct,
                'total_with_outcomes': total_with_outcome,
                'agreement_rate': both_correct / total_with_outcome if total_with_outcome > 0 else 0
            }
            
        except Exception as e:
            logging.error(f"Error getting feedback statistics: {str(e)}")
            return None
    
    def get_problematic_predictions(self, limit=20):
        """Get predictions that received negative feedback for analysis"""
        try:
            # Get predictions with 'incorrect' feedback
            problematic = db.session.query(
                Prediction, PredictionFeedback, Transaction
            ).join(
                PredictionFeedback, Prediction.id == PredictionFeedback.prediction_id
            ).join(
                Transaction, Prediction.transaction_id == Transaction.id
            ).filter(
                PredictionFeedback.user_feedback == 'incorrect'
            ).order_by(
                PredictionFeedback.created_at.desc()
            ).limit(limit).all()
            
            results = []
            for pred, feedback, txn in problematic:
                results.append({
                    'prediction_id': pred.id,
                    'transaction_id': txn.id,
                    'predicted_class': pred.final_prediction,
                    'actual_class': txn.actual_class,
                    'confidence': pred.confidence_score,
                    'ensemble_score': pred.ensemble_prediction,
                    'feedback_reason': feedback.feedback_reason,
                    'amount': txn.amount,
                    'created_at': pred.prediction_time
                })
            
            return results
            
        except Exception as e:
            logging.error(f"Error getting problematic predictions: {str(e)}")
            return []
    
    def retrain_with_feedback(self, use_feedback=True):
        """Retrain models incorporating user feedback"""
        try:
            from data_processor import DataProcessor
            data_processor = DataProcessor()
            
            # Get all transactions with ground truth
            transactions = Transaction.query.filter(
                Transaction.actual_class.in_([0, 1])
            ).all()
            
            if len(transactions) < 10:
                logging.error("Not enough transactions with ground truth for retraining")
                return False
            
            # If using feedback, prioritize transactions with feedback
            if use_feedback:
                # Get transactions that have feedback
                feedback_transaction_ids = db.session.query(
                    PredictionFeedback.transaction_id
                ).filter(
                    PredictionFeedback.actual_outcome.isnot(None)
                ).distinct().all()
                
                feedback_ids = [fid[0] for fid in feedback_transaction_ids]
                
                # Give more weight to transactions with feedback
                weighted_transactions = []
                for txn in transactions:
                    if txn.id in feedback_ids:
                        # Add feedback transactions multiple times to increase their weight
                        weighted_transactions.extend([txn] * 3)
                    else:
                        weighted_transactions.append(txn)
                
                transactions = weighted_transactions
                logging.info(f"Using {len(feedback_ids)} feedback transactions with higher weight")
            
            # Prepare features and labels
            features = []
            labels = []
            
            for txn in transactions:
                feature_row = [
                    txn.time_feature,
                    txn.v1, txn.v2, txn.v3, txn.v4, txn.v5,
                    txn.v6, txn.v7, txn.v8, txn.v9, txn.v10,
                    txn.v11, txn.v12, txn.v13, txn.v14, txn.v15,
                    txn.v16, txn.v17, txn.v18, txn.v19, txn.v20,
                    txn.v21, txn.v22, txn.v23, txn.v24, txn.v25,
                    txn.v26, txn.v27, txn.v28, txn.amount
                ]
                features.append(feature_row)
                labels.append(txn.actual_class)
            
            # Convert to arrays
            X = np.array(features)
            y = np.array(labels)
            
            # Prepare data for training
            X_scaled = data_processor.scaler.fit_transform(X)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Retrain models
            success = self.train_models(X_train, X_test, y_train, y_test)
            
            if success:
                logging.info("Successfully retrained models with feedback")
                
                # Save the updated scaler
                data_processor.save_scaler()
                
                return True
            else:
                logging.error("Failed to retrain models")
                return False
            
        except Exception as e:
            logging.error(f"Error retraining with feedback: {str(e)}")
            return False
