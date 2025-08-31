import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
from models import ModelPerformance, Prediction, FraudAlert
from app import db

class FraudDetectionModels:
    def __init__(self):
        self.isolation_forest = None
        self.logistic_model = None
        self.ensemble_weights = {'isolation': 0.4, 'logistic': 0.6}
        self.is_trained = False
        
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
        normalized_scores = (1 - (scores - scores.min()) / (scores.max() - scores.min()))
        
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
            iso_scores, iso_preds = self.predict_isolation_forest(X)
            log_probs, log_preds = self.predict_logistic(X)
            
            if iso_scores is None or log_probs is None:
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
    
    def create_fraud_alerts(self, transactions, predictions):
        """Create fraud alerts for high-risk transactions"""
        try:
            alerts_created = 0
            
            for i, (txn, pred_data) in enumerate(zip(transactions, predictions)):
                confidence = pred_data.get('confidence_scores', [0])[i] if isinstance(pred_data.get('confidence_scores'), np.ndarray) else pred_data.get('confidence_scores', 0)
                final_pred = pred_data.get('final_predictions', [0])[i] if isinstance(pred_data.get('final_predictions'), np.ndarray) else pred_data.get('final_predictions', 0)
                
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
