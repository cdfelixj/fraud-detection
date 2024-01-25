import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

class IsolationForestModel:
    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def fit(self, X):
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True

    def predict(self, X):
        if not self.is_trained:
            # Return dummy prediction for untrained model
            if len(X.shape) == 1:
                return np.array([-1])  # Normal prediction
            return np.array([-1] * len(X))
        
        # Handle single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Convert to probability scores
        scores = self.model.decision_function(X_scaled)
        return predictions, scores

    def predict_proba(self, X):
        if not self.is_trained:
            if len(X.shape) == 1:
                return np.array([[0.9, 0.1]])  # [normal_prob, fraud_prob]
            return np.array([[0.9, 0.1]] * len(X))
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        
        # Convert scores to probabilities (higher scores = more normal)
        proba = np.zeros((len(scores), 2))
        for i, score in enumerate(scores):
            fraud_prob = max(0, min(1, (0.5 - score) / 1.0))  # Normalize score
            proba[i] = [1 - fraud_prob, fraud_prob]
        
        return proba

    def save_model(self, filepath):
        if self.is_trained:
            joblib.dump(self.model, f"{filepath}_model.pkl")
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")

    def load_model(self, filepath):
        if os.path.exists(f"{filepath}_model.pkl"):
            self.model = joblib.load(f"{filepath}_model.pkl")
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
            self.is_trained = True