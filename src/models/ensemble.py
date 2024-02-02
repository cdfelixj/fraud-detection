import numpy as np
from typing import List, Any, Optional

class EnsembleModel:
    def __init__(self, models: Optional[List[Any]] = None, weights: Optional[List[float]] = None):
        self.models = models or []
        self.weights = weights or [1.0] * len(self.models) if self.models else []

    def add_model(self, model, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.weights.append(weight)

    def fit(self, X, y):
        """Fit all models in the ensemble"""
        for model in self.models:
            if hasattr(model, 'fit'):
                model.fit(X, y)

    def predict(self, X):
        """Make ensemble predictions"""
        if not self.models:
            # Return dummy prediction if no models
            if hasattr(X, '__len__') and len(X) > 0:
                return np.array([0.1] * len(X))  # Low fraud probability
            return np.array([0.1])
        
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                # Get probability predictions
                pred = model.predict_proba(X)
                if pred.ndim > 1:
                    pred = pred[:, 1]  # Get fraud probability
                predictions.append(pred)
            elif hasattr(model, 'predict'):
                # Get binary predictions and convert to probabilities
                pred = model.predict(X)
                if isinstance(pred, tuple):
                    pred = pred[1]  # Get scores if tuple
                # Convert to probabilities (assume negative values are normal)
                pred_proba = np.where(pred == -1, 0.1, 0.9)
                predictions.append(pred_proba)
        
        if not predictions:
            # Return dummy if no valid predictions
            if hasattr(X, '__len__') and len(X) > 0:
                return np.array([0.1] * len(X))
            return np.array([0.1])
        
        return self.aggregate_predictions(predictions)

    def aggregate_predictions(self, predictions):
        """Aggregate predictions using weighted average"""
        if not predictions:
            return np.array([0.1])
        
        # Normalize weights
        total_weight = sum(self.weights[:len(predictions)])
        normalized_weights = [w / total_weight for w in self.weights[:len(predictions)]]
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, normalized_weights):
            weighted_pred += np.array(pred) * weight
        
        return weighted_pred

    def predict_fraud_probability(self, X):
        """Get fraud probability predictions"""
        predictions = self.predict(X)
        return predictions

    def get_model_count(self):
        """Get number of models in ensemble"""
        return len(self.models)