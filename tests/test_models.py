from src.models.isolation_forest import IsolationForestModel
from src.models.lstm_model import LSTMModel  
from src.models.ensemble import EnsembleModel
import numpy as np
import pytest

def test_isolation_forest():
    model = IsolationForestModel()
    X_train = np.random.rand(100, 10)
    model.fit(X_train)
    predictions, scores = model.predict(X_train)
    assert len(predictions) == 100
    assert len(scores) == 100

def test_isolation_forest_untrained():
    model = IsolationForestModel()
    X_test = np.random.rand(10, 5)
    predictions, scores = model.predict(X_test)
    assert len(predictions) == 10

def test_lstm_model_untrained():
    model = LSTMModel(input_shape=(10, 5))
    X_test = np.random.rand(10, 5)
    predictions = model.predict(X_test)
    assert len(predictions) == 1  # Reshaped to (1, 10, 5)

def test_lstm_model_with_training():
    model = LSTMModel(input_shape=(10, 5))
    X_train = np.random.rand(100, 10, 5)
    y_train = np.random.randint(0, 2, (100, 1))
    
    # Build and train model
    model.model = model.build_model()
    model.train(X_train, y_train, epochs=1, validation_split=0.1)
    
    X_test = np.random.rand(5, 10, 5)
    predictions = model.predict(X_test)
    assert len(predictions) == 5

def test_ensemble_model():
    model1 = IsolationForestModel()
    model2 = LSTMModel()
    ensemble_model = EnsembleModel()
    
    # Add models to ensemble
    ensemble_model.add_model(model1, weight=0.6)
    ensemble_model.add_model(model2, weight=0.4)
    
    X_test = np.random.rand(10, 5)
    predictions = ensemble_model.predict(X_test)
    assert len(predictions) == 10

def test_ensemble_model_empty():
    ensemble_model = EnsembleModel()
    X_test = np.random.rand(5, 5)
    predictions = ensemble_model.predict(X_test)
    assert len(predictions) == 5