from src.models.isolation_forest import IsolationForestModel
from src.models.lstm_model import LSTMModel
from src.models.ensemble import EnsembleModel
import numpy as np
import pytest

def test_isolation_forest():
    model = IsolationForestModel()
    X_train = np.random.rand(100, 10)
    model.fit(X_train)
    predictions = model.predict(X_train)
    assert len(predictions) == 100

def test_lstm_model():
    model = LSTMModel()
    model.build_model()
    X_train = np.random.rand(100, 10, 1)
    y_train = np.random.rand(100, 1)
    model.train(X_train, y_train)
    predictions = model.predict(X_train)
    assert len(predictions) == 100

def test_ensemble_model():
    model1 = IsolationForestModel()
    model2 = LSTMModel()
    ensemble_model = EnsembleModel()
    
    X_train = np.random.rand(100, 10)
    model1.fit(X_train)
    model2.build_model()
    model2.train(X_train.reshape(100, 10, 1), np.random.rand(100, 1))
    
    ensemble_model.fit([model1, model2], X_train)
    predictions = ensemble_model.predict(X_train)
    assert len(predictions) == 100