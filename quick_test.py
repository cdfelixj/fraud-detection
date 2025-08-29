#!/usr/bin/env python3
"""
Quick system test to verify everything is working
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("🔍 Testing imports...")
    
    # Test model imports
    from src.models.ensemble import EnsembleModel
    from src.models.isolation_forest import IsolationForestModel
    from src.models.lstm_model import LSTMModel
    print("✅ ML models imported successfully")
    
    # Test API import
    from src.api.app import app
    print("✅ Flask app imported successfully")
    
    # Test database imports
    from src.database.postgres_handler import PostgresHandler
    from src.database.redis_cache import RedisCache
    print("✅ Database modules imported successfully")
    
    # Test basic functionality
    print("\n🧠 Testing ML models...")
    
    isolation_model = IsolationForestModel()
    lstm_model = LSTMModel()
    ensemble_model = EnsembleModel()
    
    ensemble_model.add_model(isolation_model, weight=0.6)
    ensemble_model.add_model(lstm_model, weight=0.4)
    
    import numpy as np
    test_features = np.array([100.0, 12, 1, 1, 0])
    
    # Test prediction
    result = ensemble_model.predict_fraud_probability(test_features)
    print(f"✅ Ensemble prediction: {result}")
    
    print("\n🎉 All tests passed! System is working correctly.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
