"""
Test script for prediction logging functionality
Run this after starting the application to test the new prediction logging features
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:5000"

def test_manual_prediction():
    """Test manual prediction with logging"""
    print("Testing manual prediction with logging...")
    
    # Sample transaction data
    test_data = {
        "time_feature": 1000.5,
        "amount": 150.75,
        "v1": 0.144, "v2": -0.234, "v3": 1.345, "v4": -0.876,
        "v5": 0.234, "v6": -1.234, "v7": 0.567, "v8": -0.432,
        "v9": 0.123, "v10": -0.567, "v11": 0.789, "v12": -0.345,
        "v13": 0.456, "v14": -0.123, "v15": 0.234, "v16": -0.567,
        "v17": 0.345, "v18": -0.234, "v19": 0.123, "v20": -0.456,
        "v21": 0.567, "v22": -0.123, "v23": 0.345, "v24": -0.678,
        "v25": 0.234, "v26": -0.345, "v27": 0.123, "v28": -0.234
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/predict-manual", 
                               json=test_data,
                               headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Manual prediction successful!")
            print(f"  - Transaction ID: {result.get('transaction_id', 'N/A')}")
            print(f"  - Prediction ID: {result.get('prediction_id', 'N/A')}")
            print(f"  - Prediction: {'Fraud' if result.get('prediction') == 1 else 'Normal'}")
            print(f"  - Confidence: {result.get('confidence', 0):.3f}")
            return result.get('transaction_id')
        else:
            print(f"✗ Manual prediction failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Error testing manual prediction: {e}")
        return None

def test_batch_predictions():
    """Test batch prediction generation"""
    print("\nTesting batch predictions...")
    
    test_data = {
        "limit": 10,
        "skip_existing": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/batch-predict",
                               json=test_data,
                               headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Batch predictions successful!")
            print(f"  - Message: {result.get('message', 'No message')}")
            print(f"  - Predictions saved: {result.get('predictions_saved', 0)}")
        else:
            print(f"✗ Batch predictions failed: {response.status_code}")
            print(f"  Error: {response.text}")
            
    except Exception as e:
        print(f"✗ Error testing batch predictions: {e}")

def test_validation_page():
    """Test validation page access"""
    print("\nTesting validation page...")
    
    try:
        response = requests.get(f"{BASE_URL}/validation")
        
        if response.status_code == 200:
            print("✓ Validation page accessible!")
            print(f"  - Page size: {len(response.text)} characters")
        else:
            print(f"✗ Validation page failed: {response.status_code}")
            
    except Exception as e:
        print(f"✗ Error accessing validation page: {e}")

def test_ground_truth_update(transaction_id):
    """Test updating ground truth for validation"""
    if not transaction_id:
        print("\nSkipping ground truth test - no transaction ID")
        return
        
    print(f"\nTesting ground truth update for transaction {transaction_id}...")
    
    test_data = {
        "transaction_id": transaction_id,
        "actual_class": 0  # Mark as normal transaction
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/validate-prediction",
                               json=test_data,
                               headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Ground truth update successful!")
            print(f"  - Message: {result.get('message', 'No message')}")
            print(f"  - Prediction correct: {result.get('prediction_correct', 'N/A')}")
        else:
            print(f"✗ Ground truth update failed: {response.status_code}")
            print(f"  Error: {response.text}")
            
    except Exception as e:
        print(f"✗ Error testing ground truth update: {e}")

def main():
    print("=== Testing Prediction Logging Implementation ===\n")
    
    # Test manual prediction (creates transaction + saves prediction)
    transaction_id = test_manual_prediction()
    
    # Test batch predictions (on existing transactions)
    test_batch_predictions()
    
    # Test validation page
    test_validation_page()
    
    # Test ground truth update
    test_ground_truth_update(transaction_id)
    
    print("\n=== Test Complete ===")
    print("\nTo verify the implementation:")
    print("1. Check the application UI at http://localhost:5000")
    print("2. Navigate to 'Validation' in the menu")
    print("3. Try generating batch predictions")
    print("4. View the validation statistics")

if __name__ == "__main__":
    main()
