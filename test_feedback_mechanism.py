"""
Test script for feedback mechanism functionality
Run this after starting the application to test the new feedback features
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"

def test_manual_prediction_with_feedback():
    """Test manual prediction and then provide feedback"""
    print("Testing manual prediction with feedback...")
    
    # Sample transaction data
    test_data = {
        "time_feature": 2000.0,
        "amount": 250.75,
        "v1": 0.244, "v2": -0.434, "v3": 1.545, "v4": -0.976,
        "v5": 0.334, "v6": -1.334, "v7": 0.667, "v8": -0.532,
        "v9": 0.223, "v10": -0.667, "v11": 0.889, "v12": -0.445,
        "v13": 0.556, "v14": -0.223, "v15": 0.334, "v16": -0.667,
        "v17": 0.445, "v18": -0.334, "v19": 0.223, "v20": -0.556,
        "v21": 0.667, "v22": -0.223, "v23": 0.445, "v24": -0.778,
        "v25": 0.334, "v26": -0.445, "v27": 0.223, "v28": -0.334
    }
    
    try:
        # Make prediction
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
            
            # Now provide feedback
            prediction_id = result.get('prediction_id')
            if prediction_id:
                return test_submit_feedback(prediction_id, result.get('prediction', 0))
            
        else:
            print(f"✗ Manual prediction failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Error testing manual prediction: {e}")
        return None

def test_submit_feedback(prediction_id, predicted_class):
    """Test submitting feedback on a prediction"""
    print(f"\nTesting feedback submission for prediction {prediction_id}...")
    
    # Test feedback data
    feedback_data = {
        "prediction_id": prediction_id,
        "feedback": "correct" if predicted_class == 0 else "incorrect",  # Assume normal is correct, fraud is incorrect for testing
        "reason": "Testing feedback mechanism - this is a test evaluation",
        "confidence_rating": 4,
        "user_id": "test_user",
        "actual_outcome": 0 if predicted_class == 1 else None  # If predicted fraud, say it was actually normal
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/feedback",
                               json=feedback_data,
                               headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Feedback submission successful!")
            print(f"  - Message: {result.get('message', 'No message')}")
            print(f"  - Feedback: {feedback_data['feedback']}")
            return True
        else:
            print(f"✗ Feedback submission failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing feedback submission: {e}")
        return False

def test_get_feedback(prediction_id):
    """Test retrieving feedback for a prediction"""
    print(f"\nTesting feedback retrieval for prediction {prediction_id}...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/feedback/{prediction_id}")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Feedback retrieval successful!")
            print(f"  - Total feedback: {result.get('total_feedback', 0)}")
            if result.get('feedback'):
                for fb in result['feedback']:
                    print(f"  - {fb['user_feedback']} by {fb['created_by']} - {fb['feedback_reason']}")
            return True
        else:
            print(f"✗ Feedback retrieval failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing feedback retrieval: {e}")
        return False

def test_feedback_dashboard():
    """Test feedback dashboard access"""
    print("\nTesting feedback dashboard...")
    
    try:
        response = requests.get(f"{BASE_URL}/feedback")
        
        if response.status_code == 200:
            print("✓ Feedback dashboard accessible!")
            print(f"  - Page size: {len(response.text)} characters")
            return True
        else:
            print(f"✗ Feedback dashboard failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error accessing feedback dashboard: {e}")
        return False

def test_retrain_with_feedback():
    """Test model retraining with feedback"""
    print("\nTesting model retraining with feedback...")
    
    test_data = {
        "use_feedback": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/retrain-with-feedback",
                               json=test_data,
                               headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Model retraining successful!")
            print(f"  - Message: {result.get('message', 'No message')}")
            print(f"  - Feedback used: {result.get('feedback_used', 0)}")
            return True
        else:
            result = response.json()
            if "Not enough feedback data" in result.get('error', ''):
                print("ℹ Model retraining skipped - not enough feedback data (expected)")
                return True
            else:
                print(f"✗ Model retraining failed: {response.status_code}")
                print(f"  Error: {response.text}")
                return False
            
    except Exception as e:
        print(f"✗ Error testing model retraining: {e}")
        return False

def test_batch_predictions_and_feedback():
    """Test generating batch predictions and providing feedback"""
    print("\nTesting batch predictions for feedback testing...")
    
    # First generate some batch predictions
    batch_data = {
        "limit": 5,
        "skip_existing": False
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/batch-predict",
                               json=batch_data,
                               headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Batch predictions generated!")
            print(f"  - Message: {result.get('message', 'No message')}")
            
            # Now test providing feedback on these predictions
            # Note: In a real scenario, you'd get actual prediction IDs from the database
            print("  - Batch predictions ready for feedback testing")
            return True
        else:
            print(f"ℹ Batch predictions: {response.status_code} - {response.text}")
            return True  # Not a failure if no data available
            
    except Exception as e:
        print(f"✗ Error testing batch predictions: {e}")
        return False

def main():
    print("=== Testing Feedback Mechanism Implementation ===\n")
    
    # Test manual prediction with feedback
    test_manual_prediction_with_feedback()
    
    time.sleep(1)  # Small delay between tests
    
    # Test batch predictions
    test_batch_predictions_and_feedback()
    
    time.sleep(1)
    
    # Test feedback dashboard
    test_feedback_dashboard()
    
    time.sleep(1)
    
    # Test model retraining
    test_retrain_with_feedback()
    
    print("\n=== Test Complete ===")
    print("\nTo verify the feedback mechanism:")
    print("1. Check the application UI at http://localhost:5000")
    print("2. Navigate to 'Feedback' in the menu")
    print("3. Try the 'Validation' page to provide feedback on predictions")
    print("4. Make manual predictions and provide feedback")
    print("5. Use 'Retrain with Feedback' when enough feedback is collected")
    print("\nThe feedback mechanism enables:")
    print("- User validation of predictions (correct/incorrect/uncertain)")
    print("- Collection of ground truth data")
    print("- Model improvement through feedback-weighted retraining")
    print("- Tracking of prediction accuracy over time")
    print("- Analysis of problematic prediction patterns")

if __name__ == "__main__":
    main()
