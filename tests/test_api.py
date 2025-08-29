from src.api.app import app
import json
import unittest

class ApiTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check(self):
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')

    def test_fraud_detection(self):
        test_data = {
            "features": [1000, 12, 1, 1, 0]  # amount, hour, weekday, tx_type, merchant_category
        }
        response = self.app.post('/api/detect-fraud', 
                                json=test_data,
                                content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('is_fraud', data)
        self.assertIn('fraud_probability', data)
        self.assertIn('confidence', data)
        self.assertIsInstance(data['is_fraud'], bool)
        self.assertIsInstance(data['fraud_probability'], (int, float))

    def test_fraud_detection_missing_features(self):
        test_data = {}
        response = self.app.post('/api/detect-fraud', 
                                json=test_data,
                                content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_predict_endpoint(self):
        test_data = {
            "features": [1000, 12, 1, 1, 0]
        }
        response = self.app.post('/api/predict', 
                                json=test_data,
                                content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('ensemble', data)

if __name__ == '__main__':
    unittest.main()