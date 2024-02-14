from src.api.app import app
import json
import unittest

class ApiTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data), {'status': 'healthy'})

    def test_fraud_detection(self):
        test_data = {
            "transaction_id": "12345",
            "amount": 1000,
            "location": "New York",
            "timestamp": "2023-10-01T12:00:00Z"
        }
        response = self.app.post('/detect-fraud', json=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('is_fraud', json.loads(response.data))

if __name__ == '__main__':
    unittest.main()