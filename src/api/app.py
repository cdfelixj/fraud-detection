import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from src.api.routes import setup_routes

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Enable CORS for React frontend
CORS(app, origins=["http://localhost:3000"])

# Load configuration settings
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
app.config['TESTING'] = os.getenv('FLASK_TESTING', 'False').lower() == 'true'

# Set up API routes
setup_routes(app)

# Health check endpoint for the root
@app.route('/')
def root():
    return {
        'message': 'Fraud Detection System API',
        'version': '1.0.0',
        'status': 'running'
    }

@app.errorhandler(404)
def not_found(error):
    return {'error': 'Endpoint not found'}, 404

@app.errorhandler(500)
def internal_error(error):
    return {'error': 'Internal server error'}, 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)