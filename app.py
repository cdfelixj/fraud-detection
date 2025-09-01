import os
import logging
from datetime import datetime
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging for Docker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database - prioritize Docker environment
database_url = os.environ.get("DATABASE_URL")
if not database_url:
    # Fallback to SQLite for local development
    database_url = "sqlite:///fraud_detection.db"
    
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the app with the extension
db.init_app(app)

with app.app_context():
    # Import models to ensure tables are created
    import models
    try:
        db.create_all()
        logging.info("Database tables created successfully")
        
        # Initialize Kafka topics
        try:
            from kafka_config import create_topics_if_not_exist
            create_topics_if_not_exist()
            logging.info("Kafka topics initialized")
        except Exception as kafka_error:
            logging.warning(f"Kafka initialization failed (will retry): {kafka_error}")
            
    except Exception as e:
        logging.error(f"Error creating database tables: {e}")

# Add health check endpoint for Docker
@app.route('/health')
def health_check():
    """Health check endpoint for Docker containers"""
    try:
        # Test database connection
        db.session.execute(text('SELECT 1'))
        db_healthy = True
    except Exception as e:
        logging.error(f"Database health check failed: {e}")
        db_healthy = False
    
    # Test Kafka connection
    kafka_healthy = False
    try:
        from kafka_config import kafka_manager
        kafka_healthy = kafka_manager.health_check()
    except Exception as e:
        logging.warning(f"Kafka health check failed: {e}")
    
    status = "healthy" if db_healthy else "unhealthy"
    response_data = {
        "status": status,
        "database": "connected" if db_healthy else "disconnected",
        "kafka": "connected" if kafka_healthy else "disconnected",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    status_code = 200 if db_healthy else 500
    return jsonify(response_data), status_code

# Import routes
from routes import *

