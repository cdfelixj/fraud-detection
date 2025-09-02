# Fraud Detection System

Enterprise-grade real-time credit card fraud detection platform leveraging ensemble machine learning models, distributed streaming architecture, and comprehensive monitoring capabilities.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [System Design](#system-design)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Development](#development)
- [API Documentation](#api-documentation)
- [Data Models](#data-models)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Security](#security)
- [Testing Strategy](#testing-strategy)
- [Deployment](#deployment)
- [Performance & Scalability](#performance--scalability)


## Overview

The Fraud Detection System is a Flask-based web application that identifies fraudulent credit card transactions using machine learning. The system processes transaction data through an ensemble of ML models and provides both web interface and API access for fraud prediction.

### Key Capabilities

- **ML-Based Detection**: Ensemble approach using Isolation Forest and Logistic Regression
- **Multiple Input Methods**: CSV upload, manual entry, and Kafka streaming
- **Real-time Processing**: Immediate fraud scoring with confidence levels
- **Web Dashboard**: Interactive interface for data upload, training, and monitoring
- **Docker Deployment**: Containerized setup with PostgreSQL, Redis, and Kafka

### Business Impact

- **Fraud Prevention**: Automated detection of fraudulent credit card transactions
- **Risk Assessment**: Confidence scoring for transaction risk levels
- **Data Management**: Comprehensive transaction storage and prediction logging
- **Feedback Integration**: Manual validation and model improvement through feedback

## Architecture

### High-Level Architecture

![System Architecture](SystemArchitecture.drawio.svg)

### System Components

| Component | Purpose | Technology | Current Deployment |
|-----------|---------|------------|-------------------|
| Flask Application | Web interface and API endpoints | Flask 3.1.2+ | Single container (port 5000) |
| ML Engine | Fraud detection inference | scikit-learn ensemble | Integrated with Flask app |
| Kafka Consumer | Real-time stream processing | Apache Kafka | Separate container service |
| Data Store | Transaction persistence | PostgreSQL 13 | Single instance (port 5432) |
| Cache Layer | Session storage | Redis 7 | Single instance (port 6379) |
| Message Broker | Event streaming | Kafka + Zookeeper | Single broker (port 9092) |

### Data Flow Architecture

![Data Flow Diagram](DataDiagram.drawio.svg)

## System Design

### Design Principles

1. **Microservices Architecture**: Loosely coupled services with clear boundaries
2. **Event-Driven Design**: Asynchronous processing with message queues
3. **Immutable Data**: Append-only transaction logs for audit compliance
4. **Circuit Breaker Pattern**: Graceful degradation under load
5. **Observability First**: Comprehensive logging, metrics, and tracing

### Database Design

## Technology Stack

### Core Technologies

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Application** | Python | 3.11+ | Primary runtime environment |
| **Web Framework** | Flask | 3.1.2+ | HTTP API and web interface |
| **ML Framework** | scikit-learn | 1.7.1+ | Machine learning models |
| **Data Processing** | pandas | 2.3.2+ | Data manipulation and analysis |
| **Database** | PostgreSQL | 13+ | Primary data persistence |
| **Cache** | Redis | 7+ | Session storage and caching |
| **Message Broker** | Apache Kafka | 2.8+ | Stream processing |
| **Container Platform** | Docker | 20.10+ | Application containerization |
| **Orchestration** | Docker Compose | 2.0+ | Multi-service orchestration |

### Development Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Package Management** | uv | Fast Python package manager |
| **Code Formatting** | Black | Code style enforcement |
| **Linting** | Flake8 | Static code analysis |
| **Testing** | pytest | Unit and integration testing |
| **Type Checking** | mypy | Static type analysis |
| **Documentation** | Sphinx | API documentation generation |

## Getting Started

#### Required Software
- **Docker Desktop** 4.0+ with Docker Compose v2
- **Git** 2.30+ for version control
- **Python** 3.11+ (for local development)

### Installation

```bash
# Clone the repository
git clone https://github.com/cdfelixj/fraud-detection.git
cd fraud-detection

# Verify system requirements
docker --version
docker-compose --version

# Start the system
docker-compose up --build -d

# Verify deployment
curl http://localhost:5000/health
```

### Initial Configuration

```bash
# Set production environment variables
export SESSION_SECRET="your-secure-session-key"
export DATABASE_PASSWORD="your-secure-db-password"

# Initialize database schema
docker-compose exec fraud-detection python -c "from app import create_app; create_app().app_context()"

# Load sample data (optional)
docker-compose exec fraud-detection python data_simulator.py --load-sample
```

### Quick Validation

```bash
# Health check
curl http://localhost:5000/health

# Test prediction endpoint
curl -X POST http://localhost:5000/api/predict-manual \
  -H "Content-Type: application/json" \
  -d '{"time": 0, "v1": -1.36, "v2": -0.07, "amount": 149.62}'

# Access web interface
open http://localhost:5000
```

## Development

### Development Environment Setup

#### Docker Development (Recommended)

```bash
# Clone the repository
git clone https://github.com/cdfelixj/fraud-detection.git
cd fraud-detection

# Start all services
docker-compose up --build -d

# Verify services are running
docker-compose ps

# Access the application
open http://localhost:5000
```

#### Local Development

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables (optional for local development)
export FLASK_ENV=development

# Start Flask application
python main.py

# Start Kafka consumer separately
python kafka_consumer.py
```

### Environment Configuration

#### Docker Compose Environment

The application is configured through docker-compose.yml with the following default settings:

| Service | Configuration | Port | Purpose |
|---------|---------------|------|---------|
| **fraud-detection** | Flask application | 5000 | Main web interface and API |
| **postgres** | PostgreSQL database | 5432 | Data persistence |
| **redis** | Redis cache | 6379 | Session storage |
| **kafka** | Message broker | 9092 | Event streaming |
| **zookeeper** | Kafka coordination | 2181 | Kafka cluster management |
| **kafka-consumer** | Stream processor | N/A | Real-time event processing |

#### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Runtime environment | `production` |
| `FLASK_DEBUG` | Debug mode | `False` |
| `DATABASE_URL` | PostgreSQL connection | Configured in docker-compose |
| `REDIS_URL` | Redis connection | Configured in docker-compose |

### Code Quality Standards

#### Current Implementation

The project follows basic Python coding standards:

```bash
# Available tools (if installed)
python -m black . --line-length 88
python -m flake8 . --max-line-length 88

# Manual code review and testing
# Docker-based development workflow
# Git version control
```

### Development Workflow

#### Current Workflow

- **main**: Primary development branch
- **Feature development**: Direct commits or feature branches as needed
- **Testing**: Manual testing via web interface and API endpoints
- **Deployment**: Docker Compose for local deployment

## API Documentation

## API Documentation

### Authentication

The system uses session-based authentication for web interface access. API endpoints are accessible without authentication in the current implementation.

```http
# No authentication required for current API endpoints
# Session-based authentication for web interface
```

### Core Endpoints

#### Manual Fraud Prediction

```http
POST /api/predict-manual
Content-Type: application/json

{
  "time": 0,
  "v1": -1.359807,
  "v2": -0.072781,
  "v3": 2.536347,
  "v4": 1.378155,
  "v5": -0.338321,
  "v6": 0.462388,
  "v7": 0.239599,
  "v8": 0.098698,
  "v9": 0.363787,
  "v10": 0.090794,
  "v11": -0.551600,
  "v12": -0.617801,
  "v13": -0.991390,
  "v14": -0.311169,
  "v15": 1.468177,
  "v16": -0.470401,
  "v17": 0.207971,
  "v18": 0.025791,
  "v19": 0.403993,
  "v20": 0.251412,
  "v21": -0.018307,
  "v22": 0.277838,
  "v23": -0.110474,
  "v24": 0.066928,
  "v25": 0.128539,
  "v26": -0.189115,
  "v27": 0.133558,
  "v28": -0.021053,
  "amount": 149.62
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "legitimate",
  "isolation_forest_prediction": 1,
  "logistic_regression_prediction": 0,
  "isolation_forest_confidence": 0.85,
  "logistic_regression_confidence": 0.92,
  "ensemble_confidence": 0.89
}
```

#### Batch Processing

```http
POST /api/batch-predict
Content-Type: multipart/form-data

file: <CSV file with transaction data>
```

#### Feedback Collection

```http
POST /api/feedback
Content-Type: application/json

{
  "prediction_id": 123,
  "feedback": "fraud|legitimate",
  "confidence": 5,
  "notes": "Customer confirmed transaction"
}
```

#### Get Feedback

```http
GET /api/feedback/{feedback_id}
```

#### Transaction Details

```http
GET /api/transaction/{transaction_id}
```

### Monitoring Endpoints

#### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-02T10:30:00Z"
}
```

#### Dashboard Data

```http
GET /api/chart-data
```

#### Simulation Statistics

```http
GET /api/simulation-data
```

### Management Endpoints

#### Clear Data

```http
POST /api/clear-predictions
POST /api/clear-feedback  
POST /api/clear-transactions
```

#### Model Retraining

```http
POST /api/retrain-with-feedback
```

#### Prediction Validation

```http
POST /api/validate-prediction
```

### Error Handling

All API responses follow a consistent error format:

```json
{
  "success": false,
  "error": "Invalid input data",
  "details": "Missing required field: amount"
}
```

## Data Models

### Core Data Models

#### Transaction Model
- **Purpose**: Store credit card transaction data
- **Fields**: 30 features (Time, V1-V28, Amount) + metadata
- **Usage**: Training data and real-time transaction processing

#### Prediction Model  
- **Purpose**: Log fraud detection predictions
- **Fields**: Prediction result, confidence scores, model outputs
- **Usage**: Audit trail and performance monitoring

#### ModelPerformance Model
- **Purpose**: Track ML model accuracy metrics
- **Fields**: Training metrics, evaluation scores, timestamps
- **Usage**: Model performance monitoring and retraining decisions

#### FraudAlert Model
- **Purpose**: Store fraud detection alerts
- **Fields**: Alert details, severity, resolution status
- **Usage**: Alert management and investigation workflow

#### PredictionFeedback Model
- **Purpose**: Collect manual validation feedback
- **Fields**: Feedback type, analyst notes, confidence ratings
- **Usage**: Model improvement and accuracy validation

### Data Relationships

- Each **Transaction** can have multiple **Predictions**
- Each **Prediction** can have associated **PredictionFeedback**
- **FraudAlerts** are triggered by high-risk **Transactions**
- **ModelPerformance** tracks training sessions and accuracy metrics


## Machine Learning Pipeline

### Model Architecture

![ML Pipeline](MLPipeline.drawio.svg)

### Ensemble Strategy

The system employs a two-stage ensemble approach:

1. **Anomaly Detection**: Isolation Forest for unsupervised outlier detection
2. **Classification**: Logistic Regression for supervised fraud classification
3. **Ensemble Fusion**: Weighted combination of model outputs

#### Model Configuration

```python
# Isolation Forest Configuration
isolation_forest_params = {
    'n_estimators': 100,
    'contamination': 0.1,
    'random_state': 42,
    'n_jobs': -1
}

# Logistic Regression Configuration  
logistic_regression_params = {
    'max_iter': 1000,
    'random_state': 42,
    'class_weight': 'balanced',
    'solver': 'liblinear'
}

# Ensemble Weights
ensemble_weights = {
    'isolation_forest': 0.4,
    'logistic_regression': 0.6
}
```

### Feature Engineering

#### Data Preprocessing Pipeline

1. **Data Validation**: Schema validation for required columns (V1-V28, Time, Amount)
2. **Feature Scaling**: StandardScaler normalization for V1-V28 features
3. **Feature Preservation**: Time and Amount features used as-is from dataset
4. **Data Cleaning**: Handle missing values and remove duplicates
5. **Model Input Preparation**: Convert to NumPy array format for ML models

### Model Training Process

#### Training Workflow

1. **Data Upload**: Upload CSV file through web interface (/train endpoint)
2. **Data Loading**: Load and validate CSV using data_processor.py
3. **Feature Preparation**: Apply StandardScaler to V1-V28 features
4. **Model Training**: Train Isolation Forest and Logistic Regression models
5. **Model Evaluation**: Calculate performance metrics on test data
6. **Model Persistence**: Save models as pickle files (isolation_forest.pkl, logistic_model.pkl)
7. **Performance Logging**: Store metrics in ModelPerformance database table
8. **Model Activation**: Set is_trained flag to enable predictions

### Performance Metrics

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **Precision** | True fraud / (True fraud + False fraud) | Calculated during training evaluation |
| **Recall** | True fraud / (True fraud + Missed fraud) | Calculated during training evaluation |
| **F1-Score** | Harmonic mean of precision and recall | Calculated during training evaluation |
| **AUC-ROC** | Area under ROC curve | Calculated during training evaluation |
| **Model Confidence** | Ensemble prediction confidence | Real-time calculation during inference |
| **Processing Time** | Prediction latency | Measured per API call |

## Security

### Data Protection

#### Current Security Implementation
- **Session Management**: Flask session-based authentication for web interface
- **Container Security**: Docker isolation between services
- **Database Access**: PostgreSQL with password authentication
- **Network Security**: Internal Docker network for service communication

#### Recommended Security Enhancements
- **Data at Rest**: Implement database encryption for sensitive transaction data
- **Data in Transit**: Add TLS termination at load balancer level
- **API Security**: Implement API key authentication for programmatic access
- **Input Validation**: Enhanced parameter validation and sanitization



#### Current Monitoring Implementation

| Component | Current Status | Monitoring Method |
|-----------|----------------|-------------------|
| **Application** | Basic health endpoint | /health endpoint |
| **Infrastructure** | Docker container logs | docker logs command |
| **Database** | Basic connection monitoring | Flask SQLAlchemy connection |
| **ML Models** | Performance logging to database | ModelPerformance table |
| **APIs** | Request/response logging | Console output |

#### Recommended Monitoring Enhancements

**Business Metrics**
- Fraud detection accuracy tracking
- Prediction confidence distribution
- Transaction processing volume
- Model retraining frequency

**Technical Metrics**
- API response times and error rates
- Database connection pool utilization  
- Memory usage for ML model inference
- Kafka consumer lag monitoring

**Operational Metrics**
- Container resource utilization
- Disk space usage for model storage
- Network connectivity between services
- Log volume and error patterns

### Alerting Strategy

#### Current Alerting
- Basic application exceptions logged to console
- Docker container health status monitoring
- Manual monitoring of system performance

#### Recommended Alerting
- Model prediction accuracy degradation
- High API error rates or timeouts
- Database connectivity issues
- Kafka message processing delays

### Logging Architecture

#### Current Logging Implementation

```python
# Current basic logging approach
print(f"Training completed with accuracy: {accuracy}")
print(f"Prediction made: {prediction_result}")
print(f"Error occurred: {error_message}")
```

#### Recommended Structured Logging

```python
# Proposed structured logging configuration
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('fraud_detection')
logger.info('Model training started')
logger.error('Prediction failed', extra={'transaction_id': txn_id})
```

## Testing Strategy

### Test Architecture

#### Current Testing Status

| Test Level | Status | Implementation |
|------------|--------|----------------|
| **Unit Tests** | ❌ Not Implemented | pytest available in requirements.txt |
| **Integration Tests** | ❌ Not Implemented | Could test API endpoints and database |
| **Contract Tests** | ❌ Not Implemented | API contract validation needed |
| **End-to-End Tests** | ❌ Not Implemented | Web interface workflow testing |
| **Performance Tests** | ❌ Not Implemented | Load testing for prediction endpoints |
| **Security Tests** | ❌ Not Implemented | Input validation and vulnerability testing |

#### Current Testing Approach

```bash
# Manual testing methods currently used
# 1. Web interface testing
curl http://localhost:5000/health

# 2. API endpoint validation
curl -X POST http://localhost:5000/api/predict-manual \
  -H "Content-Type: application/json" \
  -d '{"time": 0, "v1": -1.36, "amount": 149.62}'

# 3. Docker deployment testing
docker-compose up --build
docker-compose exec fraud-detection python -c "from app import create_app; print('App works')"
```

#### Recommended Testing Implementation

```bash
# Unit tests for ML models
pytest tests/unit/test_ml_models.py
pytest tests/unit/test_data_processor.py

# Integration tests for APIs
pytest tests/integration/test_api_endpoints.py
pytest tests/integration/test_database.py

# End-to-end tests for workflows
pytest tests/e2e/test_prediction_workflow.py
pytest tests/e2e/test_training_workflow.py
```

### Quality Gates

#### Continuous Integration Requirements

```yaml
# Quality gates for CI/CD pipeline
quality_gates:
  code_coverage: ">= 90%"
  test_success_rate: "100%"
  performance_regression: "< 5%"
  security_vulnerabilities: "0 high/critical"
  code_complexity: "< 10 (cyclomatic)"
```

## Deployment

### Current Deployment Strategy

#### Single Environment Deployment

| Environment | Purpose | Configuration | Access |
|-------------|---------|---------------|--------|
| **Local Development** | Development and testing | Docker Compose | http://localhost:5000 |

### Container Strategy

#### Current Docker Implementation

```dockerfile
# Actual Dockerfile approach
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "main.py"]
```

#### Container Services

```yaml
# docker-compose.yml services
services:
  fraud-detection:    # Flask application (port 5000)
  kafka-consumer:     # Stream processing service
  kafka:              # Message broker (port 9092)
  zookeeper:          # Kafka coordination (port 2181)  
  postgres:           # Database (port 5432)
  redis:              # Cache/sessions (port 6379)
```

### Database Management

```bash
# Current database operations
docker-compose exec postgres psql -U postgres -d fraud_detection

# Database initialization from init.sql
# Tables created: transactions, predictions, model_performance, fraud_alerts, prediction_feedback

# View data
docker-compose exec postgres psql -U postgres -d fraud_detection -c "SELECT * FROM transactions LIMIT 5;"
```

## Performance & Scalability

### Current Performance Characteristics

#### Throughput (Estimated)

| Component | Current Setup | Estimated Capacity |
|-----------|---------------|-------------------|
| **Flask Application** | Single container | ~100-500 requests/second |
| **ML Pipeline** | In-memory models | ~50-200 predictions/second |
| **Database** | Single PostgreSQL instance | ~1,000 transactions/second |
| **Cache Layer** | Single Redis instance | ~10,000 operations/second |

#### Actual Response Times

| Operation | Description | Typical Response |
|-----------|-------------|------------------|
| **Health Check** | GET /health | ~5ms |
| **Manual Prediction** | POST /api/predict-manual | ~50-200ms |
| **Batch Upload** | POST /api/batch-predict | Depends on file size |
| **Model Training** | File upload to training complete | ~30-300 seconds |

### Scaling Strategies

#### Horizontal Scaling Options

**Application Tier Scaling**
- Multiple Flask application containers behind load balancer
- Stateless application design (sessions in Redis)
- Container auto-scaling based on CPU/memory

**Data Tier Scaling**
- PostgreSQL read replicas for query distribution
- Database connection pooling
- Query optimization with proper indexing

**Cache Tier Scaling**
- Redis Cluster configuration for high availability
- Distributed caching strategies
- Session replication across nodes

#### Performance Optimization

```bash
# Database performance monitoring
docker-compose exec postgres psql -U postgres -d fraud_detection -c "SELECT * FROM pg_stat_activity;"

# Redis performance monitoring  
docker-compose exec redis redis-cli info stats

# Application resource monitoring
docker stats fraud-detection
```

