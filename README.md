# Fraud Detection System

A real-time credit card fraud detection system using an ensemble of machine learning models, built with Flask, PostgreSQL, Redis, Apache Kafka, and Docker. It supports live analysis, streaming data processing, model training, feedback, and monitoring.

## Tech Stack

- Backend: Python 3.11+, Flask, SQLAlchemy
- Machine Learning: scikit-learn, pandas, numpy
- Streaming: Apache Kafka, Zookeeper
- Data Store: PostgreSQL 13
- Cache: Redis 7
- Web UI: Jinja2 templates, HTML/CSS/JavaScript
- Deployment: Docker, Docker Compose, Gunicorn

## Features

### Core Functionality
- **Real-time Fraud Detection**: Ensemble ML approach using Isolation Forest and Logistic Regression
- **Streaming**: High-throughput real-time transaction processing with Apache Kafka
- **Interactive Web Dashboard**: Monitor transactions, predictions, and system performance
- **Streaming Data Simulation**: Realistic transaction stream generation with configurable patterns
- **Manual Transaction Analysis**: Input custom transactions for immediate fraud assessment
- **Batch Processing**: Upload and process CSV transaction data via Kafka streaming
- **Model Training & Evaluation**: Train models on custom datasets with performance metrics

### Advanced Capabilities
- **Prediction Logging**: Track all predictions with detailed scoring and confidence metrics
- **Feedback Mechanism**: User feedback system to improve model accuracy over time
- **Model Validation**: Comprehensive validation dashboard with accuracy metrics
- **Performance Monitoring**: Real-time system health and prediction statistics
- **Data Persistence**: PostgreSQL database with Redis caching for optimal performance

### Security & Deployment
- **Containerized Deployment**: Full Docker Compose setup with health checks
- **Database Security**: Secure PostgreSQL configuration with persistent volumes
- **Production Ready**: Gunicorn WSGI server with proper logging and error handling

## Prerequisites

- **Docker Desktop** (with Docker Compose)
- **Git** (for cloning the repository)
- **4GB+ RAM** (recommended for optimal performance)
- **Port availability**: 5000 (Flask), 5432 (PostgreSQL), 6379 (Redis)

## Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd FraudDetect
```

### 2. Start the System

**Option A: Standard Mode**
```bash
docker-compose up --build
```

**Option B: Kafka Streaming Mode (Recommended)**
```bash
start-kafka.bat
```

### 3. Access the Application
- **Main Dashboard**: http://localhost:5000
- **Streaming Dashboard**: http://localhost:5000/kafka/dashboard
- **Health Check**: http://localhost:5000/health
- Wait for all health checks to pass before using the system

### 4. Load Sample Data (Optional)
Navigate to the Upload page and upload the sample dataset from `attached_assets/creditcardcsv` to get started with pre-existing data.

## Kafka Streaming Features

### High-Throughput Processing
- **Real-time transaction streaming**: Process 100+ transactions per second
- **Configurable simulation**: Adjust fraud rates and transaction patterns
- **Burst mode support**: Handle traffic spikes automatically
- **Real-time fraud alerts**: Immediate notifications for suspicious activity

### Enhanced Data Input Methods
1. **Kafka Streaming API**: REST endpoints for real-time transaction submission
2. **Batch Upload via Kafka**: High-throughput CSV processing
3. **Data Simulation**: Realistic transaction stream generation
4. **Manual Testing**: Individual transaction fraud assessment

For detailed Kafka documentation, see [KAFKA_INTEGRATION.md](KAFKA_INTEGRATION.md).

## Development

### Environment Variables

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://fraud_user:fraud_password@postgres:5432/fraud_detection` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379` |
| `SESSION_SECRET` | Flask session encryption key | `your-secret-key-change-in-production` |
| `FLASK_ENV` | Environment mode | `production` |
| `PORT` | Application port | `5000` |

### Development Mode with Hot Reload

For active development with file watching:

```bash
# Use the development compose file
docker-compose -f docker-compose.dev.yml up --build
```

### Local Development (Non-Docker)

```bash
# Install dependencies with uv
pip install uv
uv sync

# Set up environment variables
set DATABASE_URL=sqlite:///fraud_detection.db
set FLASK_ENV=development

# Run the application
uv run python main.py
```

### Code Quality Tools

The project includes development dependencies for code quality:

```bash
# Format code with Black
uv run black .

# Lint with Flake8
uv run flake8 .

# Run tests
uv run pytest
```

## System Architecture

### Services Overview

| Service | Technology | Port | Purpose |
|---------|------------|------|---------|
| **fraud-detection** | Flask + Gunicorn | 5000 | Main web application and API |
| **postgres** | PostgreSQL 13 | 5432 | Primary data storage |
| **redis** | Redis 7 Alpine | 6379 | Caching and session storage |

### Machine Learning Pipeline

1. **Data Preprocessing**: Feature scaling and normalization
2. **Ensemble Model**: Combination of Isolation Forest (anomaly detection) and Logistic Regression (classification)
3. **Prediction Scoring**: Weighted ensemble with confidence metrics
4. **Feedback Integration**: Continuous learning from user feedback

### Database Schema

#### Core Tables
- **`transactions`**: Transaction data with 28 features (Time, V1-V28, Amount, Class)
- **`predictions`**: ML model predictions with scores and confidence
- **`prediction_feedback`**: User feedback on predictions for model improvement

## User Interface

### Main Dashboard (`/`)
- Real-time fraud detection statistics
- Recent transaction overview
- Quick access to all system features
- System health indicators

### Transaction Analysis (`/predict`)
- Manual transaction input form
- Real-time fraud risk assessment
- Detailed prediction breakdown with confidence scores

### Data Management (`/upload`)
- CSV file upload for batch processing
- Data validation and preview
- Automatic feature extraction and processing

### Model Training (`/train`)
- Interactive model training interface
- Performance metrics and validation
- Model comparison and selection tools

### Monitoring Dashboards
- **`/transactions`**: Complete transaction history with filtering
- **`/validation`**: Model accuracy metrics and confusion matrices
- **`/feedback`**: Feedback management and model improvement tools
- **`/monitoring`**: System performance and health metrics

## API Reference

### Prediction APIs
```http
POST /api/predict-manual
Content-Type: application/json

{
  "time": 12345.67,
  "v1": -1.234,
  "v2": 0.567,
  // ... v3-v28
  "amount": 149.62
}
```

### Data Management APIs
```http
POST /api/upload          # Upload CSV data
GET  /api/transactions    # Get transaction history
POST /api/batch-predict   # Generate batch predictions
```

### Feedback APIs
```http
POST /api/feedback                    # Submit prediction feedback
GET  /api/feedback/<prediction_id>    # Get feedback for prediction
GET  /api/transaction/<id>/details    # Get transaction details
```

### Training APIs
```http
POST /api/train           # Train models on uploaded data
GET  /api/model-metrics   # Get current model performance
```

### System APIs
```http
GET /health              # System health check
GET /api/stats           # System statistics
```

## Project Structure

```
├── 📁 Application Core
│   ├── app.py              # Flask application factory and configuration
│   ├── main.py             # Application entry point and server startup
│   ├── routes.py           # Web routes and API endpoint definitions
│   ├── models.py           # SQLAlchemy database models
│   ├── ml_models.py        # Machine learning model implementations
│   └── data_processor.py   # Data preprocessing and feature engineering
│
├── 📁 Frontend Assets
│   ├── templates/          # Jinja2 HTML templates
│   │   ├── base.html       # Base template with navigation
│   │   ├── dashboard.html  # Main dashboard interface
│   │   ├── predict_manual.html # Manual prediction form
│   │   ├── upload.html     # Data upload interface
│   │   ├── train_enhanced.html # Model training interface
│   │   ├── transactions.html   # Transaction history viewer
│   │   ├── validation.html     # Model validation dashboard
│   │   ├── feedback.html       # Feedback management interface
│   │   ├── monitoring.html     # System monitoring dashboard
│   │   └── admin.html          # Administrative interface
│   └── static/
│       ├── css/custom.css      # Application styling
│       └── js/dashboard.js     # Frontend JavaScript functionality
│
├── 📁 Data & Models
│   ├── saved_models/       # Trained ML model files (.pkl)
│   ├── attached_assets/    # Sample datasets and uploads
│   └── scaler.pkl         # Feature scaling parameters
│
├── 📁 Infrastructure
│   ├── Dockerfile         # Application container definition
│   ├── docker-compose.yml # Production multi-service setup
│   ├── docker-compose.dev.yml # Development configuration
│   ├── init.sql          # Database initialization script
│   ├── start.bat         # Windows startup script
│   ├── start-dev.bat     # Windows development startup
│   └── stop.bat          # Windows shutdown script
│
├── 📁 Configuration
│   ├── requirements.txt   # Python dependencies
│   ├── pyproject.toml    # Modern Python project configuration
│   └── uv.lock          # Dependency lock file
│
└── 📁 Documentation
    ├── README.md                    # This file
    ├── FEEDBACK_MECHANISM_SUMMARY.md # Feedback system documentation
    └── PREDICTION_LOGGING_SUMMARY.md # Prediction tracking documentation
```

## Usage

### Getting Started
1. **Start the system** using Docker Compose
2. **Upload training data** via the Upload page
3. **Train models** using the Training interface
4. **Analyze transactions** through manual prediction or batch processing
5. **Monitor performance** via the Validation and Monitoring dashboards

### Training Your Model
1. Navigate to `/upload` and upload a CSV file with transaction data
2. Go to `/train` and click "Train Enhanced Models"
3. Monitor training progress and view performance metrics
4. Models are automatically saved and deployed when training completes

### Making Predictions
1. **Manual Analysis**: Use `/predict` for single transaction analysis
2. **Batch Processing**: Upload CSV files for bulk transaction analysis
3. **API Integration**: Use REST endpoints for programmatic access

### Providing Feedback
1. View predictions in the transaction history
2. Mark predictions as correct/incorrect via the feedback interface
3. Add detailed reasoning to help improve the model
4. Monitor feedback statistics in the Feedback dashboard

### Monitoring System Health
1. **Validation Dashboard**: View model accuracy and performance metrics
2. **Monitoring Dashboard**: Check system health and prediction statistics
3. **Feedback Analytics**: Analyze user feedback patterns and model improvement

## Configuration

### Model Configuration
Models can be configured in `ml_models.py`:
- **Isolation Forest**: Anomaly detection parameters
- **Logistic Regression**: Classification hyperparameters
- **Ensemble Weights**: Adjust model combination strategy

### Database Configuration
Database settings in `docker-compose.yml`:
- Connection pooling
- Memory allocation
- Backup and recovery settings

### Performance Tuning
- **Redis Configuration**: Adjust cache sizes and TTL
- **Application Scaling**: Configure Gunicorn workers
- **Database Optimization**: Tune PostgreSQL parameters

## Security

### Production Deployment
- **Change Default Secrets**: Update `SESSION_SECRET` and database passwords
- **Enable HTTPS**: Configure SSL/TLS certificates
- **Network Security**: Use Docker networks and firewall rules
- **Data Encryption**: Enable PostgreSQL encryption at rest

### Data Privacy
- Transaction data is stored securely in PostgreSQL
- User feedback is anonymized by default
- Sensitive data processing follows privacy best practices

## Monitoring & Observability

### Health Checks
- Application health endpoint: `/health`
- Database connectivity monitoring
- Redis cache status verification
- Model availability checks

### Logging
- Structured application logging
- Database query logging
- Error tracking and alerting
- Performance metrics collection

### Metrics & Analytics
- **Prediction Accuracy**: Real-time model performance tracking
- **System Performance**: Response times and throughput
- **User Engagement**: Feedback submission rates and patterns
- **Data Quality**: Transaction data validation and monitoring

## Testing

### Running Tests
```bash
# Install development dependencies
uv sync --dev

# Run all tests
uv run pytest

# Run specific test files
uv run pytest tests/test_models.py

# Run with coverage
uv run pytest --cov=. --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **Model Tests**: ML model validation
- **Database Tests**: Data persistence verification

## Deployment

### Production Deployment
```bash
# Production startup
docker-compose up -d --build

# Check service status
docker-compose ps

# View logs
docker-compose logs -f fraud-detection
```

### Scaling Considerations
- **Horizontal Scaling**: Multiple Flask application instances
- **Database Scaling**: PostgreSQL read replicas
- **Caching Strategy**: Redis cluster configuration
- **Load Balancing**: Nginx or cloud load balancer integration

### Environment-Specific Configurations
- **Development**: Use `docker-compose.dev.yml` for hot reloading
- **Staging**: Mirror production with test data
- **Production**: Full security hardening and monitoring

## Troubleshooting

### Common Issues

#### Application Won't Start
```bash
# Check service health
docker-compose ps

# View detailed logs
docker-compose logs fraud-detection

# Restart specific service
docker-compose restart fraud-detection
```

#### Database Connection Issues
```bash
# Check PostgreSQL health
docker-compose logs postgres

# Test database connection
docker-compose exec postgres psql -U fraud_user -d fraud_detection

# Reset database (WARNING: deletes all data)
docker-compose down -v && docker-compose up --build
```

#### Model Training Failures
- Ensure uploaded data has required columns (Time, V1-V28, Amount, Class)
- Check data format and handle missing values
- Verify sufficient training data (minimum 1000 transactions recommended)

#### Performance Issues
- Monitor Redis cache hit rates
- Check database query performance
- Scale services based on load requirements

### Debug Mode
Enable detailed logging by setting:
```bash
FLASK_ENV=development
```

## API Examples

### Request/Response Examples

#### Manual Prediction
```bash
curl -X POST http://localhost:5000/api/predict-manual \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

#### Feedback Submission
```bash
curl -X POST http://localhost:5000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": 123,
    "feedback": "incorrect",
    "actual_outcome": 1,
    "reason": "Transaction was verified as fraudulent",
    "confidence": 5,
    "user_id": "analyst_1"
  }'
```

## Architecture Details

### Technology Stack
- **Backend**: Python 3.11+, Flask, SQLAlchemy
- **Database**: PostgreSQL 13 with Redis caching
- **ML Libraries**: scikit-learn, pandas, numpy
- **Frontend**: HTML5, Bootstrap, JavaScript
- **Deployment**: Docker, Docker Compose, Gunicorn

### Data Flow
1. **Input**: Transaction data via web form or CSV upload
2. **Processing**: Feature scaling and preprocessing
3. **Prediction**: Ensemble model scoring
4. **Storage**: Results saved to PostgreSQL
5. **Feedback Loop**: User validation improves future predictions

### Security Architecture
- Session-based authentication
- SQL injection prevention via SQLAlchemy ORM
- Input validation and sanitization
- Docker container isolation
- Environment variable configuration

## Dependencies

### Core Dependencies
- **Flask** (3.1.2+): Web framework
- **SQLAlchemy** (2.0.43+): Database ORM
- **scikit-learn** (1.7.1+): Machine learning
- **pandas** (2.3.2+): Data manipulation
- **psycopg2-binary** (2.9.10+): PostgreSQL adapter
- **redis** (5.0.0+): Caching and session storage

### Development Dependencies
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting
- **pytest-flask**: Flask testing utilities

## 🚀 Production Deployment

### Environment Setup
1. **Security**: Change all default passwords and secrets
2. **Networking**: Configure firewall and reverse proxy
3. **SSL/TLS**: Set up HTTPS certificates
4. **Backup**: Configure database backup strategy

### Performance Optimization
- **Database**: Tune PostgreSQL for your workload
- **Caching**: Optimize Redis configuration
- **Application**: Scale Gunicorn workers based on CPU cores
- **Monitoring**: Set up application performance monitoring

### Maintenance
- **Updates**: Regular security updates for base images
- **Backups**: Automated database backups
- **Monitoring**: Log aggregation and alerting
- **Health Checks**: Automated system health monitoring

## Support & Contributing

### Getting Help
- Check the troubleshooting section above
- Review system logs for error details
- Verify all prerequisites are installed correctly

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use Black for code formatting
- Include docstrings for functions and classes
- Write comprehensive tests for new features

## License

This project is available under the MIT License. See the LICENSE file for more details.

## Version History

- **v1.0.0**: Initial release with core fraud detection
- **v1.1.0**: Added prediction logging and validation
- **v1.2.0**: Implemented feedback mechanism and model improvement
- **v1.3.0**: Enhanced monitoring and administrative features

---


