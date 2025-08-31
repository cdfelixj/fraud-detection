# Fraud Detection System

A real-time credit card fraud detection system using machine learning, built with Flask and Docker.

## Features

- Real-time fraud detection using Isolation Forest and Logistic Regression
- Web dashboard for monitoring transactions
- Manual transaction prediction
- Model training and evaluation
- Data upload and processing
- PostgreSQL database with Redis caching

## Prerequisites

- Docker and Docker Compose
- Git (for cloning)

## Quick Start

1. **Clone and navigate to the project:**
   ```bash
   git clone <your-repo-url>
   cd FraudDetect
   ```

2. **Start all services:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Open your browser to `http://localhost:5000`
   - The application will be ready once all health checks pass

## Development

### Environment Variables

Key environment variables you can customize:

- `DATABASE_URL`: PostgreSQL connection string (default: uses Docker postgres service)
- `REDIS_URL`: Redis connection string (default: uses Docker redis service)
- `SESSION_SECRET`: Flask session secret (change in production!)
- `FLASK_ENV`: Environment mode (development/production)

### Running in Development Mode

For development with hot reloading:

```bash
# Copy the docker-compose file
cp docker-compose.yml docker-compose.dev.yml

# Edit the fraud-detection service in docker-compose.dev.yml:
# Add:
#   environment:
#     - FLASK_ENV=development
#   volumes:
#     - .:/app
#   command: ["uv", "run", "python", "main.py"]

docker-compose -f docker-compose.dev.yml up --build
```

### Database Management

The PostgreSQL database will be automatically initialized with the schema. Data persists in Docker volumes.

**Reset database:**
```bash
docker-compose down -v
docker-compose up --build
```

**View database logs:**
```bash
docker-compose logs postgres
```

## Services

- **fraud-detection**: Main Flask application (port 5000)
- **postgres**: PostgreSQL database (port 5432)
- **redis**: Redis cache (port 6379)

## API Endpoints

- `GET /`: Main dashboard
- `GET /health`: Health check endpoint
- `POST /predict`: Manual fraud prediction
- `POST /upload`: Upload CSV data
- `GET /train`: Model training interface
- `GET /transactions`: View transaction history

## Project Structure

```
├── app.py              # Flask application setup
├── main.py             # Application entry point
├── routes.py           # Web routes and API endpoints
├── models.py           # Database models
├── ml_models.py        # Machine learning models
├── data_processor.py   # Data processing utilities
├── templates/          # HTML templates
├── static/             # CSS/JS assets
├── saved_models/       # Trained ML models
└── attached_assets/    # Data files
```

## Stopping the Application

```bash
docker-compose down
```

To also remove volumes (WARNING: deletes all data):
```bash
docker-compose down -v
```

## Troubleshooting

**Check service health:**
```bash
docker-compose ps
```

**View application logs:**
```bash
docker-compose logs -f fraud-detection
```

**Access database directly:**
```bash
docker-compose exec postgres psql -U fraud_user -d fraud_detection
```

**Rebuild after code changes:**
```bash
docker-compose up --build
```
