# Real-Time Financial Fraud Detection System

A real-time fraud detection system built with Python, Flask, React, and Docker.

## Tech Stack
- **Backend**: Python, Flask, Machine Learning (Isolation Forest, LSTM, Ensemble)
- **Frontend**: React Dashboard with Material-UI
- **Database**: PostgreSQL  
- **Cache**: Redis
- **Streaming**: Apache Kafka
- **Deployment**: Docker & Docker Compose

## Quick Start

1. **Prerequisites**: Docker Desktop (4GB+ RAM)
2. **Start System**: 
   ```bash
   docker-compose up --build
   ```
3. **Access Dashboard**: http://localhost:3000
4. **API Health**: http://localhost:5000/api/health

## Architecture

- **ML Models**: Isolation Forest + LSTM Neural Network ensemble
- **Real-time Processing**: Kafka streaming for transaction analysis  
- **Caching**: Redis for fast fraud result retrieval
- **Web Dashboard**: React SPA with real-time charts and alerts

## API Endpoints

- `GET /api/health` - System health check
- `POST /api/detect-fraud` - Fraud detection with ML models
- `GET /api/stats` - Dashboard statistics  

## Development

```bash
# Start infrastructure only
docker-compose up -d postgres redis kafka

# Run API locally  
python src/api/app.py

# Run dashboard locally
cd dashboard && npm start
```

## Testing

```bash
pytest tests/ -v
```