# Real-Time Financial Fraud Detection System

A comprehensive real-time fraud detection system that uses machine learning models to identify potentially fraudulent financial transactions. Built with modern technologies including Python, Apache Kafka, Redis, PostgreSQL, and React.

## 🚀 Features

- **Real-time Transaction Processing**: Stream processing with Apache Kafka
- **Machine Learning Models**: Isolation Forest, LSTM networks, and ensemble methods
- **Interactive Dashboard**: React-based dashboard with real-time updates
- **Scalable Architecture**: Microservices with Docker containers
- **Data Pipeline**: ETL processes with automated feature engineering
- **Alert System**: Real-time fraud alerts with configurable thresholds
- **API Integration**: RESTful API for integration with external systems

## 🛠 Tech Stack

- **Backend**: Python, Flask, Gunicorn
- **Machine Learning**: scikit-learn, TensorFlow, NumPy, pandas
- **Streaming**: Apache Kafka
- **Database**: PostgreSQL
- **Caching**: Redis
- **Frontend**: React, Chart.js, Axios
- **Infrastructure**: Docker, Docker Compose, Nginx
- **Testing**: pytest, unittest

## 🏗 Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   React         │    │  Flask API   │    │  ML Models      │
│   Dashboard     │◄──►│  Gateway     │◄──►│  (Ensemble)     │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   PostgreSQL    │◄──►│   Stream     │◄──►│  Apache Kafka   │
│   Database      │    │  Processor   │    │  (Messages)     │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │    Redis     │
                       │   (Cache)    │
                       └──────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- 4GB+ RAM available
- Ports 3000, 5000, 5432, 6379, 9092, 2181 available

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd fraud-detection-system
```

### 2. Start the System
**Windows:**
```cmd
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

**Manual Docker Compose:**
```bash
docker-compose up --build -d
```

### 3. Access the System
- **Dashboard**: http://localhost:3000
- **API**: http://localhost:5000
- **API Health**: http://localhost:5000/api/health

## 📊 Usage

### Dashboard Features
- **Real-time Statistics**: Total transactions, fraud detected, risk levels
- **Interactive Charts**: Fraud probability trends over time
- **Alert Management**: View and manage fraud alerts
- **Transaction Monitor**: Recent high-risk transactions

### API Endpoints
```http
GET  /api/health                 # System health check
POST /api/detect-fraud           # Fraud detection
POST /api/predict               # Legacy prediction endpoint
GET  /api/cache/<key>           # Get cached data
POST /api/cache                 # Store cached data
```

### Example API Usage
```bash
# Test fraud detection
curl -X POST http://localhost:5000/api/detect-fraud \
  -H "Content-Type: application/json" \
  -d '{
    "features": [100.50, 12, 1, 2, 1],
    "transaction_id": "T123456"
  }'
```

## 🧠 Machine Learning Models

### Isolation Forest
- **Purpose**: Anomaly detection in transaction patterns
- **Features**: Amount, time, location, merchant category
- **Output**: Anomaly score and binary classification

### LSTM Network
- **Purpose**: Sequential pattern analysis
- **Features**: Time series of transaction amounts and frequencies  
- **Output**: Fraud probability based on temporal patterns

### Ensemble Method
- **Combination**: Weighted average of Isolation Forest and LSTM
- **Weights**: Configurable (default: 60% IF, 40% LSTM)
- **Output**: Final fraud probability and risk classification

## 🔧 Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:

```env
# Database
POSTGRES_HOST=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=fraud_detection_db

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=fraud_detection_topic

# Model Parameters
IF_CONTAMINATION=0.1
LSTM_SEQUENCE_LENGTH=10
LSTM_FEATURES=5
```

### Model Tuning
Adjust model parameters in `src/utils/config.py`:
- Isolation Forest contamination rate
- LSTM network architecture
- Ensemble weights
- Risk thresholds

## 🧪 Testing

Run the test suite:
```bash
# Unit tests
python -m pytest tests/ -v

# Specific test modules
python -m pytest tests/test_models.py -v
python -m pytest tests/test_api.py -v
python -m pytest tests/test_streaming.py -v
```

## 📈 Monitoring

### System Health
- API health endpoint: `/api/health`
- Database connectivity checks
- Redis availability
- Kafka consumer status

### Performance Metrics
- Transaction processing rate
- Fraud detection accuracy
- Model response times
- System resource usage

## 🔒 Security

- Input validation and sanitization
- SQL injection prevention
- API rate limiting
- Secure container configurations
- Environment variable management

## 📦 Deployment

### Development
```bash
docker-compose up --build
```

### Production
1. Update environment variables for production
2. Enable SSL/TLS certificates
3. Configure load balancing
4. Set up monitoring and logging
5. Implement backup strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Port Conflicts**
```bash
# Check port usage
netstat -an | findstr :5000
# Kill process if needed
taskkill /PID <pid> /F
```

**Docker Issues**
```bash
# Reset Docker containers
docker-compose down -v
docker system prune -a
docker-compose up --build
```

**Database Connection**
```bash
# Check PostgreSQL logs
docker-compose logs postgres
# Reset database
docker-compose down -v
docker-compose up postgres
```

### Logs
```bash
# View all logs
docker-compose logs -f

# Specific service logs
docker-compose logs -f fraud-api
docker-compose logs -f dashboard
```

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

---

**Built with ❤️ for financial security**
     cd dashboard
     npm install
     ```

3. **Configure Database and Kafka**
   - Update the configuration settings in `src/utils/config.py` to match your database and Kafka setup.

4. **Run the Application**
   - Start the backend:
     ```bash
     python src/api/app.py
     ```
   - Start the Kafka producer and consumer:
     ```bash
     python src/streaming/kafka_producer.py
     python src/streaming/kafka_consumer.py
     ```
   - Start the React dashboard:
     ```bash
     cd dashboard
     npm start
     ```

## Usage Guidelines
- The system listens for incoming financial transaction data through Kafka.
- Detected anomalies are processed and can be viewed on the React dashboard.
- Use the provided API to interact with the system and retrieve data.

## Testing
- Unit tests are available in the `tests` directory. Run the tests to ensure the functionality of the components:
  ```bash
  pytest tests/
  ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.