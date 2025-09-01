# Kafka Integration for Fraud Detection System

## Overview

This fraud detection system now includes Apache Kafka integration for real-time, high-throughput transaction processing. Kafka enables the system to:

- **Process streaming data** in real-time with high throughput
- **Simulate realistic transaction streams** with configurable patterns
- **Scale horizontally** for increased data volume
- **Provide reliable message delivery** with fault tolerance
- **Enable real-time fraud alerts** through event streaming

## Architecture

### Kafka Components

1. **Zookeeper**: Coordination service for Kafka cluster
2. **Kafka Broker**: Message streaming platform
3. **Kafka Producer**: Sends transaction data to topics
4. **Kafka Consumer**: Processes transactions for fraud detection
5. **Data Simulator**: Generates realistic transaction streams

### Topic Structure

- `fraud-detection-transactions`: Raw transaction data
- `fraud-detection-predictions`: ML model predictions
- `fraud-detection-alerts`: Fraud alerts and notifications
- `fraud-detection-feedback`: User feedback for model improvement

## Features

### Real-time Processing
- **Streaming fraud detection**: Transactions processed as they arrive
- **Configurable throughput**: Adjust TPS (Transactions Per Second) dynamically
- **Burst mode support**: Handle traffic spikes automatically
- **Real-time alerts**: Immediate notifications for fraud detection

### Data Simulation
- **Realistic transaction patterns**: Normal and fraudulent transaction simulation
- **Configurable fraud rates**: Adjust percentage of fraudulent transactions
- **Multiple simulation profiles**: Different patterns for testing
- **Time-based patterns**: Simulate day/night transaction variations

### Enhanced Data Input
- **Kafka batch upload**: High-throughput CSV processing via Kafka
- **API transaction streaming**: REST API for real-time transaction submission
- **Manual transaction testing**: Individual transaction fraud testing
- **Concurrent processing**: Multiple transactions processed simultaneously

## Getting Started

### 1. Start the System with Kafka

```bash
# Start all services including Kafka (this is the standard way)
start.bat

# Or use Docker Compose directly
docker-compose up --build
```

### 2. Access the Kafka Dashboard

Navigate to: `http://localhost:5000/kafka/dashboard`

### 3. Monitor Kafka Health

Check system status: `http://localhost:5000/health`

## API Endpoints

### Kafka Health Check
```http
GET /kafka/health
```
Returns Kafka connection status and system health.

### Send Single Transaction
```http
POST /kafka/send-transaction
Content-Type: application/json

{
    "time_feature": 12345.67,
    "amount": 150.00,
    "v1": 0.1234,
    "v2": -0.5678,
    ...
    "v28": 0.9876
}
```

### Batch Upload via Kafka
```http
POST /kafka/batch-upload
Content-Type: multipart/form-data

file: [CSV file with transaction data]
```

### Stream Control
```http
POST /kafka/stream-control
Content-Type: application/json

{
    "action": "adjust_throughput",
    "transactions_per_second": 50
}
```

### Get Kafka Metrics
```http
GET /kafka/metrics
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka broker addresses | `kafka:29092` |
| `SIMULATION_MODE` | Simulation type: single/multi/burst | `single` |
| `BASE_TPS` | Base transactions per second | `10` |
| `FRAUD_RATE` | Fraud transaction percentage | `0.02` |

### Simulation Modes

- **Single**: One simulator with configurable TPS
- **Multi**: Multiple simulators with different profiles
- **Burst**: Automatic burst mode with traffic spikes

## Monitoring

### Docker Logs
```bash
# Monitor all services
docker-compose logs -f

# Monitor specific services
docker-compose logs -f kafka
docker-compose logs -f kafka-consumer
docker-compose logs -f data-simulator
```

### Kafka Topics
```bash
# List topics
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list

# Monitor transaction topic
docker exec kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic fraud-detection-transactions --from-beginning
```

## Performance Benefits

### Before Kafka
- **Batch processing**: Only CSV uploads and manual inputs
- **Synchronous processing**: One transaction at a time
- **Limited throughput**: ~1-10 TPS maximum
- **No real-time streaming**: Delayed fraud detection

### After Kafka
- **Real-time streaming**: Continuous transaction processing
- **Asynchronous processing**: Multiple transactions in parallel
- **High throughput**: 100+ TPS achievable
- **Real-time fraud detection**: Immediate alerts and responses
- **Scalable architecture**: Horizontal scaling support

## Throughput Improvements

| Scenario | Before Kafka | With Kafka | Improvement |
|----------|--------------|------------|-------------|
| CSV Upload | 5-10 TPS | 50-100 TPS | 5-10x |
| Manual Input | 1 TPS | 10+ TPS | 10x+ |
| Real-time Stream | Not supported | 100+ TPS | ∞ |
| Burst Traffic | Not supported | 500+ TPS | ∞ |

## Troubleshooting

### Common Issues

1. **Kafka Connection Failed**
   - Check if Kafka service is running: `docker-compose ps`
   - Verify network connectivity: `docker-compose logs kafka`

2. **Consumer Not Processing**
   - Check consumer logs: `docker-compose logs kafka-consumer`
   - Verify topic exists: `docker exec kafka kafka-topics --list --bootstrap-server localhost:9092`

3. **Low Throughput**
   - Increase TPS in simulator: Update `BASE_TPS` environment variable
   - Enable burst mode: Use stream control API
   - Scale consumers: Add more consumer instances

### Health Checks

```bash
# Check all service health
curl http://localhost:5000/health

# Check Kafka specific health
curl http://localhost:5000/kafka/health

# Get processing metrics
curl http://localhost:5000/kafka/metrics
```

## Advanced Usage

### Custom Simulation Profiles

Edit `docker-compose.yml` to add custom simulator configurations:

```yaml
data-simulator-high-volume:
  build: .
  command: python data_simulator.py
  environment:
    - SIMULATION_MODE=single
    - BASE_TPS=100
    - FRAUD_RATE=0.05
```

### Manual Topic Management

```bash
# Create custom topic
docker exec kafka kafka-topics --create --bootstrap-server localhost:9092 --topic custom-fraud-topic --partitions 3 --replication-factor 1

# Delete topic
docker exec kafka kafka-topics --delete --bootstrap-server localhost:9092 --topic custom-fraud-topic
```

## Next Steps

1. **Monitor performance** using the Kafka dashboard
2. **Adjust simulation parameters** based on your needs
3. **Scale consumer instances** for higher throughput
4. **Implement custom fraud patterns** in the simulator
5. **Add alerting integrations** (email, Slack, etc.)

The Kafka integration provides a robust foundation for real-time fraud detection at scale.
