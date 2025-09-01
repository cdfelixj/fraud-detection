@echo off
echo Starting Fraud Detection System in Development Mode with Kafka...

REM Use the development docker-compose file if it exists
if exist docker-compose.dev.yml (
    docker-compose -f docker-compose.dev.yml up -d
) else (
    docker-compose up -d
)

echo.
echo Development services starting...
echo.
echo Kafka Monitoring:
echo   docker-compose logs -f kafka
echo.
echo Consumer Monitoring:
echo   docker-compose logs -f kafka-consumer
echo.
echo Simulator Monitoring:
echo   docker-compose logs -f data-simulator
echo.
echo Access points:
echo   Web App: http://localhost:5000
echo   Kafka Streaming: http://localhost:5000/kafka/dashboard
echo   Health Check: http://localhost:5000/health
