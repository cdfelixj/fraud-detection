@echo off
echo Starting Fraud Detection System with Kafka Streaming...
echo.

echo Checking if Docker is running...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not running.
    echo Please install Docker Desktop and make sure it's running.
    pause
    exit /b 1
)

echo Checking if Docker Compose is available...
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker Compose is not available.
    echo Please make sure Docker Desktop is properly installed.
    pause
    exit /b 1
)

echo.
echo Starting infrastructure services first (Zookeeper, Kafka, PostgreSQL, Redis)...
docker-compose up -d zookeeper postgres redis

echo Waiting for infrastructure to initialize...
timeout /t 15 /nobreak >nul

echo Starting Kafka broker...
docker-compose up -d kafka

echo Waiting for Kafka to be ready...
timeout /t 20 /nobreak >nul

echo Starting application services...
docker-compose up -d fraud-detection kafka-consumer data-simulator

echo.
echo All services started! Access points:
echo   Web App: http://localhost:5000
echo   Kafka Dashboard: http://localhost:5000/kafka/dashboard
echo   Health Check: http://localhost:5000/health
echo.
echo To view logs: docker-compose logs -f
echo To stop: stop.bat
echo.
pause
