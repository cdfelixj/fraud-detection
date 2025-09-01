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
echo Building and starting services...
echo Services include: PostgreSQL, Redis, Zookeeper, Kafka, Web App, Consumer
echo This may take a few minutes on first run...
echo.

docker-compose up --build

echo.
echo Application stopped.
pause
