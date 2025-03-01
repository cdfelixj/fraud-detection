@echo off
echo 🚀 Starting Fraud Detection System...

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ✅ Created .env file. Please review and update the configuration.
)

REM Build and start services
echo 🔨 Building and starting services...
docker-compose up --build -d

REM Wait for services to start
echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak >nul

echo 🎉 Fraud Detection System is starting!
echo.
echo 📊 Dashboard: http://localhost:3000
echo 🔗 API: http://localhost:5000
echo 📚 API Health: http://localhost:5000/api/health
echo 🗃️ PostgreSQL: localhost:5432
echo 💾 Redis: localhost:6379
echo 📤 Kafka: localhost:9092
echo.
echo To stop the system: docker-compose down
echo To view logs: docker-compose logs -f
echo To restart: docker-compose restart
echo.
pause
