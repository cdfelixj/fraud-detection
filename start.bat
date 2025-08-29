@echo off
echo 🔧 Starting Fraud Detection System...
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    echo.
    echo To start Docker Desktop:
    echo 1. Open Docker Desktop application  
    echo 2. Wait for it to fully start (whale icon in system tray)
    echo 3. Run this script again
    echo.
    pause
    exit /b 1
)

echo ✅ Docker is running

REM Clean up any previous containers
echo 🧹 Cleaning up previous containers...
docker-compose down -v 2>nul

REM Prune unused images to free space
echo 🗑️  Cleaning up unused Docker images...
docker image prune -f

REM Build and start with better error handling
echo 🔨 Building services (this may take a few minutes)...
docker-compose build --no-cache
if %errorlevel% neq 0 (
    echo ❌ Build failed. Common fixes:
    echo 1. Check Docker Desktop has enough resources (4GB+ RAM)
    echo 2. Check internet connection for downloading dependencies
    echo 3. Try: docker system prune -a (to free space)
    echo.
    pause
    exit /b 1
)

echo ✅ Build completed successfully

echo 🚀 Starting all services...
docker-compose up -d

echo ⏳ Waiting for services to initialize (45 seconds)...
timeout /t 45 /nobreak >nul

echo 📊 Checking service status...
docker-compose ps

echo.
echo 📝 Service logs (last 10 lines each):
echo.
echo --- API Logs ---
docker-compose logs --tail=10 fraud-api
echo.
echo --- Dashboard Logs ---
docker-compose logs --tail=10 fraud-dashboard 2>nul
echo.

echo 🎉 Fraud Detection System is now running!
echo.
echo 📱 Dashboard: http://localhost:3000
echo 🔗 API Health: http://localhost:5000/api/health  
echo 📊 API Stats: http://localhost:5000/api/stats
echo 🗃️  Database: localhost:5432 (postgres/password)
echo 💾 Redis: localhost:6379
echo.
echo 📋 Useful commands:
echo   View all logs:     docker-compose logs -f
echo   View API logs:     docker-compose logs -f fraud-api  
echo   Restart services:  docker-compose restart
echo   Stop everything:   docker-compose down
echo   Full cleanup:      docker-compose down -v
echo.
echo Opening dashboard in browser...
timeout /t 3 /nobreak >nul
start http://localhost:3000

pause
