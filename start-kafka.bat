@echo off
echo Starting Fraud Detection System with Kafka...

REM Start the services
docker-compose up -d

echo.
echo Services starting... This may take a few minutes.
echo.
echo You can monitor the startup with:
echo   docker-compose logs -f
echo.
echo Once ready, access:
echo   Web App: http://localhost:5000
echo   Kafka Dashboard: http://localhost:5000/kafka/dashboard
echo.
echo To stop all services, run: stop.bat
