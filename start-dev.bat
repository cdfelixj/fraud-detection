@echo off
echo Starting Fraud Detection System in Development Mode...
echo.

echo This will start the application with hot reloading enabled.
echo Code changes will be reflected automatically.
echo.

docker-compose -f docker-compose.dev.yml up --build

echo.
echo Development application stopped.
pause
