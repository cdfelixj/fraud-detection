@echo off
echo Stopping Fraud Detection System...
echo.

docker-compose down

echo.
echo All services stopped.
echo To remove all data (WARNING: This will delete the database), run:
echo docker-compose down -v
echo.
pause
