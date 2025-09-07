@echo off
echo =====================================================
echo FRAUD DETECTION - HIGH PERFORMANCE CONSUMER
echo =====================================================
echo.

REM Check if running in Docker
if defined DOCKER_CONTAINER (
    echo Running in Docker container...
    python start_fast_consumer.py --parallel=2 --batch-size=100
) else (
    echo Running locally...
    echo.
    echo Usage options:
    echo   start_fast_consumer.bat                 - Single consumer, batch size 50
    echo   start_fast_consumer.bat parallel       - 2 parallel consumers
    echo   start_fast_consumer.bat turbo          - 4 parallel consumers, batch size 100
    echo.
    
    if "%1"=="parallel" (
        echo Starting 2 parallel consumers...
        python start_fast_consumer.py --parallel=2 --batch-size=50
    ) else if "%1"=="turbo" (
        echo Starting TURBO mode: 4 parallel consumers, batch size 100...
        python start_fast_consumer.py --parallel=4 --batch-size=100
    ) else (
        echo Starting single high-performance consumer...
        python start_fast_consumer.py --parallel=1 --batch-size=50
    )
)

pause
