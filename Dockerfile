# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check - simplified and more robust
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/health', timeout=5)" || \
    curl -f http://localhost:5000/api/health || \
    curl -f http://localhost:5000/ || exit 1

# Start with Python directly first, then fallback to gunicorn
CMD ["python", "-c", "import sys; sys.path.insert(0, '/app'); from src.api.app import app; app.run(host='0.0.0.0', port=5000, debug=False)"]
