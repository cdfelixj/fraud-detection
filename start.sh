#!/bin/bash

# Fraud Detection System Startup Script

set -e

echo "🚀 Starting Fraud Detection System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file. Please review and update the configuration."
fi

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

check_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo "✅ $service is healthy"
            return 0
        fi
        echo "⏳ Waiting for $service... (attempt $attempt/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done

    echo "❌ $service failed to start"
    return 1
}

# Check API health
if check_service "API" "http://localhost:5000/api/health"; then
    echo "✅ API is ready"
else
    echo "❌ API failed to start"
fi

# Check Dashboard
if check_service "Dashboard" "http://localhost:3000"; then
    echo "✅ Dashboard is ready"
else
    echo "❌ Dashboard failed to start"
fi

echo ""
echo "🎉 Fraud Detection System is running!"
echo ""
echo "📊 Dashboard: http://localhost:3000"
echo "🔗 API: http://localhost:5000"
echo "📚 API Health: http://localhost:5000/api/health"
echo "🗃️ PostgreSQL: localhost:5432"
echo "💾 Redis: localhost:6379"
echo "📤 Kafka: localhost:9092"
echo ""
echo "To stop the system: docker-compose down"
echo "To view logs: docker-compose logs -f"
echo "To restart: docker-compose restart"
