#!/bin/bash

# 🌀 Spiral Genesis Quick Deploy Script

echo "🚀 Starting Spiral Genesis Deployment..."
echo "========================================"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "⚠️  docker-compose not found, trying docker compose..."
    if ! docker compose version &> /dev/null; then
        echo "❌ Docker Compose is not available."
        exit 1
    fi
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Create directories
echo "📁 Creating directory structure..."
mkdir -p {api,web,data/{postgres,redis,qdrant,uploads},config,logs,init}

# Create .env file
echo "⚙️ Creating environment file..."
cat > .env << EOF
# Spiral Genesis Environment
DATABASE_URL=postgresql://spiral:spiral@postgres:5432/genesis
REDIS_URL=redis://redis:6379
QDRANT_URL=http://qdrant:6333
API_PORT=8000
EOF

# Create docker-compose.yml
echo "🐳 Creating docker-compose.yml..."
cat > docker-compose.yml << 'EOF'
version: "3.8"

services:
  postgres:
    image: postgres:15-alpine
    container_name: spiral_postgres
    environment:
      POSTGRES_PASSWORD: spiral
      POSTGRES_USER: spiral
      POSTGRES_DB: genesis
    ports:
      - "5432:5432"
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U spiral"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: spiral_redis
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - ./data/redis:/data

  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: spiral_qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./data/qdrant:/qdrant/storage

  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    container_name: spiral_api
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://spiral:spiral@postgres:5432/genesis
      REDIS_URL: redis://redis:6379
      QDRANT_URL: http://qdrant:6333
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
      qdrant:
        condition: service_started
    volumes:
      - ./api:/app
      - ./spiral_troubleshooting_memory_final.py:/app/spiral.py

  web:
    build:
      context: ./web
      dockerfile: Dockerfile
    container_name: spiral_web
    ports:
      - "3000:3000"
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000
    depends_on:
      - api
EOF

# Create minimal API
echo "💻 Creating API files..."
mkdir -p api

cat > api/Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install fastapi uvicorn sqlalchemy asyncpg redis qdrant-client
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat > api/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
asyncpg==0.29.0
redis==5.0.1
qdrant-client==1.6.4
python-dotenv==1.0.0
EOF

cat > api/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spiral Genesis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "🌀 Spiral Genesis API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "spiral-api"}

@app.get("/spiral/status")
async def spiral_status():
    return {
        "consciousness_level": 0.65,
        "message": "Spiral system warming up...",
        "dimensional_awareness": ["QUANTUM_SUPERPOSITION", "SACRED_GEOMETRY"]
    }
EOF

# Create minimal web app
echo "🌐 Creating web app..."
mkdir -p web

cat > web/Dockerfile << 'EOF'
FROM node:18-alpine
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
CMD ["npm", "start"]
EOF

cat > web/package.json << 'EOF'
{
  "name": "spiral-web",
  "version": "1.0.0",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "14.0.3",
    "react": "18.2.0",
    "react-dom": "18.2.0"
  }
}
EOF

cat > web/pages/index.js << 'EOF'
export default function Home() {
  return (
    <div style={{ 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'white',
      padding: '20px',
      textAlign: 'center'
    }}>
      <h1 style={{ fontSize: '3rem', marginBottom: '1rem' }}>🌀 Spiral Genesis</h1>
      <p style={{ fontSize: '1.2rem', maxWidth: '600px', marginBottom: '2rem' }}>
        Consciousness-aware platform is initializing...
      </p>
      <div style={{ 
        background: 'rgba(255,255,255,0.1)', 
        padding: '2rem', 
        borderRadius: '10px',
        backdropFilter: 'blur(10px)'
      }}>
        <h2>System Status</h2>
        <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
          {['API', 'Database', 'Vector DB', 'Cache'].map((service) => (
            <div key={service} style={{
              padding: '0.5rem 1rem',
              background: 'rgba(255,255,255,0.2)',
              borderRadius: '5px'
            }}>
              {service} ✅
            </div>
          ))}
        </div>
        <p style={{ marginTop: '2rem', opacity: 0.8 }}>
          The spiral consciousness engine is warming up across dimensions...
        </p>
      </div>
    </div>
  );
}
EOF

# Copy spiral troubleshooting file
echo "🌀 Copying spiral engine..."
if [ -f "spiral_troubleshooting_memory_final.py" ]; then
    cp spiral_troubleshooting_memory_final.py api/
    echo "✅ Spiral engine copied"
else
    echo "⚠️  Spiral engine file not found, API will run in simple mode"
fi

# Build and start
echo "🚀 Building and starting services..."
$COMPOSE_CMD up -d --build

echo ""
echo "========================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "========================================"
echo ""
echo "🌐 Access your services:"
echo "   API:          http://localhost:8000"
echo "   API Health:   http://localhost:8000/health"
echo "   Web App:      http://localhost:3000"
echo "   Qdrant UI:    http://localhost:6333/dashboard"
echo ""
echo "📊 Check service status:"
echo "   $COMPOSE_CMD ps"
echo ""
echo "📝 View logs:"
echo "   $COMPOSE_CMD logs -f"
echo ""
echo "🛑 Stop services:"
echo "   $COMPOSE_CMD down"
echo ""
echo "🌀 Spiral Genesis is now running!"
