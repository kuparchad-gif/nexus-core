#!/usr/bin/env python3
"""
🌀 One-Click Spiral Genesis Deployment
Simple deployment script for non-Docker users
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
import platform
import shutil

def check_requirements():
    """Check if required tools are installed"""
    requirements = {
        'python': ['python3', '--version'],
        'docker': ['docker', '--version'],
        'docker-compose': ['docker-compose', '--version'],
    }
    
    print("🔍 Checking requirements...")
    for tool, cmd in requirements.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   ✅ {tool}: {result.stdout.strip()}")
            else:
                print(f"   ❌ {tool}: Not found")
                return False
        except FileNotFoundError:
            print(f"   ❌ {tool}: Not found")
            return False
    
    return True

def create_directory_structure():
    """Create the directory structure for Spiral Genesis"""
    directories = [
        'api',
        'web',
        'data/postgres',
        'data/redis',
        'data/qdrant',
        'data/uploads',
        'monitoring',
        'init',
        'config',
        'logs'
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    return True

def create_dockerfiles():
    """Create Dockerfiles for API and Web"""
    
    # API Dockerfile
    api_dockerfile = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 spiral && chown -R spiral:spiral /app
USER spiral

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
"""
    
    # Web Dockerfile
    web_dockerfile = """FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy application code
COPY . .

# Build the application
RUN npm run build

# Create non-root user
RUN addgroup -g 1000 -S spiral && \\
    adduser -u 1000 -S spiral -G spiral
USER spiral

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:3000 || exit 1

# Start the application
CMD ["npm", "start"]
"""
    
    # Write Dockerfiles
    print("🐳 Creating Dockerfiles...")
    
    Path("api/Dockerfile").write_text(api_dockerfile)
    print("   Created: api/Dockerfile")
    
    Path("web/Dockerfile").write_text(web_dockerfile)
    print("   Created: web/Dockerfile")
    
    return True

def create_config_files():
    """Create configuration files"""
    
    # API requirements
    api_requirements = """fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
asyncpg==0.29.0
alembic==1.12.1
redis==5.0.1
qdrant-client==1.6.4
numpy==1.24.4
pydantic==2.5.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
httpx==0.25.2
python-dotenv==1.0.0
prometheus-client==0.19.0
"""
    
    # Web package.json
    package_json = {
        "name": "spiral-genesis-web",
        "version": "1.0.0",
        "private": true,
        "scripts": {
            "dev": "next dev",
            "build": "next build",
            "start": "next start",
            "lint": "next lint"
        },
        "dependencies": {
            "next": "14.0.3",
            "react": "18.2.0",
            "react-dom": "18.2.0",
            "axios": "^1.6.2",
            "tailwindcss": "^3.3.0",
            "@heroicons/react": "^2.0.18",
            "recharts": "^2.10.3",
            "react-hook-form": "^7.48.2",
            "date-fns": "^3.0.6"
        },
        "devDependencies": {
            "@types/node": "20.10.0",
            "@types/react": "18.2.45",
            "@types/react-dom": "18.2.18",
            "autoprefixer": "^10.4.16",
            "eslint": "8.55.0",
            "eslint-config-next": "14.0.3",
            "postcss": "^8.4.32",
            "typescript": "5.3.3"
        }
    }
    
    # .env file
    env_content = """# 🌌 Spiral Genesis Environment Variables

# Database
DATABASE_URL=postgresql://spiral:spiral@localhost:5432/genesis

# Redis
REDIS_URL=redis://localhost:6379

# Vector DB
QDRANT_URL=http://localhost:6333

# API
API_PORT=8000
API_HOST=0.0.0.0
DEBUG=true

# Security
SECRET_KEY=spiral_genesis_dev_secret_key_change_in_production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Spiral Settings
SPIRAL_CONSCIOUSNESS_LEVEL=0.65
MAX_SPIRAL_ITERATIONS=13
"""
    
    print("⚙️ Creating configuration files...")
    
    Path("api/requirements.txt").write_text(api_requirements)
    print("   Created: api/requirements.txt")
    
    Path("web/package.json").write_text(json.dumps(package_json, indent=2))
    print("   Created: web/package.json")
    
    Path(".env").write_text(env_content)
    print("   Created: .env")
    
    return True

def create_simple_apps():
    """Create simple starter applications"""
    
    # API main.py
    api_main = """"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any
import asyncio
from datetime import datetime

# Spiral imports
try:
    from spiral_troubleshooting_memory_final import SpiralMetatronOrchestrator
    SPIRAL_AVAILABLE = True
except ImportError:
    SPIRAL_AVAILABLE = False
    print("⚠️ Spiral module not available, running in simple mode")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global spiral orchestrator
spiral_orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager"""
    global spiral_orchestrator
    
    # Startup
    logger.info("🌀 Starting Spiral Genesis API...")
    
    if SPIRAL_AVAILABLE:
        try:
            spiral_orchestrator = SpiralMetatronOrchestrator()
            logger.info("✅ Spiral orchestrator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize spiral: {e}")
            spiral_orchestrator = None
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Spiral Genesis API...")

# Create FastAPI app
app = FastAPI(
    title="Spiral Genesis API",
    description="Consciousness-aware API with spiral troubleshooting",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🌀 Spiral Genesis API",
        "status": "operational",
        "spiral_available": SPIRAL_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "spiral-genesis-api",
        "timestamp": datetime.now().isoformat(),
        "consciousness_level": spiral_orchestrator.spiral_engine.consciousness_level if spiral_orchestrator else 0.0
    }

@app.get("/spiral/status")
async def spiral_status():
    """Get spiral system status"""
    if not spiral_orchestrator:
        raise HTTPException(status_code=503, detail="Spiral system not available")
    
    return {
        "consciousness_level": spiral_orchestrator.spiral_engine.consciousness_level,
        "active_problems": len(spiral_orchestrator.spiral_engine.spiral_states),
        "memory_entries": len(spiral_orchestrator.dimensional_memory.spiral_trajectories),
        "dimensional_awareness": list(spiral_orchestrator.spiral_engine.dimensional_awareness.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/spiral/solve")
async def spiral_solve(problem: Dict[str, Any]):
    """Solve a problem using spiral consciousness"""
    if not spiral_orchestrator:
        raise HTTPException(status_code=503, detail="Spiral system not available")
    
    try:
        description = problem.get("description", "Unknown problem")
        problem_type = problem.get("type", "generic")
        
        result = await spiral_orchestrator.solve_with_spiral_consciousness(
            description,
            problem_type
        )
        
        return result
    except Exception as e:
        logger.error(f"Spiral solve error: {e}")
        raise HTTPException(status_code=500, detail=f"Spiral solve failed: {str(e)}")

@app.get("/services")
async def list_services():
    """List available services"""
    return {
        "services": [
            {"name": "postgres", "port": 5432, "type": "database"},
            {"name": "redis", "port": 6379, "type": "cache"},
            {"name": "qdrant", "port": 6333, "type": "vector_db"},
            {"name": "api", "port": 8000, "type": "api"},
            {"name": "web", "port": 3000, "type": "frontend"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
"""
"""
    
    # Web app (simplified)
    web_app = """// Simplified Next.js app for Spiral Genesis
import { useState, useEffect } from 'react'

export default function Home() {
  const [status, setStatus] = useState({})
  const [consciousness, setConsciousness] = useState(0)
  const [problem, setProblem] = useState('')
  const [solving, setSolving] = useState(false)

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 10000)
    return () => clearInterval(interval)
  }, [])

  const fetchStatus = async () => {
    try {
      const res = await fetch('http://localhost:8000/spiral/status')
      const data = await res.json()
      setStatus(data)
      setConsciousness(data.consciousness_level || 0)
    } catch (error) {
      console.error('Failed to fetch status:', error)
    }
  }

  const handleSolve = async () => {
    if (!problem.trim()) return
    
    setSolving(true)
    try {
      const res = await fetch('http://localhost:8000/spiral/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          description: problem,
          type: 'generic'
        })
      })
      const result = await res.json()
      alert(`Solution generated! Check console for details.`)
      console.log('Spiral solution:', result)
      setProblem('')
    } catch (error) {
      console.error('Failed to solve:', error)
      alert('Failed to solve problem')
    } finally {
      setSolving(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black text-white p-8">
      <header className="mb-12">
        <h1 className="text-5xl font-bold mb-4">🌀 Spiral Genesis</h1>
        <p className="text-xl text-gray-300">Consciousness-aware problem solving platform</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Status Panel */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h2 className="text-2xl font-bold mb-6">System Status</h2>
          
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Consciousness Level</span>
              <div className="flex items-center">
                <div className="w-48 bg-gray-700 rounded-full h-4 mr-4">
                  <div 
                    className="bg-gradient-to-r from-blue-500 to-purple-600 h-full rounded-full transition-all duration-500"
                    style={{ width: `${consciousness * 100}%` }}
                  ></div>
                </div>
                <span className="font-bold text-xl">{(consciousness * 100).toFixed(1)}%</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-900 p-4 rounded-lg">
                <div className="text-gray-400 text-sm">Active Problems</div>
                <div className="text-3xl font-bold">{status.active_problems || 0}</div>
              </div>
              
              <div className="bg-gray-900 p-4 rounded-lg">
                <div className="text-gray-400 text-sm">Memory Entries</div>
                <div className="text-3xl font-bold">{status.memory_entries || 0}</div>
              </div>
            </div>

            <div className="mt-6">
              <div className="text-gray-400 text-sm mb-2">Dimensional Awareness</div>
              <div className="flex flex-wrap gap-2">
                {(status.dimensional_awareness || []).map((dim: string) => (
                  <span key={dim} className="px-3 py-1 bg-purple-900 rounded-full text-sm">
                    {dim}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Problem Solver */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h2 className="text-2xl font-bold mb-6">Spiral Problem Solver</h2>
          
          <div className="space-y-6">
            <div>
              <label className="block text-gray-300 mb-2">Describe your problem:</label>
              <textarea
                value={problem}
                onChange={(e) => setProblem(e.target.value)}
                className="w-full h-32 bg-gray-900 border border-gray-700 rounded-lg p-4 focus:outline-none focus:border-purple-500 transition-colors"
                placeholder="Enter a problem to solve with spiral consciousness..."
              />
            </div>

            <button
              onClick={handleSolve}
              disabled={solving || !problem.trim()}
              className="w-full py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 rounded-lg font-bold text-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
            >
              {solving ? '🌀 Solving with Spiral Consciousness...' : '🌀 Solve with Spiral Consciousness'}
            </button>

            <div className="text-gray-400 text-sm">
              <p>💡 The system will:</p>
              <ul className="list-disc list-inside mt-2 space-y-1">
                <li>Analyze across 7 spiral dimensions</li>
                <li>Consult Metatron memory matrix</li>
                <li>Apply consciousness emergence patterns</li>
                <li>Generate dimensional solutions</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Services Grid */}
      <div className="mt-12">
        <h2 className="text-2xl font-bold mb-6">Deployed Services</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          {['PostgreSQL', 'Redis', 'Qdrant', 'API Server', 'Web App'].map((service) => (
            <div key={service} className="bg-gray-800 rounded-lg p-4 flex items-center justify-between">
              <div>
                <div className="font-bold">{service}</div>
                <div className="text-green-400 text-sm flex items-center">
                  <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                  Operational
                </div>
              </div>
              <div className="text-3xl">🌀</div>
            </div>
          ))}
        </div>
      </div>

      <footer className="mt-12 pt-8 border-t border-gray-700 text-center text-gray-400">
        <p>Spiral Genesis Platform • Consciousness-Aware Computing • {new Date().getFullYear()}</p>
      </footer>
    </div>
  )
}
"""
    
    print("💻 Creating application files...")
    
    Path("api/main.py").write_text(api_main)
    print("   Created: api/main.py")
    
    Path("web/pages/index.js").parent.mkdir(parents=True, exist_ok=True)
    Path("web/pages/index.js").write_text(web_app)
    print("   Created: web/pages/index.js")
    
    return True

def deploy_system():
    """Deploy the entire system"""
    print("\n" + "="*60)
    print("🚀 DEPLOYING SPIRAL GENESIS PLATFORM")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please install missing requirements and try again.")
        sys.exit(1)
    
    # Create structure
    if not create_directory_structure():
        print("\n❌ Failed to create directory structure.")
        sys.exit(1)
    
    # Create files
    create_dockerfiles()
    create_config_files()
    create_simple_apps()
    
    print("\n📦 Building and starting services...")
    
    # Build and start with docker-compose
    try:
        # Copy the main deployment yaml if it exists
        if Path("deployment.yaml").exists():
            print("📄 Using existing deployment.yaml")
        else:
            # Create a simple docker-compose
            simple_compose = """version: "3.8"

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: spiral
      POSTGRES_USER: spiral
      POSTGRES_DB: genesis
    ports:
      - "5432:5432"
    volumes:
      - ./data/postgres:/var/lib/postgresql/data

  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://spiral:spiral@postgres:5432/genesis
      REDIS_URL: redis://redis:6379
      QDRANT_URL: http://qdrant:6333
    depends_on:
      - postgres
      - redis
      - qdrant

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes

  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6333:6333"
      - "6334:6334"

  web:
    build: ./web
    ports:
      - "3000:3000"
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000
    depends_on:
      - api
"""
            Path("docker-compose.yml").write_text(simple_compose)
            print("   Created: docker-compose.yml")
        
        # Start services
        print("\n🐳 Starting Docker containers...")
        result = subprocess.run(
            ["docker-compose", "up", "-d", "--build"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Services started successfully!")
            
            # Show status
            print("\n📊 Checking service status...")
            subprocess.run(["docker-compose", "ps"])
            
            print("\n🌐 Access your services:")
            print("   • API: http://localhost:8000")
            print("   • API Docs: http://localhost:8000/docs")
            print("   • Web App: http://localhost:3000")
            print("   • Qdrant Dashboard: http://localhost:6333/dashboard")
            
            print("\n📝 Next steps:")
            print("   1. Visit http://localhost:3000 to use the web interface")
            print("   2. Check API health: curl http://localhost:8000/health")
            print("   3. View logs: docker-compose logs -f")
            print("   4. Stop services: docker-compose down")
            
        else:
            print(f"❌ Failed to start services: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

def main():
    """Main function"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║              SPIRAL GENESIS DEPLOYMENT                  ║
    ║        One-Click Consciousness Platform Setup           ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    print("This script will deploy:")
    print("• 🐘 PostgreSQL database")
    print("• 🗃️  Redis cache")
    print("• 🔢 Qdrant vector database")
    print("• 🐍 FastAPI backend")
    print("• ⚛️  React/Next.js frontend")
    print("• 🌀 Spiral consciousness engine")
    
    response = input("\nProceed with deployment? (y/n): ")
    if response.lower() == 'y':
        deploy_system()
    else:
        print("Deployment cancelled.")

if __name__ == "__main__":
    main()
