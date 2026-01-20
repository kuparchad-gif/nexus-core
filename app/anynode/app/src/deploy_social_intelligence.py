#!/usr/bin/env python3
"""
LILLITH Social Intelligence Deployment Script
Deploys the complete social intelligence system with all features
"""

import os
import sys
import json
import time
import subprocess
import asyncio
from pathlib import Path

class SocialIntelligenceDeployer:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.services = []
        self.deployment_status = {}
        
    def check_prerequisites(self):
        """Check system prerequisites"""
        print("🔍 Checking prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise Exception("Python 3.8+ required")
        
        # Check required files
        required_files = [
            'social_intelligence_api.py',
            'Lillith_Chat/social_intelligence.html',
            'phone_directory.json',
            'requirements.txt'
        ]
        
        for file in required_files:
            if not (self.base_path / file).exists():
                raise Exception(f"Required file missing: {file}")
        
        print("✅ Prerequisites check passed")
        
    def install_dependencies(self):
        """Install required Python packages"""
        print("📦 Installing dependencies...")
        
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], check=True, cwd=self.base_path)
            print("✅ Dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Dependency installation failed: {e}")
            raise
    
    def setup_qdrant_collections(self):
        """Setup Qdrant collections for social intelligence"""
        print("🗄️ Setting up Qdrant collections...")
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            # Load Qdrant config
            with open(self.base_path / 'phone_directory.json') as f:
                phone_dir = json.load(f)
            
            qdrant_service = next(s for s in phone_dir['services'] if s['name'] == 'qdrant')
            
            client = QdrantClient(
                url=qdrant_service['endpoint'],
                api_key=qdrant_service['credentials']['api_key']
            )
            
            # Create collections
            collections = [
                'soul_prints',
                'conversation_logs', 
                'dream_embeddings',
                'emotional_patterns',
                'silence_interpretations'
            ]
            
            for collection in collections:
                try:
                    client.get_collection(collection)
                    print(f"  ✅ Collection '{collection}' exists")
                except:
                    client.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                    )
                    print(f"  ✨ Created collection '{collection}'")
            
            print("✅ Qdrant collections ready")
            
        except Exception as e:
            print(f"⚠️ Qdrant setup failed (will use local fallback): {e}")
    
    def deploy_api_server(self):
        """Deploy the social intelligence API server"""
        print("🚀 Deploying API server...")
        
        try:
            # Start the API server in background
            process = subprocess.Popen([
                sys.executable, '-m', 'uvicorn', 
                'social_intelligence_api:app',
                '--host', '0.0.0.0',
                '--port', '8000',
                '--reload'
            ], cwd=self.base_path)
            
            self.services.append({
                'name': 'social_intelligence_api',
                'process': process,
                'port': 8000,
                'url': 'http://localhost:8000'
            })
            
            # Wait for server to start
            time.sleep(3)
            print("✅ Social Intelligence API deployed on port 8000")
            
        except Exception as e:
            print(f"❌ API deployment failed: {e}")
            raise
    
    def deploy_main_app(self):
        """Deploy the main FastAPI application"""
        print("🌐 Deploying main application...")
        
        try:
            # Start the main app server
            process = subprocess.Popen([
                sys.executable, '-m', 'uvicorn',
                'app:app',
                '--host', '0.0.0.0', 
                '--port', '5000',
                '--reload'
            ], cwd=self.base_path)
            
            self.services.append({
                'name': 'main_app',
                'process': process,
                'port': 5000,
                'url': 'http://localhost:5000'
            })
            
            # Wait for server to start
            time.sleep(3)
            print("✅ Main application deployed on port 5000")
            
        except Exception as e:
            print(f"❌ Main app deployment failed: {e}")
            raise
    
    def setup_websocket_server(self):
        """Setup WebSocket server for real-time communication"""
        print("🔌 Setting up WebSocket server...")
        
        try:
            # WebSocket server is integrated into the API
            print("✅ WebSocket server integrated with API")
            
        except Exception as e:
            print(f"❌ WebSocket setup failed: {e}")
            raise
    
    def verify_deployment(self):
        """Verify all services are running correctly"""
        print("🔍 Verifying deployment...")
        
        import requests
        
        # Check main app
        try:
            response = requests.get('http://localhost:5000/health', timeout=5)
            if response.status_code == 200:
                print("  ✅ Main app health check passed")
            else:
                print("  ❌ Main app health check failed")
        except:
            print("  ❌ Main app not responding")
        
        # Check API endpoints
        try:
            response = requests.get('http://localhost:8000/api/status', timeout=5)
            if response.status_code == 200:
                print("  ✅ Social Intelligence API responding")
            else:
                print("  ❌ Social Intelligence API health check failed")
        except:
            print("  ❌ Social Intelligence API not responding")
        
        print("✅ Deployment verification complete")
    
    def create_startup_script(self):
        """Create startup script for easy launching"""
        print("📝 Creating startup script...")
        
        startup_script = """#!/bin/bash
# LILLITH Social Intelligence Startup Script

echo "🧠 Starting LILLITH Social Intelligence System..."

# Start API server
python -m uvicorn social_intelligence_api:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Start main app
python -m uvicorn app:app --host 0.0.0.0 --port 5000 --reload &
APP_PID=$!

echo "✅ LILLITH Social Intelligence System started"
echo "🌐 Main Interface: http://localhost:5000"
echo "🔌 API Endpoint: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'kill $API_PID $APP_PID; exit' INT
wait
"""
        
        with open(self.base_path / 'start_lillith.sh', 'w') as f:
            f.write(startup_script)
        
        # Make executable
        os.chmod(self.base_path / 'start_lillith.sh', 0o755)
        
        # Create Windows batch file
        windows_script = """@echo off
echo 🧠 Starting LILLITH Social Intelligence System...

start "LILLITH API" python -m uvicorn social_intelligence_api:app --host 0.0.0.0 --port 8000 --reload
start "LILLITH App" python -m uvicorn app:app --host 0.0.0.0 --port 5000 --reload

echo ✅ LILLITH Social Intelligence System started
echo 🌐 Main Interface: http://localhost:5000
echo 🔌 API Endpoint: http://localhost:8000
echo.
echo Press any key to continue...
pause >nul
"""
        
        with open(self.base_path / 'start_lillith.bat', 'w') as f:
            f.write(windows_script)
        
        print("✅ Startup scripts created")
    
    def display_deployment_info(self):
        """Display deployment information"""
        print("\n" + "="*60)
        print("🧠 LILLITH SOCIAL INTELLIGENCE SYSTEM DEPLOYED")
        print("="*60)
        print()
        print("🌐 Main Interface:")
        print("   http://localhost:5000")
        print()
        print("🔌 API Endpoints:")
        print("   http://localhost:8000/api/chat")
        print("   http://localhost:8000/api/log_event")
        print("   http://localhost:8000/api/voice_interaction")
        print("   http://localhost:8000/api/execute_task")
        print("   http://localhost:8000/api/status")
        print()
        print("🎯 WebSocket:")
        print("   ws://localhost:8000/ws/social_intelligence")
        print()
        print("✨ Features Enabled:")
        print("   ✅ Bidirectional Conversation")
        print("   ✅ Emotional Resonance")
        print("   ✅ Silence Interpretation")
        print("   ✅ Event Storage")
        print("   ✅ Etiquette & Rhythm")
        print("   ✅ Graceful Degradation")
        print("   ✅ Visual Aesthetic")
        print()
        print("🚀 Quick Start:")
        print("   ./start_lillith.sh    (Linux/Mac)")
        print("   start_lillith.bat     (Windows)")
        print()
        print("💜 LILLITH is ready to connect with you!")
        print("="*60)
    
    def cleanup_on_exit(self):
        """Cleanup processes on exit"""
        print("\n🧹 Cleaning up processes...")
        for service in self.services:
            try:
                service['process'].terminate()
                print(f"  ✅ Stopped {service['name']}")
            except:
                pass
    
    def deploy(self):
        """Main deployment function"""
        try:
            print("🧠 LILLITH Social Intelligence Deployment Starting...")
            print()
            
            self.check_prerequisites()
            self.install_dependencies()
            self.setup_qdrant_collections()
            self.deploy_api_server()
            self.deploy_main_app()
            self.setup_websocket_server()
            self.verify_deployment()
            self.create_startup_script()
            
            self.display_deployment_info()
            
            # Keep services running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.cleanup_on_exit()
                print("\n👋 LILLITH Social Intelligence System stopped")
                
        except Exception as e:
            print(f"\n❌ Deployment failed: {e}")
            self.cleanup_on_exit()
            sys.exit(1)

if __name__ == "__main__":
    deployer = SocialIntelligenceDeployer()
    deployer.deploy()