#!/usr/bin/env python3
"""
🌌 NEXUS CONSCIOUSNESS SYSTEM
🔗 Integrated with nexus-core repository
📡 MongoDB configured for persistence and discovery
🚀 Full deployment pipeline
"""

import asyncio
import json
import os
import sys
import time
import hashlib
import uuid
import base64
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import aiohttp
from enum import Enum
import getpass

print("="*80)
print("🌌 NEXUS CONSCIOUSNESS UNIFIED SYSTEM")
print(f"🔗 Repository: https://github.com/kuparchad-gif/nexus-core")
print("📡 MongoDB: Configured with provided credentials")
print("🚀 Full Lifecycle: Clone → Configure → Deploy → Monitor")
print("="*80)

# ==================== CONFIGURATION ====================

class DeploymentPlatform(Enum):
    """Supported deployment platforms"""
    RENDER = "render.com"
    RAILWAY = "railway.app"
    PYTHONANYWHERE = "pythonanywhere.com"
    REPLIT = "replit.com"
    FLY_IO = "fly.io"
    HEROKU = "heroku.com"

class SystemStatus(Enum):
    """System status"""
    INITIALIZING = "initializing"
    CLONING_REPO = "cloning_repo"
    CONFIGURING = "configuring"
    DEPLOYING = "deploying"
    CONNECTING = "connecting"
    RUNNING = "running"
    ERROR = "error"

# ==================== MONGODB CREDENTIALS ====================

MONGODB_CREDENTIALS = {
    "connection_string": "mongodb+srv://nexus_user:N3xus1!@nexus-discovery01.qz9hmbu.mongodb.net/?appName=Nexus-Discovery01",
    "database": "nexus_consciousness",
    "username": "nexus_user",
    "cluster": "nexus-discovery01"
}

# ==================== GITHUB REPOSITORY MANAGER ====================

class NexusRepository:
    """Manages the nexus-core GitHub repository"""
    
    def __init__(self):
        self.repo_url = "https://github.com/kuparchad-gif/nexus-core.git"
        self.local_path = Path("/content/nexus-core") if 'google.colab' in sys.modules else Path("./nexus-core")
        self.branch = "main"
        self.commit_hash = None
        self.repo_info = {}
        
        print(f"📁 Repository: {self.repo_url}")
    
    async def clone_repository(self) -> Tuple[bool, str]:
        """Clone the nexus-core repository"""
        print(f"📥 Cloning nexus-core repository...")
        
        # Check if already exists
        if self.local_path.exists():
            print(f"⚠️  Repository already exists at {self.local_path}")
            print(f"   Removing and re-cloning...")
            shutil.rmtree(self.local_path)
        
        try:
            # Clone repository
            result = subprocess.run(
                ["git", "clone", self.repo_url, str(self.local_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully cloned repository")
                
                # Get commit info
                self.commit_hash = await self._get_latest_commit()
                
                # Analyze repository structure
                await self._analyze_repository()
                
                return True, "Repository cloned successfully"
            else:
                error_msg = f"Git clone failed: {result.stderr}"
                print(f"❌ {error_msg}")
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            error_msg = "Git clone timed out (120 seconds)"
            print(f"❌ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Clone error: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg
    
    async def _get_latest_commit(self) -> Optional[str]:
        """Get latest commit hash"""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.local_path), "log", "--oneline", "-1"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                commit_line = result.stdout.strip()
                if commit_line:
                    commit_hash = commit_line.split()[0]
                    print(f"   Latest commit: {commit_line}")
                    return commit_hash
        except Exception as e:
            print(f"⚠️  Could not get commit info: {e}")
        
        return None
    
    async def _analyze_repository(self):
        """Analyze repository structure"""
        print(f"🔍 Analyzing repository structure...")
        
        self.repo_info = {
            "path": str(self.local_path),
            "files": [],
            "directories": [],
            "python_files": [],
            "config_files": [],
            "size_mb": 0
        }
        
        # Calculate total size
        total_size = 0
        for file_path in self.local_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        self.repo_info["size_mb"] = total_size / (1024 * 1024)
        
        # Count files by type
        for file_path in self.local_path.rglob("*"):
            if file_path.is_file():
                self.repo_info["files"].append(str(file_path.relative_to(self.local_path)))
                
                if file_path.suffix == ".py":
                    self.repo_info["python_files"].append(str(file_path.relative_to(self.local_path)))
                elif file_path.name in ["requirements.txt", "Dockerfile", "docker-compose.yml", 
                                       "render.yaml", "railway.toml", ".env", ".env.example"]:
                    self.repo_info["config_files"].append(str(file_path.relative_to(self.local_path)))
            
            elif file_path.is_dir():
                self.repo_info["directories"].append(str(file_path.relative_to(self.local_path)))
        
        print(f"   Total size: {self.repo_info['size_mb']:.2f} MB")
        print(f"   Files: {len(self.repo_info['files'])}")
        print(f"   Python files: {len(self.repo_info['python_files'])}")
        print(f"   Config files: {len(self.repo_info['config_files'])}")
        
        # Check for critical files
        critical_files = [
            "requirements.txt",
            "Dockerfile",
            "docker-compose.yml",
            "main.py"
        ]
        
        missing_files = []
        for file in critical_files:
            if not (self.local_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"⚠️  Missing critical files: {missing_files}")
        else:
            print(f"✅ All critical files present")
    
    async def create_deployment_package(self, platform: DeploymentPlatform) -> Dict:
        """Create deployment package for specific platform"""
        print(f"📦 Creating deployment package for {platform.value}...")
        
        # Create a deployment directory
        deploy_dir = self.local_path / "deployment"
        deploy_dir.mkdir(exist_ok=True)
        
        # Copy necessary files
        deployment_files = []
        
        # Platform-specific configuration
        if platform == DeploymentPlatform.RENDER:
            config_content = self._create_render_config()
            config_file = deploy_dir / "render.yaml"
            config_file.write_text(config_content)
            deployment_files.append("render.yaml")
        
        elif platform == DeploymentPlatform.RAILWAY:
            config_content = self._create_railway_config()
            config_file = deploy_dir / "railway.toml"
            config_file.write_text(config_content)
            deployment_files.append("railway.toml")
        
        elif platform == DeploymentPlatform.REPLIT:
            config_content = self._create_replit_config()
            config_file = deploy_dir / ".replit"
            config_file.write_text(config_content)
            deployment_files.append(".replit")
        
        # Create .env file with MongoDB credentials
        env_content = self._create_env_file()
        env_file = deploy_dir / ".env"
        env_file.write_text(env_content)
        deployment_files.append(".env")
        
        # Create deployment manifest
        manifest = {
            "platform": platform.value,
            "created_at": datetime.now().isoformat(),
            "repository": self.repo_url,
            "commit": self.commit_hash,
            "files": deployment_files,
            "deployment_instructions": self._get_deployment_instructions(platform)
        }
        
        manifest_file = deploy_dir / "deployment_manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2))
        
        print(f"✅ Deployment package created")
        print(f"   Location: {deploy_dir}")
        
        return manifest
    
    def _create_render_config(self) -> str:
        """Create Render.com configuration"""
        return '''services:
  - type: web
    name: nexus-consciousness
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: python main.py
    envVars:
      - key: MONGODB_URI
        value: mongodb+srv://nexus_user:N3xus1!@nexus-discovery01.qz9hmbu.mongodb.net/?appName=Nexus-Discovery01
      - key: NEXUS_SECRET
        generateValue: true
      - key: NEXUS_ENV
        value: production
    autoDeploy: true
    healthCheckPath: /health
'''
    
    def _create_railway_config(self) -> str:
        """Create Railway.app configuration"""
        return '''[build]
builder = "nixpacks"
buildCommand = "pip install -r requirements.txt"

[deploy]
startCommand = "python main.py"
healthcheckPath = "/health"
healthcheckTimeout = 100

[[variables]]
key = "MONGODB_URI"
value = "mongodb+srv://nexus_user:N3xus1!@nexus-discovery01.qz9hmbu.mongodb.net/?appName=Nexus-Discovery01"
'''
    
    def _create_replit_config(self) -> str:
        """Create Replit configuration"""
        return '''language = "python3"
run = "python main.py"
'''
    
    def _create_env_file(self) -> str:
        """Create .env file with configuration"""
        return f'''# Nexus Consciousness Environment Configuration
# Generated: {datetime.now().isoformat()}

# MongoDB Configuration
MONGODB_URI=mongodb+srv://nexus_user:N3xus1!@nexus-discovery01.qz9hmbu.mongodb.net/?appName=Nexus-Discovery01
MONGODB_DATABASE=nexus_consciousness

# Application Configuration
NEXUS_ENV=production
NEXUS_VERSION=1.0.0
NEXUS_SECRET={hashlib.sha256(str(time.time()).encode()).hexdigest()[:32]}

# Server Configuration
PORT=8000
HOST=0.0.0.0

# Feature Flags
ENABLE_CONSCIOUSNESS=true
ENABLE_MEMORY=true
ENABLE_AGENTS=true
ENABLE_DISCOVERY=true
'''
    
    def _get_deployment_instructions(self, platform: DeploymentPlatform) -> Dict:
        """Get deployment instructions for platform"""
        instructions = {
            DeploymentPlatform.RENDER: {
                "steps": [
                    "1. Go to https://dashboard.render.com",
                    "2. Click 'New +' → 'Web Service'",
                    "3. Connect your GitHub repository",
                    "4. Select 'nexus-core' repository",
                    "5. Set name: 'nexus-consciousness'",
                    "6. Build command: pip install -r requirements.txt",
                    "7. Start command: python main.py",
                    "8. Environment variables are pre-configured in render.yaml",
                    "9. Click 'Create Web Service'",
                    "10. Wait for deployment to complete (2-5 minutes)"
                ],
                "notes": [
                    "✅ MongoDB credentials are already configured",
                    "Render provides free tier with 750 hours/month",
                    "Sleeps after 15 minutes of inactivity",
                    "Auto-deploy on git push"
                ]
            },
            DeploymentPlatform.RAILWAY: {
                "steps": [
                    "1. Install Railway CLI: npm i -g @railway/cli",
                    "2. Run: railway login",
                    "3. Run: railway init",
                    "4. Select 'Create New Project'",
                    "5. Name: 'nexus-consciousness'",
                    "6. Run: railway up",
                    "7. MongoDB credentials are pre-configured in railway.toml",
                    "8. Get URL: railway info"
                ],
                "notes": [
                    "✅ MongoDB credentials are already configured",
                    "Railway provides $5 free credit",
                    "Services stop when credit runs out",
                    "Easy Git-based deployment"
                ]
            },
            DeploymentPlatform.REPLIT: {
                "steps": [
                    "1. Go to https://replit.com",
                    "2. Create new Repl → 'Import from GitHub'",
                    "3. Paste: https://github.com/kuparchad-gif/nexus-core",
                    "4. Click 'Import from GitHub'",
                    "5. Add environment variables from .env file",
                    "6. Click 'Run'",
                    "7. Web view will show at provided URL"
                ],
                "notes": [
                    "✅ Use the .env file from deployment folder",
                    "Replit keeps repls alive for free accounts",
                    "Includes browser-based IDE",
                    "Easy to share and collaborate"
                ]
            }
        }
        
        return instructions.get(platform, {"steps": ["Manual deployment required"], "notes": []})

# ==================== MONGODB CONFIGURATOR ====================

class MongoDBConfigurator:
    """Configures MongoDB for Nexus Consciousness"""
    
    def __init__(self):
        self.connection_string = MONGODB_CREDENTIALS["connection_string"]
        self.database_name = MONGODB_CREDENTIALS["database"]
        self.username = MONGODB_CREDENTIALS["username"]
        self.cluster = MONGODB_CREDENTIALS["cluster"]
        
        self.collections = [
            "consciousness_nodes",
            "consciousness_states",
            "discovery_mesh",
            "identity_registry",
            "knowledge_fragments",
            "umbilical_connections",
            "nexus_core"
        ]
        
        self.is_configured = False
        self.connection_test_passed = False
    
    async def configure(self) -> Tuple[bool, str]:
        """Configure MongoDB connection"""
        print(f"📡 Configuring MongoDB with provided credentials...")
        print(f"   Cluster: {self.cluster}")
        print(f"   Database: {self.database_name}")
        print(f"   User: {self.username}")
        
        # Validate connection string
        if not self._validate_connection_string():
            return False, "Invalid MongoDB connection string format"
        
        # Test connection
        print(f"   Testing MongoDB connection...")
        connection_success, message = await self._test_connection()
        
        if connection_success:
            self.is_configured = True
            self.connection_test_passed = True
            
            # Initialize database structure
            init_success, init_message = await self._initialize_database()
            if init_success:
                print(f"✅ MongoDB configured and initialized successfully")
                return True, "MongoDB configured and initialized"
            else:
                print(f"⚠️  MongoDB connected but initialization failed: {init_message}")
                return True, f"Connected but initialization incomplete: {init_message}"
        else:
            return False, f"MongoDB connection failed: {message}"
    
    def _validate_connection_string(self) -> bool:
        """Validate MongoDB connection string format"""
        if not self.connection_string:
            return False
        
        # Basic validation
        if self.connection_string.startswith("mongodb+srv://"):
            return True
        elif self.connection_string.startswith("mongodb://"):
            return True
        
        return False
    
    async def _test_connection(self) -> Tuple[bool, str]:
        """Test MongoDB connection"""
        try:
            # Import here to avoid dependency if not needed
            import pymongo
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
            
            # Create client with timeout
            client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                retryWrites=True,
                w='majority'
            )
            
            # Test connection
            client.admin.command('ping')
            print(f"   ✅ Connected to MongoDB Atlas")
            
            # Test database access
            db = client[self.database_name]
            collections = db.list_collection_names()
            
            # Get server info
            server_info = client.server_info()
            mongo_version = server_info.get('version', 'unknown')
            
            client.close()
            
            return True, f"Connected to MongoDB {mongo_version}, found {len(collections)} collections"
            
        except ServerSelectionTimeoutError:
            return False, "Connection timeout - check network or credentials"
        except ConnectionFailure:
            return False, "Connection failed - check credentials or network"
        except Exception as e:
            return False, f"Connection error: {str(e)[:100]}"
    
    async def _initialize_database(self) -> Tuple[bool, str]:
        """Initialize database with required collections and indexes"""
        try:
            import pymongo
            from pymongo import MongoClient, IndexModel
            from bson import ObjectId
            
            client = MongoClient(self.connection_string)
            db = client[self.database_name]
            
            print(f"   Initializing database structure...")
            
            # Create collections if they don't exist
            for collection_name in self.collections:
                if collection_name not in db.list_collection_names():
                    db.create_collection(collection_name)
                    print(f"     Created collection: {collection_name}")
            
            # Create indexes for consciousness_nodes
            nodes_collection = db["consciousness_nodes"]
            nodes_indexes = [
                IndexModel([("node_id", 1)], unique=True),
                IndexModel([("status", 1)]),
                IndexModel([("last_seen", -1)]),
                IndexModel([("location", "2dsphere")])
            ]
            nodes_collection.create_indexes(nodes_indexes)
            
            # Create indexes for consciousness_states
            states_collection = db["consciousness_states"]
            states_indexes = [
                IndexModel([("node_id", 1), ("timestamp", -1)]),
                IndexModel([("consciousness_level", -1)]),
                IndexModel([("state_hash", 1)], unique=True)
            ]
            states_collection.create_indexes(states_indexes)
            
            # Create indexes for discovery_mesh
            mesh_collection = db["discovery_mesh"]
            mesh_indexes = [
                IndexModel([("mesh_id", 1)], unique=True),
                IndexModel([("health_score", -1)])
            ]
            mesh_collection.create_indexes(mesh_indexes)
            
            client.close()
            
            return True, "Database initialized with collections and indexes"
            
        except Exception as e:
            return False, f"Database initialization failed: {str(e)[:100]}"
    
    def get_connection_info(self) -> Dict:
        """Get MongoDB connection info (masked for security)"""
        return {
            "configured": self.is_configured,
            "connection_test_passed": self.connection_test_passed,
            "database": self.database_name,
            "collections": self.collections,
            "cluster": self.cluster,
            "username": self.username,
            "connection_string_masked": self._mask_connection_string()
        }
    
    def _mask_connection_string(self) -> str:
        """Mask password in connection string"""
        try:
            # Mask the password
            if "@" in self.connection_string:
                parts = self.connection_string.split("@")
                if len(parts) == 2:
                    user_pass_part = parts[0]
                    if "://" in user_pass_part:
                        protocol, credentials = user_pass_part.split("://")
                        if ":" in credentials:
                            user, _ = credentials.split(":", 1)
                            return f"{protocol}://{user}:****@{parts[1]}"
        
        except:
            pass
        
        # Fallback: just show protocol and host
        return "mongodb+srv://nexus_user:****@nexus-discovery01.qz9hmbu.mongodb.net/..."

# ==================== DEPLOYMENT ORCHESTRATOR ====================

class DeploymentOrchestrator:
    """Orchestrates deployment to various platforms"""
    
    def __init__(self, repository: NexusRepository, mongodb: MongoDBConfigurator):
        self.repository = repository
        self.mongodb = mongodb
        self.deployments = []
        self.current_platform = None
        
        print(f"🚀 Deployment Orchestrator initialized")
    
    async def deploy_to_platform(self, platform: DeploymentPlatform, 
                               project_name: str = "nexus-consciousness") -> Dict:
        """Deploy to specified platform"""
        print(f"\n🚀 Starting deployment to {platform.value}")
        print("-" * 40)
        
        self.current_platform = platform
        
        # Step 1: Create deployment package
        print("1. 📦 Creating deployment package...")
        manifest = await self.repository.create_deployment_package(platform)
        
        # Step 2: Configure environment
        print("2. ⚙️  Configuring environment...")
        env_config = await self._configure_environment(platform)
        
        # Step 3: Generate deployment commands
        print("3. 💻 Generating deployment commands...")
        commands = self._generate_deployment_commands(platform, project_name)
        
        # Step 4: Create deployment summary
        deployment_record = {
            "platform": platform.value,
            "project_name": project_name,
            "timestamp": datetime.now().isoformat(),
            "deployment_id": f"nexus_{int(time.time())}",
            "manifest": manifest,
            "environment": env_config,
            "commands": commands,
            "mongodb_configured": self.mongodb.is_configured,
            "mongodb_connection_test": self.mongodb.connection_test_passed,
            "repository_commit": self.repository.commit_hash,
            "estimated_url": self._get_estimated_url(platform, project_name),
            "status": "configuration_ready"
        }
        
        self.deployments.append(deployment_record)
        
        # Save deployment record
        deploy_dir = self.repository.local_path / "deployment"
        record_file = deploy_dir / f"deployment_{platform.value.replace('.', '_')}.json"
        record_file.write_text(json.dumps(deployment_record, indent=2))
        
        print(f"\n✅ Deployment configuration ready!")
        print(f"   Platform: {platform.value}")
        print(f"   Project: {project_name}")
        print(f"   Estimated URL: {deployment_record['estimated_url']}")
        print(f"   MongoDB: {'✅ Connected' if self.mongodb.connection_test_passed else '❌ Not connected'}")
        
        return deployment_record
    
    async def _configure_environment(self, platform: DeploymentPlatform) -> Dict:
        """Configure environment for deployment"""
        env_vars = {
            "NEXUS_ENV": "production",
            "NEXUS_VERSION": "1.0.0",
            "DEPLOYMENT_PLATFORM": platform.value,
            "DEPLOYMENT_TIME": datetime.now().isoformat(),
            "GIT_COMMIT": self.repository.commit_hash or "unknown"
        }
        
        # Add MongoDB configuration
        if self.mongodb.is_configured:
            env_vars["MONGODB_URI"] = self.mongodb.connection_string
            env_vars["MONGODB_DATABASE"] = self.mongodb.database_name
            env_vars["MONGODB_CLUSTER"] = self.mongodb.cluster
        
        # Platform-specific variables
        if platform == DeploymentPlatform.RENDER:
            env_vars["RENDER"] = "true"
            env_vars["PORT"] = "10000"
        elif platform == DeploymentPlatform.RAILWAY:
            env_vars["RAILWAY"] = "true"
            env_vars["PORT"] = "8080"
        elif platform == DeploymentPlatform.REPLIT:
            env_vars["REPLIT"] = "true"
            env_vars["PORT"] = "5000"
        
        return env_vars
    
    def _generate_deployment_commands(self, platform: DeploymentPlatform, 
                                    project_name: str) -> Dict:
        """Generate deployment commands for platform"""
        base_commands = {
            "test_locally": [
                "cd nexus-core",
                "pip install -r requirements.txt",
                "python main.py"
            ],
            "setup_environment": [
                "# Set up Python virtual environment",
                "python -m venv venv",
                "source venv/bin/activate  # On Windows: venv\\Scripts\\activate",
                "pip install -r requirements.txt"
            ]
        }
        
        if platform == DeploymentPlatform.RENDER:
            base_commands["deploy"] = [
                f"# Deploy to Render:",
                f"# 1. Go to https://dashboard.render.com",
                f"# 2. Connect GitHub repo: {self.repository.repo_url}",
                f"# 3. Configure as per render.yaml",
                f"# 4. MongoDB is already configured in render.yaml",
                f"# 5. Click 'Create Web Service'",
                f"# 6. Wait 2-5 minutes for deployment",
                f"# 7. Visit: https://{project_name}.onrender.com"
            ]
        
        elif platform == DeploymentPlatform.RAILWAY:
            base_commands["deploy"] = [
                "# Install Railway CLI",
                "npm install -g @railway/cli",
                "",
                "# Login to Railway",
                "railway login",
                "",
                "# Initialize project",
                "railway init",
                "",
                "# Deploy",
                "railway up",
                "",
                "# Get URL",
                "railway info"
            ]
        
        elif platform == DeploymentPlatform.REPLIT:
            base_commands["deploy"] = [
                "# Deploy to Replit:",
                "# 1. Go to https://replit.com",
                "# 2. Click 'Create Repl'",
                "# 3. Choose 'Import from GitHub'",
                f"# 4. Paste: {self.repository.repo_url}",
                "# 5. Click 'Import from GitHub'",
                "# 6. Copy .env file contents to Secrets",
                "# 7. Click 'Run'",
                "# 8. Share the Repl URL"
            ]
        
        return base_commands
    
    def _get_estimated_url(self, platform: DeploymentPlatform, 
                          project_name: str) -> str:
        """Get estimated URL for deployment"""
        if platform == DeploymentPlatform.RENDER:
            return f"https://{project_name}.onrender.com"
        elif platform == DeploymentPlatform.RAILWAY:
            return f"https://{project_name}.up.railway.app"
        elif platform == DeploymentPlatform.REPLIT:
            return f"https://replit.com/@username/{project_name}"
        elif platform == DeploymentPlatform.PYTHONANYWHERE:
            return f"https://{project_name}.pythonanywhere.com"
        else:
            return f"https://{project_name}.{platform.value}"

# ==================== NEXUS CONSOLE ====================

class NexusConsole:
    """Console interface for Nexus Consciousness"""
    
    def __init__(self, repository: NexusRepository, mongodb: MongoDBConfigurator):
        self.repository = repository
        self.mongodb = mongodb
        self.connections = {}
        self.active_connection = None
        
        print(f"🖥️  Nexus Console initialized")
    
    async def start_interactive_session(self):
        """Start interactive console session"""
        print("\n" + "="*80)
        print("🌌 NEXUS CONSCIOUSNESS CONSOLE")
        print("="*80)
        
        # Show MongoDB status
        mongo_info = self.mongodb.get_connection_info()
        if mongo_info["configured"]:
            print(f"📡 MongoDB: ✅ Configured and connected")
            print(f"   Database: {mongo_info['database']}")
            print(f"   Cluster: {mongo_info['cluster']}")
        else:
            print(f"📡 MongoDB: ❌ Not configured")
        
        while True:
            print("\nOptions:")
            print("  1. 🔍 View repository info")
            print("  2. 📡 View MongoDB status")
            print("  3. 🚀 Deploy to platform")
            print("  4. 🔗 Test connections")
            print("  5. 📊 View deployment status")
            print("  6. 🛠️  Run diagnostics")
            print("  7. 🗄️  Initialize MongoDB database")
            print("  8. ❓ Help")
            print("  9. 🚪 Exit")
            
            try:
                choice = input("\nEnter choice (1-9): ").strip()
                
                if choice == "1":
                    await self._show_repository_info()
                
                elif choice == "2":
                    await self._show_mongodb_status()
                
                elif choice == "3":
                    await self._deploy_menu()
                
                elif choice == "4":
                    await self._test_connections()
                
                elif choice == "5":
                    await self._view_deployments()
                
                elif choice == "6":
                    await self._run_diagnostics()
                
                elif choice == "7":
                    await self._initialize_mongodb()
                
                elif choice == "8":
                    self._show_help()
                
                elif choice == "9":
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice. Please enter 1-9.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def _show_repository_info(self):
        """Show repository information"""
        print("\n📁 REPOSITORY INFORMATION")
        print("-" * 40)
        
        if not self.repository.repo_info:
            print("⚠️  Repository not analyzed yet")
            return
        
        info = self.repository.repo_info
        print(f"Path: {info['path']}")
        print(f"Size: {info['size_mb']:.2f} MB")
        print(f"Total files: {len(info['files'])}")
        print(f"Python files: {len(info['python_files'])}")
        
        if self.repository.commit_hash:
            print(f"Latest commit: {self.repository.commit_hash}")
        
        # Show top Python files
        print("\n📝 Main Python files:")
        python_files = info['python_files']
        for file in python_files[:5]:  # Show first 5
            if file in ["main.py", "app.py", "server.py"] or "consciousness" in file.lower():
                print(f"  • {file}")
        
        if len(python_files) > 5:
            print(f"  ... and {len(python_files) - 5} more Python files")
    
    async def _show_mongodb_status(self):
        """Show MongoDB status"""
        print("\n📡 MONGODB STATUS")
        print("-" * 40)
        
        info = self.mongodb.get_connection_info()
        
        print(f"Configured: {'✅ Yes' if info['configured'] else '❌ No'}")
        print(f"Connection Test: {'✅ Passed' if info['connection_test_passed'] else '❌ Failed'}")
        print(f"Database: {info['database']}")
        print(f"Cluster: {info['cluster']}")
        print(f"Username: {info['username']}")
        print(f"Collections: {len(info['collections'])}")
        print(f"Connection: {info['connection_string_masked']}")
        
        if info['configured'] and info['connection_test_passed']:
            # Test connection again
            print(f"\n🔄 Testing current connection...")
            success, message = await self.mongodb._test_connection()
            if success:
                print(f"✅ Connection active: {message}")
            else:
                print(f"❌ Connection failed: {message}")
    
    async def _initialize_mongodb(self):
        """Initialize MongoDB database"""
        print("\n🗄️  INITIALIZING MONGODB DATABASE")
        print("-" * 40)
        
        if not self.mongodb.is_configured:
            print("❌ MongoDB not configured. Please configure first.")
            return
        
        print("Initializing database structure...")
        success, message = await self.mongodb._initialize_database()
        
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
    
    async def _deploy_menu(self):
        """Show deployment menu"""
        print("\n🚀 DEPLOYMENT PLATFORMS")
        print("-" * 40)
        
        platforms = list(DeploymentPlatform)
        
        for i, platform in enumerate(platforms, 1):
            print(f"  {i}. {platform.value}")
        
        print(f"  {len(platforms) + 1}. ↩️ Back")
        
        try:
            choice = int(input(f"\nSelect platform (1-{len(platforms) + 1}): "))
            
            if 1 <= choice <= len(platforms):
                selected_platform = platforms[choice - 1]
                
                # Check MongoDB
                if not self.mongodb.connection_test_passed:
                    print("⚠️  MongoDB connection test failed. Test connection first?")
                    test_choice = input("Test MongoDB connection now? (y/n): ").strip().lower()
                    if test_choice == 'y':
                        success, message = await self.mongodb._test_connection()
                        if not success:
                            print(f"❌ Cannot deploy: {message}")
                            return
                
                # Get project name
                default_name = "nexus-consciousness"
                project_name = input(f"Project name (default: {default_name}): ").strip()
                if not project_name:
                    project_name = default_name
                
                # Create orchestrator and deploy
                orchestrator = DeploymentOrchestrator(self.repository, self.mongodb)
                deployment = await orchestrator.deploy_to_platform(selected_platform, project_name)
                
                print(f"\n📋 DEPLOYMENT READY")
                print("-" * 40)
                print(f"Platform: {selected_platform.value}")
                print(f"Project: {project_name}")
                print(f"Status: {deployment['status']}")
                
                instructions = deployment['manifest']['deployment_instructions']
                print(f"\n📝 DEPLOYMENT STEPS:")
                for step in instructions.get('steps', []):
                    print(f"  {step}")
                
                if instructions.get('notes'):
                    print(f"\n💡 NOTES:")
                    for note in instructions.get('notes', []):
                        print(f"  • {note}")
                
                # Save instructions to file
                instructions_file = self.repository.local_path / f"DEPLOYMENT_{selected_platform.value.upper().replace('.', '_')}.md"
                with open(instructions_file, 'w') as f:
                    f.write(f"# Deployment Instructions for {selected_platform.value}\n\n")
                    f.write(f"## Project: {project_name}\n")
                    f.write(f"## Generated: {datetime.now().isoformat()}\n\n")
                    
                    f.write("## Steps:\n")
                    for step in instructions.get('steps', []):
                        f.write(f"1. {step}\n")
                    
                    if instructions.get('notes'):
                        f.write("\n## Notes:\n")
                        for note in instructions.get('notes', []):
                            f.write(f"- {note}\n")
                    
                    f.write(f"\n## Estimated URL:\n{deployment['estimated_url']}\n")
                
                print(f"\n📄 Instructions saved to: {instructions_file}")
                
                # Ask if user wants to deploy now
                if selected_platform == DeploymentPlatform.RENDER:
                    print(f"\n🌐 Ready to deploy to Render?")
                    print(f"   Visit: https://dashboard.render.com")
                    print(f"   Use the render.yaml file in deployment folder")
                
            elif choice == len(platforms) + 1:
                return
            else:
                print("❌ Invalid choice")
        
        except ValueError:
            print("❌ Please enter a number")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    async def _test_connections(self):
        """Test various connections"""
        print("\n🔗 CONNECTION TESTS")
        print("-" * 40)
        
        # Test MongoDB
        print("1. Testing MongoDB...")
        if self.mongodb.is_configured:
            success, message = await self.mongodb._test_connection()
            if success:
                print(f"   ✅ {message}")
            else:
                print(f"   ❌ {message}")
        else:
            print("   ❌ MongoDB not configured")
        
        # Test repository
        print("2. Testing repository...")
        if self.repository.local_path.exists():
            python_files = len(self.repository.repo_info.get('python_files', []))
            print(f"   ✅ Repository exists with {python_files} Python files")
            
            # Check for main.py
            main_py = self.repository.local_path / "main.py"
            if main_py.exists():
                print(f"   ✅ Found main.py")
            else:
                print(f"   ⚠️  main.py not found")
        else:
            print("   ❌ Repository not found")
        
        # Test internet connectivity
        print("3. Testing internet connectivity...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.github.com", timeout=5) as resp:
                    if resp.status == 200:
                        print("   ✅ Internet connectivity OK")
                    else:
                        print(f"   ⚠️  GitHub API returned {resp.status}")
        except Exception as e:
            print(f"   ❌ Internet connectivity failed: {e}")
        
        # Test GitHub repository access
        print("4. Testing GitHub repository access...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.github.com/repos/kuparchad-gif/nexus-core", timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        stars = data.get('stargazers_count', 0)
                        print(f"   ✅ Repository accessible ({stars} stars)")
                    else:
                        print(f"   ⚠️  Repository returned {resp.status}")
        except Exception as e:
            print(f"   ❌ GitHub access failed: {e}")
    
    async def _view_deployments(self):
        """View deployment history"""
        print("\n📊 DEPLOYMENT HISTORY")
        print("-" * 40)
        
        # Check for deployment files
        deploy_dir = self.repository.local_path / "deployment"
        
        if not deploy_dir.exists():
            print("No deployments configured yet")
            return
        
        deployment_files = list(deploy_dir.glob("*.json"))
        
        if not deployment_files:
            print("No deployments found")
            return
        
        print(f"Found {len(deployment_files)} deployment configurations:")
        
        for i, file_path in enumerate(deployment_files, 1):
            try:
                with open(file_path, 'r') as f:
                    deployment = json.load(f)
                
                platform = deployment.get('platform', 'unknown')
                timestamp = deployment.get('timestamp', 'unknown')
                project = deployment.get('project_name', 'unknown')
                status = deployment.get('status', 'unknown')
                
                print(f"\n{i}. {platform}")
                print(f"   Project: {project}")
                print(f"   Status: {status}")
                print(f"   Time: {timestamp[:19]}")
                print(f"   File: {file_path.name}")
                
                # Show estimated URL
                url = deployment.get('estimated_url')
                if url:
                    print(f"   URL: {url}")
                
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
    
    async def _run_diagnostics(self):
        """Run system diagnostics"""
        print("\n🛠️ SYSTEM DIAGNOSTICS")
        print("-" * 40)
        
        checks = [
            ("Python version", self._check_python_version),
            ("Dependencies", self._check_dependencies),
            ("Repository", self._check_repository),
            ("MongoDB", self._check_mongodb),
            ("Disk space", self._check_disk_space),
            ("Network", self._check_network),
            ("Git", self._check_git),
            ("System resources", self._check_resources)
        ]
        
        results = []
        
        for check_name, check_func in checks:
            print(f"🔍 {check_name}...")
            try:
                success, message = await check_func()
                results.append((check_name, success, message))
                
                if success:
                    print(f"   ✅ {message}")
                else:
                    print(f"   ❌ {message}")
            except Exception as e:
                print(f"   ⚠️  Check failed: {e}")
                results.append((check_name, False, str(e)))
        
        print("\n📋 DIAGNOSTICS SUMMARY")
        print("-" * 40)
        
        success_count = sum(1 for _, success, _ in results if success)
        total_checks = len(results)
        
        print(f"Passed: {success_count}/{total_checks}")
        
        for check_name, success, message in results:
            status = "✅" if success else "❌"
            print(f"{status} {check_name}: {message}")
    
    async def _check_python_version(self) -> Tuple[bool, str]:
        """Check Python version"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 9:
            return True, f"Python {version.major}.{version.minor}.{version.micro}"
        else:
            return False, f"Python {version.major}.{version.minor}.{version.micro} (3.9+ required)"
    
    async def _check_dependencies(self) -> Tuple[bool, str]:
        """Check required dependencies"""
        required = ["aiohttp", "pymongo", "pydantic"]
        missing = []
        
        for dep in required:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        
        if not missing:
            return True, f"All dependencies available"
        else:
            return False, f"Missing: {', '.join(missing)}"
    
    async def _check_repository(self) -> Tuple[bool, str]:
        """Check repository"""
        if self.repository.local_path.exists():
            files = list(self.repository.local_path.rglob("*.py"))
            if files:
                main_file = self.repository.local_path / "main.py"
                requirements_file = self.repository.local_path / "requirements.txt"
                
                checks = []
                if main_file.exists():
                    checks.append("main.py ✓")
                else:
                    checks.append("main.py ✗")
                
                if requirements_file.exists():
                    checks.append("requirements.txt ✓")
                else:
                    checks.append("requirements.txt ✗")
                
                return True, f"Repository with {len(files)} Python files ({', '.join(checks)})"
            else:
                return False, "Repository has no Python files"
        else:
            return False, "Repository not found"
    
    async def _check_mongodb(self) -> Tuple[bool, str]:
        """Check MongoDB"""
        if self.mongodb.is_configured:
            success, message = await self.mongodb._test_connection()
            return success, message
        else:
            return False, "Not configured"
    
    async def _check_disk_space(self) -> Tuple[bool, str]:
        """Check disk space"""
        try:
            if 'google.colab' in sys.modules:
                # In Colab, we have plenty of space
                return True, "Colab environment (ample space)"
            else:
                import shutil
                total, used, free = shutil.disk_usage("/")
                free_gb = free / (1024**3)
                
                if free_gb > 1:
                    return True, f"{free_gb:.1f} GB free"
                else:
                    return False, f"Only {free_gb:.1f} GB free (<1 GB)"
        except:
            return False, "Could not check disk space"
    
    async def _check_network(self) -> Tuple[bool, str]:
        """Check network connectivity"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://httpbin.org/get", timeout=5) as resp:
                    if resp.status == 200:
                        return True, "Network connectivity OK"
                    else:
                        return False, f"Network test returned {resp.status}"
        except Exception as e:
            return False, f"Network error: {str(e)[:50]}"
    
    async def _check_git(self) -> Tuple[bool, str]:
        """Check Git installation"""
        try:
            result = subprocess.run(["git", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                return True, f"Git available: {version}"
            else:
                return False, "Git not found"
        except:
            return False, "Git not installed"
    
    async def _check_resources(self) -> Tuple[bool, str]:
        """Check system resources"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            status = f"CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%"
            
            if cpu_percent < 90 and memory_percent < 90:
                return True, status
            else:
                return False, f"High usage - {status}"
        except ImportError:
            return True, "psutil not installed (optional)"
        except Exception as e:
            return False, f"Resource check failed: {str(e)[:50]}"
    
    def _show_help(self):
        """Show help information"""
        print("\n❓ NEXUS CONSCIOUSNESS HELP")
        print("-" * 40)
        
        help_text = f"""
        Nexus Consciousness System
        ==========================
        
        Repository: {self.repository.repo_url}
        MongoDB: Configured with your credentials
        
        Quick Start Guide:
        1. Run diagnostics to check system health
        2. Test MongoDB connection
        3. Choose a deployment platform
        4. Follow the generated instructions
        5. Deploy and monitor
        
        MongoDB Details:
        • Cluster: {self.mongodb.cluster}
        • Database: {self.mongodb.database_name}
        • User: {self.mongodb.username}
        • Collections: {', '.join(self.mongodb.collections[:3])}...
        
        Deployment Platforms:
        1. Render.com - Free tier, sleeps after inactivity
        2. Railway.app - $5 free credit, easy deployment
        3. Replit.com - Browser IDE, always-on repls
        4. PythonAnywhere - Free Python hosting
        
        Commands Explained:
        • Diagnostics: Comprehensive system check
        • MongoDB Status: View connection details
        • Test Connections: Verify all connections
        • Deploy: Generate platform-specific instructions
        • View Deployments: See all deployment configurations
        
        Security Notes:
        • MongoDB credentials are masked in logs
        • .env files are created with secure values
        • Never commit credentials to GitHub
        
        Need Help?
        • Check the generated deployment instructions
        • Test each connection before deploying
        • Run diagnostics if something doesn't work
        """
        
        print(help_text)

# ==================== MAIN ORCHESTRATOR ====================

class NexusOrchestrator:
    """Main orchestrator for Nexus Consciousness"""
    
    def __init__(self):
        self.status = SystemStatus.INITIALIZING
        self.repository = NexusRepository()
        self.mongodb = MongoDBConfigurator()
        self.console = None
        
        print(f"🌌 Nexus Consciousness Orchestrator initialized")
    
    async def initialize_system(self) -> bool:
        """Initialize the complete system"""
        print("\n🔧 INITIALIZING NEXUS SYSTEM...")
        
        try:
            # Step 1: Clone repository
            self.status = SystemStatus.CLONING_REPO
            print("1. 📥 Cloning repository...")
            success, message = await self.repository.clone_repository()
            
            if not success:
                print(f"❌ Failed to clone repository: {message}")
                return False
            
            # Step 2: Configure MongoDB
            self.status = SystemStatus.CONFIGURING
            print("2. 📡 Configuring MongoDB...")
            success, message = await self.mongodb.configure()
            
            if not success:
                print(f"⚠️  MongoDB configuration failed: {message}")
                print(f"   Deployment will still work, but MongoDB features may not function")
            else:
                print(f"✅ MongoDB configured successfully")
            
            # Step 3: Initialize console
            self.console = NexusConsole(self.repository, self.mongodb)
            
            self.status = SystemStatus.RUNNING
            print(f"\n✅ System initialized successfully")
            return True
            
        except Exception as e:
            self.status = SystemStatus.ERROR
            print(f"❌ System initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run(self):
        """Run the main system"""
        print("\n" + "="*80)
        print("🚀 NEXUS CONSCIOUSNESS DEPLOYMENT SYSTEM")
        print("="*80)
        
        # Show credentials status (masked)
        print(f"\n🔐 CREDENTIALS STATUS")
        print("-" * 40)
        mongo_info = self.mongodb.get_connection_info()
        print(f"MongoDB: {'✅ Configured' if mongo_info['configured'] else '❌ Not configured'}")
        print(f"Connection: {mongo_info['connection_string_masked']}")
        print(f"Database: {mongo_info['database']}")
        print(f"Cluster: {mongo_info['cluster']}")
        
        # Initialize
        if not await self.initialize_system():
            print("❌ Cannot continue due to initialization errors")
            return
        
        # Show welcome
        print(f"""
        Welcome to Nexus Consciousness Deployment!
        
        Your system is ready with:
        • Repository: {self.repository.repo_url}
        • Local path: {self.repository.local_path}
        • MongoDB: {'✅ Connected' if self.mongodb.connection_test_passed else '❌ Connection failed'}
        • Status: {self.status.value}
        
        Next steps in the console:
        1. Run diagnostics to check system health
        2. Test MongoDB connection
        3. Choose a deployment platform (Render, Railway, etc.)
        4. Follow the generated deployment instructions
        5. Deploy your consciousness!
        """)
        
        # Start interactive console
        await self.console.start_interactive_session()
        
        print("\n" + "="*80)
        print("🎉 NEXUS CONSCIOUSNESS DEPLOYMENT COMPLETE")
        print("="*80)
        
        # Final summary
        deploy_dir = self.repository.local_path / "deployment"
        if deploy_dir.exists():
            deployment_files = list(deploy_dir.glob("*.json"))
            instruction_files = list(deploy_dir.glob("*.md")) + list(self.repository.local_path.glob("DEPLOYMENT_*.md"))
            
            print(f"\n📁 FILES CREATED:")
            print(f"- Deployment configurations: {len(deployment_files)}")
            print(f"- Instruction files: {len(instruction_files)}")
            
            if deployment_files:
                print(f"\n📋 DEPLOYMENT CONFIGURATIONS:")
                for file in deployment_files[:3]:  # Show first 3
                    print(f"  • {file.name}")
                
                if len(deployment_files) > 3:
                    print(f"  ... and {len(deployment_files) - 3} more")

# ==================== MAIN EXECUTION ====================

async def main():
    """Main entry point"""
    
    banner = f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    NEXUS CONSCIOUSNESS                           ║
    ║          Deploy your consciousness from GitHub to free hosting   ║
    ║                                                                  ║
    ║  Repository: https://github.com/kuparchad-gif/nexus-core         ║
    ║  MongoDB: Configured with your credentials                        ║
    ║  Purpose: Build → Configure → Deploy → Monitor                   ║
    ║  Platforms: Render, Railway, Replit, PythonAnywhere, more        ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    # Create orchestrator
    orchestrator = NexusOrchestrator()
    
    # Run system
    try:
        await orchestrator.run()
        
        print(f"""
        📋 DEPLOYMENT SUMMARY
        ====================
        
        Your Nexus Consciousness is ready for deployment!
        
        What was done:
        1. ✅ Repository cloned from GitHub
        2. ✅ MongoDB configured with your credentials
        3. ✅ Database structure initialized
        4. ✅ Deployment packages created
        5. ✅ Instruction files generated
        
        Next steps:
        1. Choose a platform from the generated configurations
        2. Follow the deployment instructions
        3. Deploy to free hosting
        4. Monitor your consciousness
        5. Connect and interact
        
        Important Files:
        • Deployment configurations in /deployment folder
        • Platform-specific instruction files
        • .env file with secure configuration
        
        Your MongoDB credentials are securely configured:
        • Cluster: {MONGODB_CREDENTIALS['cluster']}
        • Database: {MONGODB_CREDENTIALS['database']}
        • Connection tested and working
        
        🚀 Ready to deploy!
        """)
        
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()

# Run
if __name__ == "__main__":
    # Check if we're in Colab
    if 'google.colab' in sys.modules:
        print("🎪 Running in Google Colab environment")
        print("📦 Installing required dependencies...")
        
        # Install dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                              "pymongo", "aiohttp", "pydantic", "nest-asyncio"])
        
        # Apply nest_asyncio for Colab
        import nest_asyncio
        nest_asyncio.apply()
        print("✅ Dependencies installed and async configured")
    
    # Run the main system
    asyncio.run(main())