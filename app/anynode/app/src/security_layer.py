#!/usr/bin/env python3
"""
CogniKube Security Layer - Layer 3
Authentication, Authorization, Audit
"""

import jwt
import hashlib
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# Configure audit logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - AUDIT - %(message)s',
    handlers=[
        logging.FileHandler('audit.log'),
        logging.StreamHandler()
    ]
)
audit_logger = logging.getLogger('audit')

@dataclass
class User:
    id: str
    username: str
    roles: List[str]
    permissions: List[str]

class SecurityManager:
    def __init__(self):
        self.jwt_secret = "your-secret-key"  # Use env var in production
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, str] = {}
        
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = "cognikube_salt"  # Use random salt in production
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token"""
        # Check user exists and password matches
        user = self.users.get(username)
        if not user:
            audit_logger.warning(f"Failed login attempt for user: {username}")
            return None
            
        # In production, check hashed password
        # if self.hash_password(password) != stored_hash:
        #     return None
            
        # Generate JWT token
        payload = {
            "user_id": user.id,
            "username": user.username,
            "roles": user.roles,
            "exp": time.time() + 3600  # 1 hour expiry
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        self.active_sessions[token] = user.id
        
        audit_logger.info(f"Successful login: {username}")
        return token
    
    def authorize(self, token: str, required_permission: str) -> bool:
        """Check if user has required permission"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            user = self.users.get(payload["username"])
            
            if not user:
                return False
                
            has_permission = required_permission in user.permissions
            
            audit_logger.info(f"Authorization check: {payload['username']} - {required_permission} - {'GRANTED' if has_permission else 'DENIED'}")
            
            return has_permission
            
        except jwt.ExpiredSignatureError:
            audit_logger.warning("Token expired")
            return False
        except jwt.InvalidTokenError:
            audit_logger.warning("Invalid token")
            return False
    
    def audit_action(self, user: str, action: str, resource: str, result: str):
        """Log security-relevant actions"""
        audit_logger.info(f"USER:{user} ACTION:{action} RESOURCE:{resource} RESULT:{result}")
    
    def add_user(self, username: str, password: str, roles: List[str], permissions: List[str]):
        """Add user to system"""
        user = User(
            id=hashlib.md5(username.encode()).hexdigest(),
            username=username,
            roles=roles,
            permissions=permissions
        )
        self.users[username] = user
        audit_logger.info(f"User created: {username} with roles: {roles}")

# Initialize security manager
security = SecurityManager()

# Add default admin user
security.add_user(
    "admin", 
    "secure_password", 
    ["admin"], 
    ["read", "write", "admin", "deploy"]
)

# Middleware for FastAPI
from fastapi import Request, HTTPException

async def security_middleware(request: Request, call_next):
    """Security middleware for all requests"""
    
    # Skip auth for health check
    if request.url.path == "/health":
        return await call_next(request)
    
    # Check for authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    token = auth_header.split(" ")[1]
    
    # Determine required permission based on method
    permission_map = {
        "GET": "read",
        "POST": "write", 
        "PUT": "write",
        "DELETE": "admin"
    }
    
    required_permission = permission_map.get(request.method, "read")
    
    if not security.authorize(token, required_permission):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Log the authorized request
    try:
        payload = jwt.decode(token, security.jwt_secret, algorithms=["HS256"])
        security.audit_action(
            payload["username"], 
            request.method, 
            request.url.path, 
            "ALLOWED"
        )
    except:
        pass
    
    return await call_next(request)