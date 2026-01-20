#!/usr/bin/env python3
"""
Authentication Manager for Viren Platinum Edition
Handles user authentication, MFA, and session management
"""

import os
import json
import time
import logging
import hashlib
import secrets
import base64
from typing import Dict, Any, Optional, List, Tuple

# Configure logging
logger = logging.getLogger("AuthManager")

class AuthManager:
    """
    Manages user authentication, MFA, and session management
    """
    
    def __init__(self, auth_dir: str = "auth"):
        """Initialize the authentication manager"""
        self.auth_dir = auth_dir
        self.users_file = os.path.join(auth_dir, "users.json")
        self.sessions_file = os.path.join(auth_dir, "sessions.json")
        self.users = {}
        self.sessions = {}
        
        # Create auth directory if it doesn't exist
        os.makedirs(auth_dir, exist_ok=True)
        
        # Load users and sessions
        self._load_users()
        self._load_sessions()
        
        # Create default user if none exists
        if not self.users:
            self._create_default_user()
    
    def _load_users(self):
        """Load users from file"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
                logger.info(f"Loaded {len(self.users)} users")
            except Exception as e:
                logger.error(f"Error loading users: {e}")
                self.users = {}
    
    def _save_users(self):
        """Save users to file"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
            logger.info("Users saved")
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def _load_sessions(self):
        """Load sessions from file"""
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r') as f:
                    self.sessions = json.load(f)
                
                # Clean expired sessions
                current_time = time.time()
                expired_sessions = []
                for session_id, session in self.sessions.items():
                    if session.get("expires", 0) < current_time:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.sessions[session_id]
                
                logger.info(f"Loaded {len(self.sessions)} active sessions")
            except Exception as e:
                logger.error(f"Error loading sessions: {e}")
                self.sessions = {}
    
    def _save_sessions(self):
        """Save sessions to file"""
        try:
            with open(self.sessions_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def _create_default_user(self):
        """Create default user"""
        default_user = {
            "username": "admin",
            "password_hash": self._hash_password("admin"),
            "salt": self._generate_salt(),
            "mfa_enabled": False,
            "mfa_secret": "",
            "role": "admin",
            "created_at": time.time()
        }
        
        self.users["admin"] = default_user
        self._save_users()
        logger.info("Created default user: admin/admin")
    
    def _hash_password(self, password: str, salt: str = None) -> str:
        """
        Hash a password with optional salt
        
        Args:
            password: Password to hash
            salt: Optional salt
            
        Returns:
            Hashed password
        """
        if salt is None:
            salt = self._generate_salt()
        
        # Combine password and salt
        salted_password = password + salt
        
        # Hash using SHA-256
        hash_obj = hashlib.sha256(salted_password.encode())
        return hash_obj.hexdigest()
    
    def _generate_salt(self) -> str:
        """
        Generate a random salt
        
        Returns:
            Random salt string
        """
        return secrets.token_hex(16)
    
    def _generate_session_id(self) -> str:
        """
        Generate a random session ID
        
        Returns:
            Random session ID
        """
        return secrets.token_urlsafe(32)
    
    def _generate_mfa_secret(self) -> str:
        """
        Generate a random MFA secret
        
        Returns:
            Random MFA secret
        """
        return base64.b32encode(secrets.token_bytes(10)).decode('utf-8')
    
    def _verify_mfa_code(self, secret: str, code: str) -> bool:
        """
        Verify an MFA code
        
        Args:
            secret: MFA secret
            code: MFA code to verify
            
        Returns:
            True if valid, False otherwise
        """
        try:
            import pyotp
            totp = pyotp.TOTP(secret)
            return totp.verify(code)
        except ImportError:
            logger.warning("pyotp not installed, MFA verification disabled")
            return True
        except Exception as e:
            logger.error(f"Error verifying MFA code: {e}")
            return False
    
    def create_user(self, username: str, password: str, role: str = "user", mfa_enabled: bool = False) -> bool:
        """
        Create a new user
        
        Args:
            username: Username
            password: Password
            role: User role
            mfa_enabled: Whether MFA is enabled
            
        Returns:
            True if successful, False otherwise
        """
        if username in self.users:
            logger.warning(f"User {username} already exists")
            return False
        
        salt = self._generate_salt()
        password_hash = self._hash_password(password, salt)
        mfa_secret = self._generate_mfa_secret() if mfa_enabled else ""
        
        user = {
            "username": username,
            "password_hash": password_hash,
            "salt": salt,
            "mfa_enabled": mfa_enabled,
            "mfa_secret": mfa_secret,
            "role": role,
            "created_at": time.time()
        }
        
        self.users[username] = user
        self._save_users()
        logger.info(f"Created user: {username}")
        
        return True
    
    def verify_user(self, username: str, password: str, mfa_code: str = None) -> bool:
        """
        Verify a user's credentials
        
        Args:
            username: Username
            password: Password
            mfa_code: MFA code (if enabled)
            
        Returns:
            True if valid, False otherwise
        """
        if username not in self.users:
            logger.warning(f"User {username} not found")
            return False
        
        user = self.users[username]
        salt = user.get("salt", "")
        password_hash = self._hash_password(password, salt)
        
        if password_hash != user.get("password_hash", ""):
            logger.warning(f"Invalid password for user {username}")
            return False
        
        # Check MFA if enabled
        if user.get("mfa_enabled", False):
            if not mfa_code:
                logger.warning(f"MFA code required for user {username}")
                return False
            
            if not self._verify_mfa_code(user.get("mfa_secret", ""), mfa_code):
                logger.warning(f"Invalid MFA code for user {username}")
                return False
        
        logger.info(f"User {username} authenticated successfully")
        return True
    
    def create_session(self, username: str) -> Optional[str]:
        """
        Create a new session for a user
        
        Args:
            username: Username
            
        Returns:
            Session ID or None if failed
        """
        if username not in self.users:
            logger.warning(f"User {username} not found")
            return None
        
        session_id = self._generate_session_id()
        expires = time.time() + 86400  # 24 hours
        
        session = {
            "username": username,
            "created_at": time.time(),
            "expires": expires,
            "role": self.users[username].get("role", "user")
        }
        
        self.sessions[session_id] = session
        self._save_sessions()
        logger.info(f"Created session for user {username}")
        
        return session_id
    
    def verify_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Verify a session
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data or None if invalid
        """
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found")
            return None
        
        session = self.sessions[session_id]
        
        # Check if expired
        if session.get("expires", 0) < time.time():
            logger.warning(f"Session {session_id} expired")
            del self.sessions[session_id]
            self._save_sessions()
            return None
        
        return session
    
    def end_session(self, session_id: str) -> bool:
        """
        End a session
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found")
            return False
        
        del self.sessions[session_id]
        self._save_sessions()
        logger.info(f"Ended session {session_id}")
        
        return True
    
    def enable_mfa(self, username: str) -> Optional[str]:
        """
        Enable MFA for a user
        
        Args:
            username: Username
            
        Returns:
            MFA secret or None if failed
        """
        if username not in self.users:
            logger.warning(f"User {username} not found")
            return None
        
        user = self.users[username]
        
        # Generate MFA secret if not already present
        if not user.get("mfa_secret"):
            user["mfa_secret"] = self._generate_mfa_secret()
        
        user["mfa_enabled"] = True
        self._save_users()
        logger.info(f"Enabled MFA for user {username}")
        
        return user["mfa_secret"]
    
    def disable_mfa(self, username: str) -> bool:
        """
        Disable MFA for a user
        
        Args:
            username: Username
            
        Returns:
            True if successful, False otherwise
        """
        if username not in self.users:
            logger.warning(f"User {username} not found")
            return False
        
        user = self.users[username]
        user["mfa_enabled"] = False
        self._save_users()
        logger.info(f"Disabled MFA for user {username}")
        
        return True
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """
        Change a user's password
        
        Args:
            username: Username
            old_password: Old password
            new_password: New password
            
        Returns:
            True if successful, False otherwise
        """
        if username not in self.users:
            logger.warning(f"User {username} not found")
            return False
        
        user = self.users[username]
        salt = user.get("salt", "")
        old_hash = self._hash_password(old_password, salt)
        
        if old_hash != user.get("password_hash", ""):
            logger.warning(f"Invalid old password for user {username}")
            return False
        
        # Generate new salt and hash
        new_salt = self._generate_salt()
        new_hash = self._hash_password(new_password, new_salt)
        
        user["password_hash"] = new_hash
        user["salt"] = new_salt
        self._save_users()
        logger.info(f"Changed password for user {username}")
        
        return True
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user information
        
        Args:
            username: Username
            
        Returns:
            User information or None if not found
        """
        if username not in self.users:
            logger.warning(f"User {username} not found")
            return None
        
        user = self.users[username].copy()
        
        # Remove sensitive information
        user.pop("password_hash", None)
        user.pop("salt", None)
        user.pop("mfa_secret", None)
        
        return user
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """
        Get all users
        
        Returns:
            List of user information
        """
        users = []
        for username, user in self.users.items():
            user_info = user.copy()
            
            # Remove sensitive information
            user_info.pop("password_hash", None)
            user_info.pop("salt", None)
            user_info.pop("mfa_secret", None)
            
            users.append(user_info)
        
        return users
    
    def delete_user(self, username: str) -> bool:
        """
        Delete a user
        
        Args:
            username: Username
            
        Returns:
            True if successful, False otherwise
        """
        if username not in self.users:
            logger.warning(f"User {username} not found")
            return False
        
        # Don't delete the last admin
        if self.users[username].get("role") == "admin":
            admin_count = sum(1 for user in self.users.values() if user.get("role") == "admin")
            if admin_count <= 1:
                logger.warning("Cannot delete the last admin user")
                return False
        
        del self.users[username]
        self._save_users()
        logger.info(f"Deleted user {username}")
        
        # End all sessions for this user
        session_ids = [
            session_id for session_id, session in self.sessions.items()
            if session.get("username") == username
        ]
        
        for session_id in session_ids:
            self.end_session(session_id)
        
        return True
    
    def generate_mfa_qr_code(self, username: str) -> Optional[str]:
        """
        Generate MFA QR code for a user
        
        Args:
            username: Username
            
        Returns:
            QR code URL or None if failed
        """
        if username not in self.users:
            logger.warning(f"User {username} not found")
            return None
        
        user = self.users[username]
        secret = user.get("mfa_secret")
        
        if not secret:
            logger.warning(f"No MFA secret for user {username}")
            return None
        
        try:
            import pyotp
            import qrcode
            import io
            import base64
            
            # Create OTP URI
            totp = pyotp.TOTP(secret)
            uri = totp.provisioning_uri(username, issuer_name="Viren Platinum")
            
            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(uri)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
        except ImportError:
            logger.warning("pyotp or qrcode not installed, QR code generation disabled")
            return None
        except Exception as e:
            logger.error(f"Error generating MFA QR code: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create auth manager
    auth_manager = AuthManager()
    
    # Create a test user
    auth_manager.create_user("test", "password", "user")
    
    # Verify user
    if auth_manager.verify_user("test", "password"):
        print("User verified")
    else:
        print("User verification failed")
    
    # Create session
    session_id = auth_manager.create_session("test")
    if session_id:
        print(f"Session created: {session_id}")
        
        # Verify session
        session = auth_manager.verify_session(session_id)
        if session:
            print(f"Session verified: {session}")
        
        # End session
        auth_manager.end_session(session_id)
    
    # Enable MFA
    secret = auth_manager.enable_mfa("test")
    if secret:
        print(f"MFA enabled with secret: {secret}")
        
        # Generate QR code
        qr_code = auth_manager.generate_mfa_qr_code("test")
        if qr_code:
            print("QR code generated")
        
        # Disable MFA
        auth_manager.disable_mfa("test")
    
    # Get user info
    user_info = auth_manager.get_user_info("test")
    if user_info:
        print(f"User info: {user_info}")
    
    # Delete user
    auth_manager.delete_user("test")