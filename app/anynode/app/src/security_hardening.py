# security_hardening.py
# Purpose: Security hardening for Viren

import os
import sys
import json
import hashlib
import secrets
import logging
import argparse
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("security_hardening")

class SecurityHardening:
    """
    Security hardening for Viren.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the security hardening."""
        self.config_path = config_path or "security_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the security configuration."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading security configuration: {e}")
        
        # Default configuration
        return {
            "auth_token_length": 32,
            "password_min_length": 12,
            "password_complexity": True,
            "session_timeout": 3600,  # 1 hour
            "max_login_attempts": 5,
            "lockout_duration": 300,  # 5 minutes
            "secure_communication": True,
            "encryption_key_rotation": 30,  # 30 days
            "access_control": {
                "admin": ["all"],
                "user": ["read", "execute"],
                "guest": ["read"]
            }
        }
    
    def save_config(self) -> bool:
        """Save the security configuration."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved security configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving security configuration: {e}")
            return False
    
    def generate_auth_token(self) -> str:
        """
        Generate a secure authentication token.
        
        Returns:
            Secure authentication token
        """
        token_length = self.config.get("auth_token_length", 32)
        return secrets.token_hex(token_length)
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password securely.
        
        Args:
            password: Password to hash
            
        Returns:
            Hashed password
        """
        # Generate a random salt
        salt = os.urandom(32)
        
        # Hash the password with the salt
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        
        # Return the salt and key as a hex string
        return salt.hex() + ':' + key.hex()
    
    def verify_password(self, stored_hash: str, password: str) -> bool:
        """
        Verify a password against a stored hash.
        
        Args:
            stored_hash: Stored password hash
            password: Password to verify
            
        Returns:
            True if the password is correct, False otherwise
        """
        # Split the stored hash into salt and key
        salt_hex, key_hex = stored_hash.split(':')
        salt = bytes.fromhex(salt_hex)
        stored_key = bytes.fromhex(key_hex)
        
        # Hash the password with the salt
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        
        # Compare the keys
        return key == stored_key
    
    def validate_password_complexity(self, password: str) -> bool:
        """
        Validate password complexity.
        
        Args:
            password: Password to validate
            
        Returns:
            True if the password meets complexity requirements, False otherwise
        """
        # Check minimum length
        min_length = self.config.get("password_min_length", 12)
        if len(password) < min_length:
            return False
        
        # Check complexity if enabled
        if self.config.get("password_complexity", True):
            # Check for uppercase, lowercase, digit, and special character
            has_upper = any(c.isupper() for c in password)
            has_lower = any(c.islower() for c in password)
            has_digit = any(c.isdigit() for c in password)
            has_special = any(not c.isalnum() for c in password)
            
            return has_upper and has_lower and has_digit and has_special
        
        return True
    
    def generate_encryption_key(self) -> str:
        """
        Generate a secure encryption key.
        
        Returns:
            Secure encryption key
        """
        return secrets.token_hex(32)
    
    def encrypt_file(self, file_path: str, key: str) -> bool:
        """
        Encrypt a file.
        
        Args:
            file_path: Path to the file to encrypt
            key: Encryption key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from cryptography.fernet import Fernet
            
            # Convert key to Fernet key
            fernet_key = hashlib.sha256(key.encode()).digest()
            f = Fernet(base64.b64encode(fernet_key))
            
            # Read the file
            with open(file_path, 'rb') as file:
                file_data = file.read()
            
            # Encrypt the data
            encrypted_data = f.encrypt(file_data)
            
            # Write the encrypted data
            with open(file_path + '.enc', 'wb') as file:
                file.write(encrypted_data)
            
            logger.info(f"Encrypted file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error encrypting file {file_path}: {e}")
            return False
    
    def decrypt_file(self, file_path: str, key: str) -> bool:
        """
        Decrypt a file.
        
        Args:
            file_path: Path to the encrypted file
            key: Encryption key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from cryptography.fernet import Fernet
            
            # Convert key to Fernet key
            fernet_key = hashlib.sha256(key.encode()).digest()
            f = Fernet(base64.b64encode(fernet_key))
            
            # Read the encrypted file
            with open(file_path, 'rb') as file:
                encrypted_data = file.read()
            
            # Decrypt the data
            decrypted_data = f.decrypt(encrypted_data)
            
            # Write the decrypted data
            output_path = file_path[:-4] if file_path.endswith('.enc') else file_path + '.dec'
            with open(output_path, 'wb') as file:
                file.write(decrypted_data)
            
            logger.info(f"Decrypted file: {file_path} -> {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error decrypting file {file_path}: {e}")
            return False
    
    def check_access(self, user: str, action: str) -> bool:
        """
        Check if a user has access to perform an action.
        
        Args:
            user: User name
            action: Action to perform
            
        Returns:
            True if the user has access, False otherwise
        """
        access_control = self.config.get("access_control", {})
        
        # Get user permissions
        permissions = access_control.get(user, [])
        
        # Check if user has permission
        return "all" in permissions or action in permissions
    
    def harden_system(self) -> bool:
        """
        Apply security hardening to the system.
        
        Returns:
            True if successful, False otherwise
        """
        success = True
        
        # Secure communication
        if self.config.get("secure_communication", True):
            success = success and self._secure_communication()
        
        # Access control
        success = success and self._setup_access_control()
        
        # Encryption
        success = success and self._setup_encryption()
        
        # Audit logging
        success = success and self._setup_audit_logging()
        
        return success
    
    def _secure_communication(self) -> bool:
        """
        Secure communication channels.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate SSL certificate
            self._generate_ssl_certificate()
            
            # Update service discovery to use SSL
            self._update_service_discovery_ssl()
            
            logger.info("Secured communication channels")
            return True
        except Exception as e:
            logger.error(f"Error securing communication channels: {e}")
            return False
    
    def _generate_ssl_certificate(self) -> bool:
        """
        Generate SSL certificate.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create certificates directory
            os.makedirs("certificates", exist_ok=True)
            
            # Generate self-signed certificate
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization
            import datetime
            
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Write private key
            with open("certificates/private.key", "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Generate certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Viren"),
                x509.NameAttribute(NameOID.COMMON_NAME, "viren.local")
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.datetime.utcnow()
            ).not_valid_after(
                datetime.datetime.utcnow() + datetime.timedelta(days=365)
            ).add_extension(
                x509.SubjectAlternativeName([x509.DNSName("localhost")]),
                critical=False
            ).sign(private_key, hashes.SHA256())
            
            # Write certificate
            with open("certificates/certificate.crt", "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            
            logger.info("Generated SSL certificate")
            return True
        except Exception as e:
            logger.error(f"Error generating SSL certificate: {e}")
            return False
    
    def _update_service_discovery_ssl(self) -> bool:
        """
        Update service discovery to use SSL.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if service discovery file exists
            service_discovery_path = "Systems/engine/core/service_discovery.py"
            if not os.path.exists(service_discovery_path):
                logger.warning(f"Service discovery file not found: {service_discovery_path}")
                return False
            
            # Read the file
            with open(service_discovery_path, 'r') as f:
                content = f.read()
            
            # Check if SSL is already enabled
            if "ssl_context" in content:
                logger.info("SSL already enabled in service discovery")
                return True
            
            # Add SSL imports
            ssl_imports = """import ssl
import os.path
"""
            content = content.replace("import socket", "import socket\n" + ssl_imports)
            
            # Add SSL context creation
            ssl_context_code = """
    def _create_ssl_context(self):
        \"\"\"Create SSL context.\"\"\"
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_path = "certificates/certificate.crt"
        key_path = "certificates/private.key"
        
        if os.path.exists(cert_path) and os.path.exists(key_path):
            context.load_cert_chain(certfile=cert_path, keyfile=key_path)
            return context
        else:
            logger.warning(f"SSL certificate or key not found: {cert_path}, {key_path}")
            return None
"""
            
            # Find the class definition
            class_def_index = content.find("class ServiceDiscovery:")
            if class_def_index == -1:
                logger.warning("ServiceDiscovery class not found in service discovery file")
                return False
            
            # Find the end of the class methods
            method_indent = "    "
            last_method_index = content.rfind(f"\n{method_indent}def ")
            if last_method_index == -1:
                logger.warning("Could not find last method in ServiceDiscovery class")
                return False
            
            # Find the end of the last method
            last_method_end = content.find("\n\n", last_method_index)
            if last_method_end == -1:
                last_method_end = len(content)
            
            # Insert SSL context creation method
            content = content[:last_method_end] + ssl_context_code + content[last_method_end:]
            
            # Update socket creation to use SSL
            content = content.replace(
                "self.registry_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)",
                "self.registry_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n        self.ssl_context = self._create_ssl_context()"
            )
            
            content = content.replace(
                "self.registry_socket.listen(5)",
                "self.registry_socket.listen(5)\n            if self.ssl_context:"
            )
            
            content = content.replace(
                "client, addr = self.registry_socket.accept()",
                "client, addr = self.registry_socket.accept()\n                    if self.ssl_context:\n                        client = self.ssl_context.wrap_socket(client, server_side=True)"
            )
            
            # Write the updated file
            with open(service_discovery_path, 'w') as f:
                f.write(content)
            
            logger.info("Updated service discovery to use SSL")
            return True
        except Exception as e:
            logger.error(f"Error updating service discovery to use SSL: {e}")
            return False
    
    def _setup_access_control(self) -> bool:
        """
        Set up access control.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create access control file
            access_control_path = "Systems/security/access_control.json"
            os.makedirs(os.path.dirname(access_control_path), exist_ok=True)
            
            # Write access control configuration
            with open(access_control_path, 'w') as f:
                json.dump({
                    "roles": {
                        "admin": {
                            "permissions": ["all"]
                        },
                        "user": {
                            "permissions": ["read", "execute"]
                        },
                        "guest": {
                            "permissions": ["read"]
                        }
                    },
                    "users": {
                        "viren": {
                            "role": "admin",
                            "password_hash": self.hash_password("viren_admin_password")
                        },
                        "user": {
                            "role": "user",
                            "password_hash": self.hash_password("user_password")
                        },
                        "guest": {
                            "role": "guest",
                            "password_hash": self.hash_password("guest_password")
                        }
                    }
                }, f, indent=2)
            
            logger.info("Set up access control")
            return True
        except Exception as e:
            logger.error(f"Error setting up access control: {e}")
            return False
    
    def _setup_encryption(self) -> bool:
        """
        Set up encryption.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create encryption keys file
            encryption_keys_path = "Systems/security/encryption_keys.json"
            os.makedirs(os.path.dirname(encryption_keys_path), exist_ok=True)
            
            # Generate encryption keys
            keys = {
                "data_key": self.generate_encryption_key(),
                "communication_key": self.generate_encryption_key(),
                "session_key": self.generate_encryption_key(),
                "created": int(time.time()),
                "rotation_days": self.config.get("encryption_key_rotation", 30)
            }
            
            # Write encryption keys
            with open(encryption_keys_path, 'w') as f:
                json.dump(keys, f, indent=2)
            
            logger.info("Set up encryption")
            return True
        except Exception as e:
            logger.error(f"Error setting up encryption: {e}")
            return False
    
    def _setup_audit_logging(self) -> bool:
        """
        Set up audit logging.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create audit logger
            audit_logger = logging.getLogger("audit")
            audit_logger.setLevel(logging.INFO)
            
            # Create audit log directory
            os.makedirs("logs/audit", exist_ok=True)
            
            # Create audit log handler
            handler = logging.FileHandler("logs/audit/audit.log")
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            audit_logger.addHandler(handler)
            
            # Test audit logger
            audit_logger.info("Audit logging initialized")
            
            logger.info("Set up audit logging")
            return True
        except Exception as e:
            logger.error(f"Error setting up audit logging: {e}")
            return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Security hardening for Viren")
    parser.add_argument("--config", help="Path to security configuration file")
    args = parser.parse_args()
    
    # Create security hardening
    security = SecurityHardening(args.config)
    
    # Apply security hardening
    success = security.harden_system()
    
    if success:
        print("Security hardening completed successfully")
    else:
        print("Security hardening completed with errors")
        sys.exit(1)

if __name__ == "__main__":
    main()