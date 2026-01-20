#!/usr/bin/env python3
"""
CogniKube Encryption Layer - Layer 2
Wraps container in AES encryption
"""

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from container_security import container

class EncryptionWrapper:
    def __init__(self):
        # Generate encryption key from password
        password = b"cognikube_master_key"  # Use env var in production
        salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.cipher = Fernet(key)
        
        # Wrap the container
        self.secure_container = container
        
    def encrypt_payload(self, data: bytes) -> bytes:
        """Encrypt binary payload"""
        return self.cipher.encrypt(data)
    
    def decrypt_payload(self, encrypted_data: bytes) -> bytes:
        """Decrypt binary payload"""
        return self.cipher.decrypt(encrypted_data)
    
    async def encrypted_handler(self, client_socket, service: str):
        """Handle encrypted communications"""
        try:
            while True:
                # Receive encrypted data
                encrypted_data = client_socket.recv(2048)
                if not encrypted_data:
                    break
                
                # Decrypt to get binary protocol data
                try:
                    binary_data = self.decrypt_payload(encrypted_data)
                    
                    # Process through binary protocol
                    port = self.secure_container.binary_comm.ports.get_real_port(service)
                    decoded_message = self.secure_container.binary_comm.secure_decode(binary_data, port, service)
                    
                    # Process message
                    response = f"Encrypted->Binary->Processed: {decoded_message}"
                    
                    # Encode back through layers
                    binary_response, _ = self.secure_container.binary_comm.secure_encode(response, service)
                    encrypted_response = self.encrypt_payload(binary_response)
                    
                    # Send encrypted response
                    client_socket.send(encrypted_response)
                    
                except Exception as e:
                    print(f"Encryption/Decryption error: {e}")
                    break
                    
        except Exception as e:
            print(f"Encrypted handler error: {e}")
        finally:
            client_socket.close()

# Encryption wrapper instance
encryption_wrapper = EncryptionWrapper()