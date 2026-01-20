#!/usr/bin/env python3
"""
CogniKube Authentication Gateway - Layer 3
JWT + API Key authentication wrapping encryption
"""

import jwt
import hashlib
import time
from encryption_layer import encryption_wrapper
import json

class AuthGateway:
    def __init__(self):
        self.jwt_secret = "cognikube_jwt_secret"
        self.api_keys = {
            "admin": "ck_admin_key_12345",
            "user": "ck_user_key_67890"
        }
        self.encryption_layer = encryption_wrapper
        
    def validate_api_key(self, api_key: str) -> str:
        """Validate API key and return role"""
        for role, key in self.api_keys.items():
            if key == api_key:
                return role
        return None
    
    def generate_jwt(self, role: str) -> str:
        """Generate JWT token for authenticated role"""
        payload = {
            "role": role,
            "exp": time.time() + 3600,  # 1 hour
            "iat": time.time()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def validate_jwt(self, token: str) -> str:
        """Validate JWT and return role"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload.get("role")
        except:
            return None
    
    async def authenticated_handler(self, client_socket, service: str):
        """Handle authenticated communications"""
        try:
            # First message must be authentication
            auth_data = client_socket.recv(1024)
            
            # Decrypt auth data
            try:
                decrypted_auth = self.encryption_layer.decrypt_payload(auth_data)
                
                # Decode binary protocol
                port = self.encryption_layer.secure_container.binary_comm.ports.get_real_port(service)
                auth_message = self.encryption_layer.secure_container.binary_comm.secure_decode(decrypted_auth, port, service)
                
                # Parse auth message
                auth_data = json.loads(auth_message)
                
                # Validate authentication
                role = None
                if "api_key" in auth_data:
                    role = self.validate_api_key(auth_data["api_key"])
                elif "jwt" in auth_data:
                    role = self.validate_jwt(auth_data["jwt"])
                
                if not role:
                    # Send auth failure
                    error_response = json.dumps({"error": "authentication_failed"})
                    binary_error, _ = self.encryption_layer.secure_container.binary_comm.secure_encode(error_response, service)
                    encrypted_error = self.encryption_layer.encrypt_payload(binary_error)
                    client_socket.send(encrypted_error)
                    return
                
                # Send auth success with JWT
                jwt_token = self.generate_jwt(role)
                success_response = json.dumps({"status": "authenticated", "jwt": jwt_token, "role": role})
                binary_success, _ = self.encryption_layer.secure_container.binary_comm.secure_encode(success_response, service)
                encrypted_success = self.encryption_layer.encrypt_payload(binary_success)
                client_socket.send(encrypted_success)
                
                # Now handle authenticated requests
                await self.handle_authenticated_requests(client_socket, service, role)
                
            except Exception as e:
                print(f"Authentication error: {e}")
                
        except Exception as e:
            print(f"Auth gateway error: {e}")
        finally:
            client_socket.close()
    
    async def handle_authenticated_requests(self, client_socket, service: str, role: str):
        """Handle requests from authenticated client"""
        while True:
            try:
                # Receive encrypted request
                encrypted_request = client_socket.recv(2048)
                if not encrypted_request:
                    break
                
                # Process through all layers
                binary_data = self.encryption_layer.decrypt_payload(encrypted_request)
                port = self.encryption_layer.secure_container.binary_comm.ports.get_real_port(service)
                decoded_message = self.encryption_layer.secure_container.binary_comm.secure_decode(binary_data, port, service)
                
                # Add role-based processing
                response = f"[{role.upper()}] Processed: {decoded_message}"
                
                # Send back through all layers
                binary_response, _ = self.encryption_layer.secure_container.binary_comm.secure_encode(response, service)
                encrypted_response = self.encryption_layer.encrypt_payload(binary_response)
                client_socket.send(encrypted_response)
                
            except Exception as e:
                print(f"Authenticated request error: {e}")
                break

# Auth gateway instance
auth_gateway = AuthGateway()