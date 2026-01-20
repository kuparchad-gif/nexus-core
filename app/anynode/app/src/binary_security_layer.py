#!/usr/bin/env python3
"""
CogniKube 13-Bit Binary Security Foundation
Layer 0: Binary protocol obscurity with checksum
"""

import struct
import random
from typing import Dict, List
import hashlib

class BinaryProtocol:
    def __init__(self):
        self.bit_mask = 0b1111111111111  # 13 bits
        self.shadow_bit = 0b0000000000000  # Shadow bit (bit 0)
        
    def encode_13bit(self, data: str) -> bytes:
        """Encode string to 13-bit binary packets with checksum"""
        binary_packets = []
        
        # Calculate checksum
        checksum = hashlib.sha256(data.encode()).digest()[:4]  # 4-byte checksum
        
        for char in data:
            ascii_val = ord(char)
            packed = (ascii_val & self.bit_mask) | self.shadow_bit
            binary_packets.append(struct.pack('>H', packed))
            
        # Append checksum
        binary_packets.append(checksum)
        return b''.join(binary_packets)
    
    def decode_13bit(self, binary_data: bytes) -> str:
        """Decode 13-bit binary back to string with checksum validation"""
        if len(binary_data) < 4:
            raise ValueError("Invalid data length")
            
        # Extract data and checksum
        data_bytes = binary_data[:-4]
        received_checksum = binary_data[-4:]
        decoded_chars = []
        
        for i in range(0, len(data_bytes), 2):
            if i + 1 < len(data_bytes):
                packed = struct.unpack('>H', data_bytes[i:i+2])[0]
                ascii_val = packed & self.bit_mask
                if ascii_val > 0:
                    decoded_chars.append(chr(ascii_val))
        
        decoded = ''.join(decoded_chars)
        # Verify checksum
        calculated_checksum = hashlib.sha256(decoded.encode()).digest()[:4]
        if calculated_checksum != received_checksum:
            raise ValueError("Checksum validation failed")
                    
        return decoded

class ObscuredPorts:
    def __init__(self):
        self.port_map = {
            "api": 31337,
            "ws": 13370,
            "health": 7331,
            "admin": 1337
        }
        self.port_shift = random.randint(1000, 9999)
        
    def get_real_port(self, service: str) -> int:
        base_port = self.port_map.get(service, 8080)
        return base_port + self.port_shift
    
    def is_valid_port(self, port: int, service: str) -> bool:
        expected = self.get_real_port(service)
        return port == expected

binary_proto = BinaryProtocol()
port_obscurer = ObscuredPorts()

class SecureBinaryComm:
    def __init__(self):
        self.protocol = binary_proto
        self.ports = port_obscurer
        
    def secure_encode(self, message: str, service: str) -> tuple:
        binary_data = self.protocol.encode_13bit(message)
        port = self.ports.get_real_port(service)
        return binary_data, port
    
    def secure_decode(self, binary_data: bytes, port: int, service: str) -> str:
        if not self.ports.is_valid_port(port, service):
            raise ValueError("Invalid port for service")
        return self.protocol.decode_13bit(binary_data)

secure_comm = SecureBinaryComm()

if __name__ == "__main__":
    test_message = "Hello CogniKube"
    binary_data, port = secure_comm.secure_encode(test_message, "api")
    print(f"Encoded to binary: {binary_data.hex()}")
    print(f"Using obscured port: {port}")
    decoded = secure_comm.secure_decode(binary_data, port, "api")
    print(f"Decoded: {decoded}")
    print(f"Port mapping: {port_obscurer.port_map}")
    print(f"Port shift: {port_obscurer.port_shift}")