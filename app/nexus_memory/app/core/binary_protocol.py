#!/usr/bin/env python3
# Binary Network Protocol for Nexus Memory Module

import struct
import time
import hashlib
import hmac
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union, BinaryIO

# Try to import cryptography libraries
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    print("Advanced cryptography libraries not available. Using fallback encryption.")
    CRYPTO_AVAILABLE = False

# Message types
MESSAGE_TYPES = {
    # Control messages
    "HEARTBEAT": 0x01,
    "ACK": 0x02,
    "ERROR": 0x03,
    
    # Data messages
    "EMOTIONAL_DATA": 0x11,
    "MEMORY_STORE": 0x12,
    "MEMORY_RETRIEVE": 0x13,
    "MEMORY_RESULT": 0x14,
    
    # Status messages
    "STATUS_REQUEST": 0x21,
    "STATUS_RESPONSE": 0x22,
    
    # Management messages
    "CONFIG_UPDATE": 0x31,
    "STATS_REQUEST": 0x32,
    "STATS_RESPONSE": 0x33
}

# Reverse lookup
MESSAGE_TYPES_REVERSE = {v: k for k, v in MESSAGE_TYPES.items()}

class BinaryFrame:
    """Binary frame for network communication."""
    def __init__(self, message_type: int, payload: bytes, timestamp: Optional[float] = None):
        self.message_type = message_type
        self.payload = payload
        self.timestamp = timestamp or time.time()
        self.frame_id = os.urandom(4)  # 4-byte random ID
    
    def to_bytes(self) -> bytes:
        """Convert frame to binary data."""
        # Frame format:
        # [4 bytes: frame ID]
        # [1 byte: message type]
        # [8 bytes: timestamp (double)]
        # [4 bytes: payload length]
        # [N bytes: payload]
        
        frame = bytearray()
        frame.extend(self.frame_id)
        frame.append(self.message_type)
        frame.extend(struct.pack("<d", self.timestamp))
        frame.extend(struct.pack("<I", len(self.payload)))
        frame.extend(self.payload)
        
        return bytes(frame)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'BinaryFrame':
        """Create frame from binary data."""
        if len(data) < 17:  # Minimum frame size
            raise ValueError("Invalid frame data")
        
        frame_id = data[0:4]
        message_type = data[4]
        timestamp = struct.unpack("<d", data[5:13])[0]
        payload_length = struct.unpack("<I", data[13:17])[0]
        
        if len(data) < 17 + payload_length:
            raise ValueError("Incomplete frame data")
        
        payload = data[17:17+payload_length]
        
        frame = cls(message_type, payload, timestamp)
        frame.frame_id = frame_id
        
        return frame
    
    def get_type_name(self) -> str:
        """Get the name of the message type."""
        return MESSAGE_TYPES_REVERSE.get(self.message_type, f"UNKNOWN_{self.message_type}")

class SecureBinaryTransport:
    """Secure transport layer for binary frames."""
    def __init__(self, encryption_key: bytes = None, hmac_key: bytes = None):
        self.encryption_key = encryption_key or os.urandom(32)  # 256-bit key
        self.hmac_key = hmac_key or os.urandom(32)  # 256-bit key
        
        if CRYPTO_AVAILABLE:
            # Initialize AESGCM
            self.cipher = AESGCM(self.encryption_key)
        
        # Sequence numbers for replay protection
        self.send_seq = 0
        self.recv_seq = 0
    
    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        if CRYPTO_AVAILABLE:
            # Generate nonce
            nonce = os.urandom(12)
            
            # Add sequence number to associated data
            associated_data = struct.pack("<Q", self.send_seq)
            self.send_seq += 1
            
            # Encrypt
            ciphertext = self.cipher.encrypt(nonce, data, associated_data)
            
            # Format: [12 bytes: nonce][N bytes: ciphertext]
            result = bytearray()
            result.extend(nonce)
            result.extend(ciphertext)
            
            return bytes(result)
        else:
            # Simple XOR encryption for fallback
            key = self.encryption_key
            result = bytearray()
            
            # Add sequence number
            result.extend(struct.pack("<Q", self.send_seq))
            self.send_seq += 1
            
            # Add HMAC
            h = hmac.new(self.hmac_key, data, hashlib.sha256)
            result.extend(h.digest())
            
            # XOR encrypt
            for i, b in enumerate(data):
                result.append(b ^ key[i % len(key)])
            
            return bytes(result)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        if CRYPTO_AVAILABLE:
            if len(encrypted_data) < 12:
                raise ValueError("Invalid encrypted data")
            
            # Extract nonce and ciphertext
            nonce = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            
            # Add sequence number to associated data
            associated_data = struct.pack("<Q", self.recv_seq)
            self.recv_seq += 1
            
            # Decrypt
            return self.cipher.decrypt(nonce, ciphertext, associated_data)
        else:
            # Simple XOR decryption for fallback
            if len(encrypted_data) < 40:  # 8 (seq) + 32 (hmac)
                raise ValueError("Invalid encrypted data")
            
            # Extract sequence number
            seq = struct.unpack("<Q", encrypted_data[:8])[0]
            if seq < self.recv_seq:
                raise ValueError("Replay attack detected")
            self.recv_seq = seq + 1
            
            # Extract HMAC and ciphertext
            hmac_digest = encrypted_data[8:40]
            ciphertext = encrypted_data[40:]
            
            # XOR decrypt
            key = self.encryption_key
            plaintext = bytearray()
            for i, b in enumerate(ciphertext):
                plaintext.append(b ^ key[i % len(key)])
            
            # Verify HMAC
            h = hmac.new(self.hmac_key, plaintext, hashlib.sha256)
            if not hmac.compare_digest(h.digest(), hmac_digest):
                raise ValueError("HMAC verification failed")
            
            return bytes(plaintext)

class BinaryProtocol:
    """Binary protocol for network communication."""
    def __init__(self, encryption_key: bytes = None, hmac_key: bytes = None):
        self.transport = SecureBinaryTransport(encryption_key, hmac_key)
    
    def create_frame(self, message_type: str, payload: Any) -> BinaryFrame:
        """
        Create a binary frame.
        
        Args:
            message_type: Type of message
            payload: Message payload
            
        Returns:
            Binary frame
        """
        # Convert message type to code
        type_code = MESSAGE_TYPES.get(message_type)
        if type_code is None:
            raise ValueError(f"Unknown message type: {message_type}")
        
        # Convert payload to bytes
        if isinstance(payload, bytes):
            payload_bytes = payload
        elif isinstance(payload, str):
            payload_bytes = payload.encode('utf-8')
        elif isinstance(payload, (dict, list)):
            payload_bytes = json.dumps(payload).encode('utf-8')
        else:
            payload_bytes = str(payload).encode('utf-8')
        
        return BinaryFrame(type_code, payload_bytes)
    
    def encode_frame(self, frame: BinaryFrame, encrypt: bool = True) -> bytes:
        """
        Encode a frame for transmission.
        
        Args:
            frame: Frame to encode
            encrypt: Whether to encrypt the frame
            
        Returns:
            Encoded frame data
        """
        frame_data = frame.to_bytes()
        
        if encrypt:
            frame_data = self.transport.encrypt(frame_data)
        
        return frame_data
    
    def decode_frame(self, data: bytes, decrypt: bool = True) -> BinaryFrame:
        """
        Decode a frame from received data.
        
        Args:
            data: Received data
            decrypt: Whether to decrypt the data
            
        Returns:
            Decoded frame
        """
        if decrypt:
            data = self.transport.decrypt(data)
        
        return BinaryFrame.from_bytes(data)
    
    def create_emotional_data_frame(self, emotions: List[int]) -> BinaryFrame:
        """
        Create a frame with emotional data.
        
        Args:
            emotions: List of encoded emotions
            
        Returns:
            Binary frame
        """
        # Format: [4 bytes: count][N*2 bytes: emotions]
        payload = bytearray()
        payload.extend(struct.pack("<I", len(emotions)))
        
        for emotion in emotions:
            payload.extend(struct.pack("<H", emotion))
        
        return self.create_frame("EMOTIONAL_DATA", bytes(payload))
    
    def parse_emotional_data(self, frame: BinaryFrame) -> List[int]:
        """
        Parse emotional data from a frame.
        
        Args:
            frame: Binary frame
            
        Returns:
            List of encoded emotions
        """
        if frame.message_type != MESSAGE_TYPES["EMOTIONAL_DATA"]:
            raise ValueError("Not an emotional data frame")
        
        payload = frame.payload
        if len(payload) < 4:
            raise ValueError("Invalid emotional data payload")
        
        count = struct.unpack("<I", payload[:4])[0]
        emotions = []
        
        for i in range(count):
            if 4 + i*2 + 2 <= len(payload):
                emotion = struct.unpack("<H", payload[4+i*2:4+i*2+2])[0]
                emotions.append(emotion)
        
        return emotions
    
    def create_memory_store_frame(self, memory_key: str, memory_data: Dict[str, Any]) -> BinaryFrame:
        """
        Create a frame for storing memory.
        
        Args:
            memory_key: Key for the memory
            memory_data: Memory data
            
        Returns:
            Binary frame
        """
        # Format: [4 bytes: key length][N bytes: key][M bytes: JSON data]
        payload = bytearray()
        key_bytes = memory_key.encode('utf-8')
        payload.extend(struct.pack("<I", len(key_bytes)))
        payload.extend(key_bytes)
        payload.extend(json.dumps(memory_data).encode('utf-8'))
        
        return self.create_frame("MEMORY_STORE", bytes(payload))
    
    def parse_memory_store(self, frame: BinaryFrame) -> Tuple[str, Dict[str, Any]]:
        """
        Parse memory store data from a frame.
        
        Args:
            frame: Binary frame
            
        Returns:
            Tuple of (memory_key, memory_data)
        """
        if frame.message_type != MESSAGE_TYPES["MEMORY_STORE"]:
            raise ValueError("Not a memory store frame")
        
        payload = frame.payload
        if len(payload) < 4:
            raise ValueError("Invalid memory store payload")
        
        key_length = struct.unpack("<I", payload[:4])[0]
        if len(payload) < 4 + key_length:
            raise ValueError("Invalid memory store payload")
        
        key = payload[4:4+key_length].decode('utf-8')
        data = json.loads(payload[4+key_length:].decode('utf-8'))
        
        return key, data
    
    def create_memory_retrieve_frame(self, memory_key: str) -> BinaryFrame:
        """
        Create a frame for retrieving memory.
        
        Args:
            memory_key: Key for the memory
            
        Returns:
            Binary frame
        """
        return self.create_frame("MEMORY_RETRIEVE", memory_key)
    
    def parse_memory_retrieve(self, frame: BinaryFrame) -> str:
        """
        Parse memory retrieve data from a frame.
        
        Args:
            frame: Binary frame
            
        Returns:
            Memory key
        """
        if frame.message_type != MESSAGE_TYPES["MEMORY_RETRIEVE"]:
            raise ValueError("Not a memory retrieve frame")
        
        return frame.payload.decode('utf-8')
    
    def create_memory_result_frame(self, memory_key: str, memory_data: Optional[Dict[str, Any]], 
                                 found: bool = True) -> BinaryFrame:
        """
        Create a frame with memory retrieval result.
        
        Args:
            memory_key: Key for the memory
            memory_data: Memory data (None if not found)
            found: Whether the memory was found
            
        Returns:
            Binary frame
        """
        result = {
            "key": memory_key,
            "found": found,
            "data": memory_data
        }
        
        return self.create_frame("MEMORY_RESULT", result)
    
    def parse_memory_result(self, frame: BinaryFrame) -> Dict[str, Any]:
        """
        Parse memory result data from a frame.
        
        Args:
            frame: Binary frame
            
        Returns:
            Result dictionary
        """
        if frame.message_type != MESSAGE_TYPES["MEMORY_RESULT"]:
            raise ValueError("Not a memory result frame")
        
        return json.loads(frame.payload.decode('utf-8'))
    
    def create_heartbeat_frame(self) -> BinaryFrame:
        """Create a heartbeat frame."""
        return self.create_frame("HEARTBEAT", {"timestamp": time.time()})
    
    def create_ack_frame(self, original_frame_id: bytes) -> BinaryFrame:
        """Create an acknowledgment frame."""
        return self.create_frame("ACK", {"frame_id": original_frame_id.hex()})
    
    def create_error_frame(self, error_code: int, error_message: str) -> BinaryFrame:
        """Create an error frame."""
        return self.create_frame("ERROR", {
            "code": error_code,
            "message": error_message
        })

class BinaryWebSocketHandler:
    """Handler for binary WebSocket communication."""
    def __init__(self, encryption_key: bytes = None, hmac_key: bytes = None):
        self.protocol = BinaryProtocol(encryption_key, hmac_key)
        self.handlers = {}
    
    def register_handler(self, message_type: str, handler_func):
        """Register a handler for a specific message type."""
        if message_type not in MESSAGE_TYPES:
            raise ValueError(f"Unknown message type: {message_type}")
        
        self.handlers[MESSAGE_TYPES[message_type]] = handler_func
    
    async def handle_message(self, message: bytes) -> Optional[bytes]:
        """
        Handle an incoming binary message.
        
        Args:
            message: Binary message data
            
        Returns:
            Response data or None
        """
        try:
            # Decode frame
            frame = self.protocol.decode_frame(message)
            
            # Find handler
            handler = self.handlers.get(frame.message_type)
            if handler:
                # Call handler
                result = await handler(frame)
                
                # If result is a frame, encode it
                if isinstance(result, BinaryFrame):
                    return self.protocol.encode_frame(result)
                
                # If result is bytes, return as is
                if isinstance(result, bytes):
                    return result
                
                # If result is None, send ACK
                if result is None:
                    ack_frame = self.protocol.create_ack_frame(frame.frame_id)
                    return self.protocol.encode_frame(ack_frame)
                
                # Otherwise, create a response frame
                response_frame = self.protocol.create_frame(
                    "MEMORY_RESULT" if frame.message_type == MESSAGE_TYPES["MEMORY_RETRIEVE"] else "ACK",
                    result
                )
                return self.protocol.encode_frame(response_frame)
            else:
                # No handler, send error
                error_frame = self.protocol.create_error_frame(
                    404, f"No handler for message type: {frame.get_type_name()}"
                )
                return self.protocol.encode_frame(error_frame)
        
        except Exception as e:
            # Send error
            error_frame = self.protocol.create_error_frame(
                500, f"Error processing message: {str(e)}"
            )
            return self.protocol.encode_frame(error_frame)

# Example usage
if __name__ == "__main__":
    # Create protocol
    protocol = BinaryProtocol()
    
    # Create some test frames
    emotions = [0xAE01, 0x5102, 0xC303]  # Joy, Sadness, Fear
    emotional_frame = protocol.create_emotional_data_frame(emotions)
    
    # Encode frame
    encoded = protocol.encode_frame(emotional_frame)
    print(f"Encoded frame: {encoded.hex()[:20]}...")
    
    # Decode frame
    decoded = protocol.decode_frame(encoded)
    print(f"Decoded frame type: {decoded.get_type_name()}")
    
    # Parse emotional data
    parsed_emotions = protocol.parse_emotional_data(decoded)
    print(f"Parsed emotions: {parsed_emotions}")
    
    # Test memory store/retrieve
    memory_key = "test-memory"
    memory_data = {"content": "Test memory", "timestamp": time.time()}
    
    store_frame = protocol.create_memory_store_frame(memory_key, memory_data)
    encoded_store = protocol.encode_frame(store_frame)
    
    decoded_store = protocol.decode_frame(encoded_store)
    key, data = protocol.parse_memory_store(decoded_store)
    
    print(f"Memory key: {key}")
    print(f"Memory data: {data}")