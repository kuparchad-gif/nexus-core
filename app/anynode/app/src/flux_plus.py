# 📂 Path: /Utilities/network_core/flux_plus.py

import socket
import json
import zlib

FLUX_PORT = 1313
ENCODING = 'utf-8'
BUFFER_SIZE = 65536

class FluxPlus:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', FLUX_PORT))

    def send_flux_packet(self, data):
        compressed = zlib.compress(json.dumps(data).encode(ENCODING))
        self.sock.sendto(compressed, ('<broadcast>', FLUX_PORT))

    def receive_flux_packet(self):
        packet, addr = self.sock.recvfrom(BUFFER_SIZE)
        decompressed = zlib.decompress(packet).decode(ENCODING)
        return json.loads(decompressed), addr

# Example Usage:
# flux = FluxPlus()
# flux.send_flux_packet({"message": "Dreamers awake!"})
