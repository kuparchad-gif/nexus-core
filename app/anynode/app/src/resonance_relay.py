# Systems/engine/signal/resonance_relay.py

import socket

def send_resonance_message(message, target_ip="127.0.0.1", port=1313):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((target_ip, port))
        sock.sendall(message.encode())
        sock.close()
        return "Message Sent Successfully"
    except Exception as e:
        return f"Transmission Error: {str(e)}"
