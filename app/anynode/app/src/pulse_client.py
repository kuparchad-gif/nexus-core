import socket

def send_pulse(target_ip):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((target_ip, 1313))
        client_socket.send(b"Pulse request.")
        response = client_socket.recv(1024)
        print(f"🔔 Pulse response from {target_ip}: {response.decode()}")
        client_socket.close()
    except Exception as e:
        print(f"❌ Failed to pulse {target_ip}: {e}")

if __name__ == "__main__":
    # Example: send_pulse("10.13.13.1")
    pass
