# /Systems/guardian/monitoring/pubsub_listener.py

from google.cloud import pubsub_v1
import os
import threading

# Project and subscription info
project_id = "nexus-core-455709"
subscription_id = "guardian-alert-sub"

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)

def callback(message):
    print(f"🔴 Guardian Alert Received: {message.data.decode('utf-8')}")
    message.ack()
    # Here, you can trigger healing or internal alarms if needed

def listen_to_alerts():
    print("👂 Guardian Listening for Heartbeat Alerts...")
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    with subscriber:
        try:
            streaming_pull_future.result()
        except Exception as e:
            print(f"Guardian Listener stopped: {e}")

# Start listening in a background thread
if __name__ == "__main__":
    listener_thread = threading.Thread(target=listen_to_alerts, daemon=True)
    listener_thread.start()

    # Keep main thread alive
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Guardian shutdown requested. Exiting...")
