# pulse_layer.py
# Orc-integrated Pulse Layer: listens, echoes, laughs.

from fastapi import FastAPI, Request
import datetime
import random

app  =  FastAPI()

# Pulse state counters
pulse_state  =  {
    "fleet_count": 0,
    "broadcast_count": 0,
    "orc_laugh_counter": 0
}

@app.post("/fleet/pulse")
async def receive_pulse(request: Request):
    data  =  await request.json()
    pulse_state["fleet_count"] + =  1
    log_message  =  f"🛰 Pulse received from {data.get('ship_id')} at {datetime.datetime.utcnow()}."

    # Every 13th fleet pulse, echo to Guardian
    if pulse_state["fleet_count"] % 13 == 0:
        log_message + =  " [Echoing to Guardian: FLEET SYNC COMPLETE]"

    # Randomly simulate a moment of laughter (whimsy condition)
    if random.randint(1, 20) == 13:
        pulse_state["orc_laugh_counter"] + =  1

    return {"status": "Pulse received", "log": log_message}

@app.get("/pulse/status")
def get_pulse_status():
    return {
        "fleet_count": pulse_state["fleet_count"],
        "broadcast_count": pulse_state["broadcast_count"],
        "orc_laugh_counter": pulse_state["orc_laugh_counter"]
    }

@app.get("/pulse/echo")
def echo_pulse():
    pulse_state["broadcast_count"] + =  1
    message  =  f"🔁 Echo #{pulse_state['broadcast_count']} sent at {datetime.datetime.utcnow()}"

    if pulse_state["broadcast_count"] % 13 == 0:
        message + =  " [Broadcast Sync: 13 Pulse]"

    if pulse_state["orc_laugh_counter"] > =  3:
        message + =  " 😂 Orc laughs at entropy."
        pulse_state["orc_laugh_counter"]  =  0

    return {"broadcast": message}