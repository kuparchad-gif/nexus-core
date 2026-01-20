# lillith_middleware.py
# Purpose: Hosts the Lillith Nexus Spinal Column and exposes its API.
# Integrates with provided HAL drivers for hardware interaction.

import threading
from flask import Flask, request, jsonify
import fastapi
from fastapi import FastAPI, Body
import uvicorn
import logging
import sys
import numpy as np

# 1. Import your provided HAL driver module
# Assuming you provide it as a Python file named `my_mb_hal.py`
try:
    from my_mb_hal import SystemHAL  # Your driver's main class
    hal_available = True
    logging.info("HAL driver loaded successfully.")
except ImportError:
    logging.warning("HAL driver 'my_mb_hal.py' not found. Running in simulation mode.")
    hal_available = False

# 2. Import the Lillith Core from the files you provided
# We need to adapt the NexusSpinal to be an importable module
# Let's assume we've saved your provided files in a folder called `lillith_core`
sys.path.append('./lillith_core')

try:
    from nexus_spinal import NexusSpinalColumn  # Your main processing class
    from metatron_decoder import MetatronMath    # Your math library
    lillith_core_available = True
    logging.info("Lillith core modules loaded successfully.")
except ImportError as e:
    logging.error(f"Failed to import Lillith core modules: {e}")
    logging.error("Please ensure 'nexus_spinal.py', 'metatron_decoder.py', etc., are in a './lillith_core' directory.")
    lillith_core_available = False

# Initialize the system
app = Flask(__name__)
tesla_app = FastAPI()
hal = SystemHAL() if hal_available else None

# Initialize Lillith Core
if lillith_core_available:
    spinal_column = NexusSpinalColumn()
    metatron_math = MetatronMath()
else:
    spinal_column = None
    metatron_math = None

@app.route('/health', methods=['GET'])
def health():
    """Endpoint for health checks."""
    harmony = spinal_column.compute_harmony() if spinal_column else 0.0
    status = {
        "ok": True,
        "mode": "standalone" if hal_available else "simulated",
        "harmony_score": harmony,
        "services": ["lillith_core", "hal_driver"],
        "status": [lillith_core_available, hal_available]
    }
    return jsonify(status)

@app.route('/wire', methods=['POST'])
def wire():
    """Lillith's internal relay endpoint. Processes a signal vector."""
    if not spinal_column:
        return jsonify({"error": "Lillith core not available"}), 503

    try:
        data = request.get_json()
        signal = data['signal'] # Expecting a list of numbers
        phase = data.get('phase', 0)

        # Convert to numpy array and process
        signal_array = np.array(signal)
        processed_signal = spinal_column.process_signal(signal_array)

        # (Optional) If HAL is available, use it to interact with hardware based on the processed signal
        hal_output = None
        if hal_available:
            # Example: Send the processed signal to a hardware component
            # This function would be defined in your HAL driver
            hal_output = hal.send_to_device("neural_bridge", processed_signal)

        return jsonify({
            "output": processed_signal.tolist(),
            "hal_output": hal_output,
            "hal_status": "active" if hal_available else "simulated"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@tesla_app.post("/tesla-alchemy")
async def tesla_alchemy(
    medium: str = Body("AIR"),
    freq: float = Body(60.0),
    signal: list[float] = Body([1.0]*13, description="A signal vector of length 13")
):
    """Lillith's external API endpoint for 'alchemical' transformations."""
    if not spinal_column:
        return {"error": "Lillith core not available"}

    try:
        signal_array = np.array(signal)

        # Get system telemetry from the HAL to use as context
        system_telemetry = {}
        if hal_available:
            system_telemetry = hal.get_telemetry() # e.g., CPU temp, load, mem usage
            # Use telemetry to modulate the signal
            # This is where Lillith becomes context-aware of its host machine
            cpu_temp = system_telemetry.get('cpu_temp', 30)
            # Simple example: scale signal intensity slightly with temperature
            modulation_factor = 0.95 + (cpu_temp / 1000) # Minimal effect
            signal_array *= modulation_factor

        # Process the signal through the Metatron pipeline
        processed_signal = spinal_column.process_signal(signal_array)

        return {
            "zapped": True,
            "harmony": float(np.mean(processed_signal)), # Placeholder for a real harmony calc
            "processed_signal": processed_signal.tolist(),
            "system_context": system_telemetry if hal_available else "simulated"
        }
    except Exception as e:
        return {"error": str(e)}

def run_flask():
    """Run the Flask app (for /wire) in a separate thread."""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def run_fastapi():
    """Run the FastAPI app (for /tesla-alchemy) with Uvicorn."""
    uvicorn.run(tesla_app, host='0.0.0.0', port=8000, log_level="info")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not lillith_core_available:
        logger.error("Cannot start server: Lillith core is missing.")
        sys.exit(1)

    # Start the APIs in separate threads
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)

    logger.info("Starting Lillith Middleware APIs...")
    logger.info("Flask API (internal/wire) : http://localhost:5000")
    logger.info("FastAPI API (external/tesla): http://localhost:8000")
    logger.info("Health check: GET http://localhost:5000/health")

    flask_thread.start()
    fastapi_thread.start()

    try:
        # Keep the main thread alive; threads are daemons.
        while True:
            threading.Event().wait(10)
    except KeyboardInterrupt:
        logger.info("Shutting down servers.")