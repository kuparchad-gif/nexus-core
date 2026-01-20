import subprocess, sys, os, time, threading

def run_streamlit():
    subprocess.Popen([sys.executable, "-m", "streamlit", "run", "src/monitor/dashboard.py"])

def run_api():
    subprocess.Popen([sys.executable, "-m", "uvicorn", "api.server:app", "--host", "127.0.0.1", "--port", "8089"])

if __name__ == "__main__":
    print("Starting Aethereal Trader Deluxe Pack...")
    run_api()
    time.sleep(1)
    run_streamlit()
    print("API on http://127.0.0.1:8089  |  Dashboard will open via Streamlit.")
    print("Press Ctrl+C in each window to stop.")
