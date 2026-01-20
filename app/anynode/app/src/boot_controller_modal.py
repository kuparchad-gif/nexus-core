import time
from bridge_engine_modal import bridge_input

def run_boot():
    print("🧬 VIREN Booting...")
    time.sleep(1)
    print("🔗 Bridge online.")
    test = "Who handles relational logic best?"
    result = bridge_input(test)
    print("🧠 Identity:", result['soul'])
    print("📡 Routing:\n", result['route'])
    print("💬 Response:\n", result['response'])

if __name__ == "__main__":
    run_boot()
