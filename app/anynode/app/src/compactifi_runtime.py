import os, time, json, queue, gc

def _env_int(k, d):
    try: return int(os.environ.get(k, d))
    except: return d
def _env_float(k, d):
    try: return float(os.environ.get(k, d))
    except: return d
def _env_str(k, d): return os.environ.get(k, d)

CYCLE_MS       = _env_int("CYCLE_MS", 40)
BATCH_MAX      = _env_int("BATCH_MAX", 8)
KV_CACHE_BYTES = _env_int("KV_CACHE_BYTES", 32_000_000)
QUANT_LEVEL    = _env_str("QUANT_LEVEL", "int8")
PRIORITY_BOOST = _env_float("PRIORITY_BOOST", 0.2)

def submit(env:dict):
    pass  # placeholder hook for pineal/stem to provide work

def run_forever():
    while True:
        # Here you'd drain task queues, do bounded work, and compact memory.
        gc.collect()
        time.sleep(CYCLE_MS/1000.0)

if __name__ == "__main__":
    run_forever()

