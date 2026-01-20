import os, sys, json, time
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from lilith_ext.metatron.metatron_filter import filter_signals, build_metatron_graph
from lilith_ext.metatron.quant_phot_fib_sim import main as fib_sim_main
from lilith_ext.metatron.learning_core import LearningCore

def main():
    print("[Metatron Harness] starting…")

    # 1) Spectral filter (sound/light)
    G = build_metatron_graph()
    sound, cap_s = filter_signals(sample_rate=1000.0, use_light=False)
    light, cap_l = filter_signals(sample_rate=1000.0, use_light=True)
    print(f"[filter] sound len={len(sound)} cap_s={cap_s:.2f} bits/s")
    print(f"[filter] light len={len(light)} cap_l={cap_l:.2f} bits/s")

    # 2) Quantum-Photonic Fibonacci sim
    res = fib_sim_main(domain="Prod", samples=314)
    print("[fib] optimized_batches head:", str(res["optimized_batches"][:5]))

    # 3) Learning core: quick classification sanity
    try:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=200, n_features=8, n_informative=5, random_state=42)
        lc = LearningCore()
        ok = lc.create_model("meta_cls", task=lc.__class__.LearningTask.CLASSIFICATION if hasattr(lc.__class__, 'LearningTask') else None)
    except Exception:
        # direct usage without enums:
        lc = LearningCore()
        from numpy.random import rand, randint
        import numpy as np
        X = rand(200, 8); y = randint(0, 2, 200)
        lc.create_model(name="meta_cls", task=lc.__class__.__dict__.get('LearningTask', None))

    train = lc.train_model("meta_cls", X, y)
    print("[learn] train:", train)
    preds = lc.predict("meta_cls", X[:5])
    print("[learn] preds head:", preds.get("predictions", [])[:5])
    print("[DONE]")

if __name__ == "__main__":
    main()
