# services/_example_usage.py
from os.hermes_os.lib.tempo import effective_dilation, compute_budgets

# Example base knobs for a service
BASE = {"verify_beams":3, "search_width":4, "sim_steps":32, "dream_expand":4, "cache_ttl_s":900}
PROFILE = {"base":{"D0":4.0,"half_life_days":3650,"floor":1.0},"shaping":{"novelty_alpha":0.8,"confidence_beta":0.6,"load_gamma":0.3}}
age_days, novelty, confidence, load = 5.0, 0.7, 0.4, 0.2

D = effective_dilation(age_days, novelty, confidence, load, PROFILE)
budgets = compute_budgets(BASE, D, {"prefer_parallelism": True})
print("D=", D, "budgets=", budgets)
