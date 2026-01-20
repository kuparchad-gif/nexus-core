from ulam_utils import ConsciousnessGraph
graph = ConsciousnessGraph(size=15)  # ~25 primes
paths, energy = graph.vqe_route(center=(7,7), layers=2)  # Or qaoa_route w/ ZNE
# Broadcast paths to ANYNODE (freq 9Hz via Hub)