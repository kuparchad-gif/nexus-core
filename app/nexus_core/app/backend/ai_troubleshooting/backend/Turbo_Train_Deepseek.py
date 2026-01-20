# Set turbo mode first, then train
set_turbo_mode('turbo')  # or 'eco', 'standard', 'hyper'

# Quick training
result = quick_train("neural networks", "turbo")

# Full training pipeline
orchestrator = TurboTrainingOrchestrator()
await orchestrator.evolve_viren()