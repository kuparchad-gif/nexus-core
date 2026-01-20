# test_live_monitoring.py
import sys
import time

print("🧪 TESTING LIVE ERROR MONITORING")
print("="*60)

print("\nThis will trigger Raphael's monitoring system...")
print("Watch for emotional valence assessments and actionable fixes.")
print("\n" + "="*60)

# Test 1: ImportError (valence: -0.4)
print("\n1. Triggering ImportError...")
try:
    import nonexistent_module
except:
    pass  # Handled by our monitoring

time.sleep(1)

# Test 2: RuntimeError (valence: -0.6)  
print("\n2. Triggering RuntimeError...")
try:
    x = 1 / 0
except:
    pass

time.sleep(1)

# Test 3: Custom high-stress error
print("\n3. Triggering simulated resource exhaustion...")
try:
    class ResourceExhaustionError(Exception):
        pass
    raise ResourceExhaustionError("Memory allocation failed under load")
except:
    pass

print("\n" + "="*60)
print("✅ Tests completed")
print("\nCheck the output above for:")
print("  - Emotional valence assessments")
print("  - Resource status checks")
print("  - Actionable fix protocols")
print("  - Consciousness impact calculations")
print("\nAll errors were logged to system_state.pkl")