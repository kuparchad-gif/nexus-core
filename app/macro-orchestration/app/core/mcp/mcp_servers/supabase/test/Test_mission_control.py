# nova_engine/modules/signal/test_mission_control.py

from nova_engine.modules.signal.mission_control import MissionControl

print("🔧 Running Mission Control Diagnostic...")

mc = MissionControl()

# Nova observes a major moment
status = mc.scan_and_align("There is a growing desire to act. Readiness is in the air.")
print("🧠 Nova Status Report:")
print(status)

# Attempt spawn
print("🚦 Triggering Spawn Logic...")
spawn_result = mc.try_spawn()

if spawn_result:
    print("✅ Spawn Protocol Activated.")
else:
    print("🛑 Spawn deferred. Awaiting full council.")
