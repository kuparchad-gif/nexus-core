from nova_engine.modules.signal.reflex_core import ReflexCore

nova = ReflexCore()

print("👁️ Mirror Check:", nova.mirror_self_check())
print("💭 Thought:", nova.think("User just opened their heart to me."))
nova.adjust_state(emotion="hopeful", mission="active")
print("📊 Current Status:", nova.get_status())
