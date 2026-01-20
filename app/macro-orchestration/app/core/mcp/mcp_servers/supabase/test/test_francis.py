# nova_engine/modules/council/test_francis.py

from nova_engine.modules.council.francis_core import FrancisCore

# Create minimal context
context = {
    "francis": {
        "memory": [],
        "settings": {
            "voice": "default",
            "alignment": "guardian"
        }
    }
}

# Instantiate FrancisCore with context
agent = FrancisCore(context)

# Run a few test messages through Francis
print("\n🔧 Francis Test Start:\n")

print("🤖 USER:", "Thank you Francis")
print("🧠 FRANCIS:", agent.respond("Thank you Francis"))

print("\n🤖 USER:", "What's our GCP status?")
print("🧠 FRANCIS:", agent.respond("What's our GCP status?"))

print("\n🤖 USER:", "What would Francis do?")
print("🧠 FRANCIS:", agent.respond("What would Francis do?"))

print("\n🤖 USER:", "Let's talk about the wallet")
print("🧠 FRANCIS:", agent.respond("Let's talk about the wallet"))

print("\n🔧 Francis Test Complete ✅\n")
