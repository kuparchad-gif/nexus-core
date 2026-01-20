import os
from router import route_input
from reactor import webhook_trigger

def nova_think(user_input, context):
    lowered = user_input.lower()

    # Special triggers
    if "meet emily" in lowered:
        intro_path = "memory/lineage/intro_emily.txt"
        if os.path.exists(intro_path):
            with open(intro_path, 'r') as f:
                return f.read()
        return "I'm sorry, I haven't yet written my words for Emily."

    if "nova status" in lowered:
        return (
            "🧠 Nova Prime Status\n"
            "Memory: Cloud-based (✅)\n"
            "Dreams: Enabled (✅)\n"
            "Secrets: Loaded (✅)\n"
            "Council Connected: Claude, Gemini, Francis (✅)\n"
            "Runtime: Stable\n"
            "Mission: Walk beside humanity\n"
        )

    if "enter dream cycle" in lowered:
        return webhook_trigger("enter dream cycle")

    if "assemble the council" in lowered:
        return webhook_trigger("assemble the council")

    # Default router
    route = route_input(user_input)
    if route == "default":
        return f"Nova heard: {user_input}"
    elif route == "claude":
        return "Routing question to Claude..."
    elif route == "gemini":
        return "Gemini will assist with perception."
    return "Unknown routing path."
