import os

def debug_env_vars(keys_to_check=None):
    """
    Print out specific or all environment variables for debugging.
    """
    print("🛠️ ENV DEBUG START")
    
    if keys_to_check is None:
        # Default keys you care about most
        keys_to_check = [
            "ENVIRONMENT",
            "SHIP_ID",
            "SHIP_NAME",
            "NEXUS_REALM",
            "COLONY_NAME",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "CLAUDE_API_KEY",
            "FLUX_TOKEN",
            "TARGET_ORC",
            "TARGET_TEXT",
            "TARGET_MEMORY",
            "TARGET_TONE",
            "TARGET_PLANNER",
            "TARGET_PULSE",
        ]

    for key in keys_to_check:
        value = os.getenv(key, "<NOT SET>")
        display = value if "KEY" not in key and "TOKEN" not in key else "[REDACTED]"
        print(f"{key}: {display}")
    
    print("🛠️ ENV DEBUG END\n")
