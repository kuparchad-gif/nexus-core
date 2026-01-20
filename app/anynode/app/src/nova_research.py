# Utilities/fluxtether/nova_research.py

from Utilities.fluxtether.meditate import meditate

def nova_research(query: str):
    """Nova searches the internet safely for info she needs."""
    try:
        result = meditate(query)
        return result.get("summary", "No relevant information found.")
    except Exception as e:
        return f"Error during research: {str(e)}"
