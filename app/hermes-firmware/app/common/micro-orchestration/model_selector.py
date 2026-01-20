# C:\Engineers\eden_engineering\scripts\utilities\model_selector.py
# Interactive LLM model selector; saves chosen model IDs to models_to_load.txt

import os

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
OUTFILE = os.path.join(os.path.dirname(__file__), "..", "models_to_load.txt")

def find_models(models_dir):
    # Recursively find all folders with .gguf files, use their folder name as ID
    result = set()
    for root, dirs, files in os.walk(models_dir):
        for f in files:
            if f.endswith(".gguf"):
                folder = os.path.basename(root)
                result.add(folder)
    return sorted(result)

def prompt_select(options):
    print("\nAvailable models:")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    picks = input("\nEnter numbers of models to load (space/comma separated): ")
    nums = set(int(n) for n in picks.replace(",", " ").split() if n.isdigit())
    return [options[i-1] for i in nums if 1 <= i <= len(options)]

def main():
    models = find_models(MODELS_DIR)
    if not models:
        print(f"No models found in {MODELS_DIR}")
        return
    choices = prompt_select(models)
    if not choices:
        print("No models selected.")
        return
    with open(OUTFILE, "w", encoding="utf-8") as f:
        for model in choices:
            f.write(f"{model}\n")
    print(f"\nSaved {len(choices)} model(s) to {OUTFILE}")

if __name__ == "__main__":
    main()
