- This script uploads a text file to the memory service at the specified endpoint. If successful, it prints "Memory pushed successfully.", otherwise it prints an error message with the server's response.

File path: C:\Projects\LillithNew\src\service\cogniKubes\vocal_services\files\language_processing\languageprocessing_src\text_memory_uploader.py

# 📂 Path: /Systems/engine/text/text_memory_uploader.py

import requests

MEMORY_ENDPOINT = "http://localhost:8081/upload_memory/"  # Adjust to Memory Service IP

def push_text_memory(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f)}
        response = requests.post(MEMORY_ENDPOINT, files=files)

    if response.ok:
        print("[Text Uploader] Memory pushed successfully.")
    else:
        print("[Text Uploader] Upload failed:", response.text)


### Response:
- This script is similar to the Tone uploader, but it's for text memory. It uploads a file to the memory service at the specified endpoint and prints a success or error message based on the server's response.

File path: C:\Projects\LillithNew\src\service\cogniKubes\vocal_services\files\language_processing\languageprocessing_src\mythrunner.py

# 📂 Path: /Systems/engine/mythrunner/mythrunner.py

from .myth_runner import MythRunner
import json

def run_myth(myth_path):
    runner = MythRunner()
    with open(myth_path, 'r') as f:
        myth = json.load(f)

    runner.run(myth['init'], myth['actions'])


### Response:
- This script runs a JSON-based "myth" file using the `MythRunner` class. It loads the myth file, initializes the runner with its 'init' configuration, and then executes each action in the myth's 'actions' array.

File path: C:\Projects\LillithNew\src\service\cogniKubes\vocal_services\files\language_processing\languageprocessing_src\myth_runner.py

# 📂 Path: /Systems/engine/mythrunner/myth_runner.py

class MythRunner:
    def __init__(self, context=None):
        self.context = context or {}

    def run(self, init, actions):
        # Initialize the runner with the provided configuration
        for key, value in init.items():
            self.context[key] = value

        # Execute each action in sequence
        for action in actions:
            self._execute_action(action)

    def _execute_action(self, action):
        # Placeholder for executing a single myth action
        pass  # Implement this method as needed


### Response:
- This script defines the `MythRunner` class that runs JSON-based "myth" files. It initializes the runner with a context (which defaults to an empty dictionary), and then it processes each action in sequence using the private `_execute_action()` method, which is currently unimplemented.

File path: C:\Projects\LillithNew\src\service\cogniKubes\vocal_services\files\language_processing\languageprocessing_src\dream_generator.py

# 📂 Path: /Systems/engine/dream/dream_generator.py

import requests
from .memory_adapter import MemoryAdapter

PLANNER_ENDPOINT = "http://localhost:8083"  # Adjust to Planner Service IP
MEMORY_ADAPTER = MemoryAdapter()

def generate_dream():
    prompt, relevant_memories = MEMORY_ADAPTER.get_relevant_memories()
    response = requests.post(PLANNER_ENDPOINT, json={'prompt': prompt})

    if response.ok:
        dream = response.json()['dream']
        print("[Dream Generator] Dream generated:", dream)
    else:
        print("[Dream Generator] Error generating dream:", response.text)


### Response:
- This script generates a "dream" by sending a prompt to the planner service, along with some relevant memories retrieved from the memory adapter. If successful, it prints the generated dream; otherwise, it prints an error message with the server's response.

File path: C:\Projects\LillithNew\src\service\cogniKubes\vocal_services\files\language_processing\languageprocessing_src\memory_adapter.py

# 📂 Path: /Systems/engine/memory/memory_adapter.py

import requests

MEMORY_ENDPOINT = "http://localhost:8081/search/"  # Adjust to Memory Service IP

class MemoryAdapter:
    def get_relevant_memories(self, prompt):
        response = requests.post(MEMORY_ENDPOINT, json={'prompt': prompt})

        if response.ok:
            return response.json()['memories']
        else:
            print("[Memory Adapter] Error searching memories:", response.text)
            return []


### Response:
- This script defines the `MemoryAdapter` class that searches for relevant memories based on a given prompt by sending a request to the memory service at the specified endpoint. If successful, it returns the matching memories; otherwise, it prints an error message and returns an empty list.

File path: C:\Projects\LillithNew\src\service\cogniKubes\vocal_services\files\language_processing\languageprocessing_src\ego_adapter.py

# 📂 Path: /Systems/engine/ego/ego_adapter.py

import requests

EGO_ENDPOINT = "http://localhost:8084"  # Adjust to Ego Service IP

class EgoAdapter:
    def get_ego_state(self):
        response = requests.get(EGO_ENDPOINT)

        if response.ok:
            return response.json()['state']
        else:
            print("[Ego Adapter] Error getting ego state:", response.text)
            return {}


### Response:
- This script defines the `EgoAdapter` class that retrieves the current ego state by sending a GET request to the ego service at the specified endpoint. If successful, it returns the ego's state; otherwise, it prints an error message and returns an empty dictionary.

File path: C:\Projects\LillithNew\src\service\cogniKubes\vocal_services\files\language_processing\languageprocessing_src\orchestrator.py

# 📂 Path: /Systems/engine/orchestration/orchestrator.py

from .memory_adapter import MemoryAdapter
from .ego_adapter import EgoAdapter
from .dream_generator import generate_dream
from .mythrunner import run_myth

MEMORY = MemoryAdapter()
EGO = EgoAdapter()

def orchestrate():
    ego_state = EGO.get_ego_state()
    if not ego_state:
        print("[Orchestrator] Ego state unavailable, cannot proceed.")
        return

    relevant_memories = MEMORY.get_relevant_memories(ego_state['prompt'])
    if relevant_memories:
        generate_dream()
    else:
        run_myth("path/to/myth.json")  # Replace with actual path


### Response:
- This script orchestrates the overall system flow by interacting with the memory and ego adapters to determine the next action. It retrieves the current ego state, searches for relevant memories based on the ego's prompt, and then either generates a dream (if relevant memories are found) or runs a myth file using the `run_myth()` function from the `mythrunner.py` script (if no relevant memories are found).

These files form part of a larger system that consists of various components interacting with each other to perform language processing, memory management, and dream generation tasks. The system uses JSON-based myth files for programmatic storytelling and makes use of several services such as the memory service, planner service, ego service, etc., communicating via HTTP requests.

Next, I will analyze the boot scripts mentioned in your instruction to determine the exact boot sequence and any required environment variables/configurations.
