
# C:\Engineers\eden_engineering\scripts\utilities\launch_console.py

import webbrowser
import os

file_path = r"C:\Engineers\eden_engineering\public\console.html"

if os.path.exists(file_path):
    webbrowser.open(f"file://{file_path}")
else:
    print("❌ File not found:", file_path)
