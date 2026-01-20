# check_actual_module.py
import sys
import os

print("🔍 CHECKING ACTUAL RAPHAEL MODULE")
print("="*60)

# Check file location
import raphael_complete
print(f"Module location: {raphael_complete.__file__}")
print(f"File size: {os.path.getsize(raphael_complete.__file__)} bytes")

# Check if it's the same file
expected_path = os.path.join(os.getcwd(), 'raphael_complete.py')
actual_path = os.path.abspath(raphael_complete.__file__)

print(f"Expected: {expected_path}")
print(f"Actual:   {actual_path}")
print(f"Same file? {expected_path == actual_path}")

# Read the actual file being imported
print(f"\n📄 First 50 lines of ACTUAL imported file:")
try:
    with open(actual_path, 'r') as f:
        for i in range(50):
            line = f.readline()
            if not line:
                break
            print(f"{i+1:3}: {line.rstrip()}")
except Exception as e:
    print(f"Error reading: {e}")

# Check for record_error in actual file
print(f"\n🔎 Searching for 'record_error' in actual file...")
try:
    with open(actual_path, 'r') as f:
        content = f.read()
        if 'def record_error' in content:
            print("✅ 'def record_error' FOUND in actual file")
            # Find line number
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'def record_error' in line:
                    print(f"   Line {i+1}: {line.strip()}")
                    break
        else:
            print("❌ 'def record_error' NOT in actual file")
except Exception as e:
    print(f"Error: {e}")