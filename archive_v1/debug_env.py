import sys
import subprocess
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"CWD: {os.getcwd()}")

print("\nAttempting to import dotenv...")
try:
    import dotenv
    print(f"Success! dotenv file: {dotenv.__file__}")
except ImportError as e:
    print(f"Import failed: {e}")

print("\n--- Pip List ---")
try:
    subprocess.run([sys.executable, "-m", "pip", "list"])
except Exception as e:
    print(f"Could not run pip list: {e}")
