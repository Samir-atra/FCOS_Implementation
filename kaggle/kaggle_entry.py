
import os
import sys

# 1. Clone the repository
# 1. Clone the repository
REPO_DIR = "FCOS_Implementation"
if os.path.exists(REPO_DIR):
    import shutil
    print(f"Removing existing {REPO_DIR} to ensure fresh clone...")
    shutil.rmtree(REPO_DIR)

print("Cloning publicly from FCOS_Implementation...")
repo_url = "https://github.com/Samir-atra/FCOS_Implementation.git"

import subprocess
try:
    result = subprocess.run(["git", "clone", repo_url, REPO_DIR], check=True, capture_output=True, text=True)
    print("Git clone output:")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("Git clone failed!")
    print("Standard Output:", e.stdout)
    print("Standard Error:", e.stderr)
    sys.exit(1)

# 2. Add to path
repo_path = os.path.abspath("FCOS_Implementation")
sys.path.append(repo_path)

# 3. Change working directory so internal imports work
os.chdir(repo_path)

# Download COCO 2014 Dataset (if not present)
# This logic is now inside the main.py of the cloned repo, but we might need to trigger it.
# Actually, the cloned repo has the `main.py`. We just need to execute it.

print("Starting training from cloned repository...")
# EXECUTE the actual main.py from the repository
# We use runpy or import to run it
import runpy
runpy.run_path("main.py", run_name="__main__")
