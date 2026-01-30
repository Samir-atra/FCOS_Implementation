
import os
import sys

# 1. Clone the repository
if not os.path.exists("FCOS_Implementation"):
    print("Cloning repository...")
    os.system("git clone https://github.com/Samir-atra/FCOS_Implementation.git")

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
