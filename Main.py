import sys
import subprocess
import os

# Set UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Define the project directory
project_dir = r"C:\Users\Asus\Desktop\Seam Carving new"

# List of scripts to run in order
scripts = [
    "NewSeamCarving.py",
    "DetectFeatures.py",
    "FeatureMatching.py",
    "EstimateHomography.py",
    "ImageWarping.py",
    "ImageStitching.py"
]

# Iterate and run each script sequentially
for script in scripts:
    script_path = os.path.join(project_dir, script)
    if os.path.exists(script_path):
        print(f"Running {script}...")
        result = subprocess.run(["python", script_path], capture_output=True, text=True, encoding="utf-8")
        print(result.stdout)
        print(result.stderr)
    else:
        print(f"Error: {script} not found in {project_dir}")

print("All scripts executed successfully!")
