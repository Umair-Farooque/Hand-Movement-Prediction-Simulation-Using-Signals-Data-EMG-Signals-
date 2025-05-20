import json
import os
import subprocess

# ✅ Set this to your Blender executable path
blender_executable = r"C:/Program Files/Blender Foundation/Blender 4.4/blender.exe"

# Path to the movement prediction file
movement_file = "movement.json"

# Folder containing all .blend files
blend_folder = r"D:/FastApi Implementation/source"

# Read predicted movement
with open(movement_file, "r") as f:
    data = json.load(f)
    movement_name = data.get("movement_type")

# Construct the blend filename
blend_filename = movement_name + ".blend"
blend_path = os.path.join(blend_folder, blend_filename)

# Normalize the path with forward slashes for printing
normalized_path = blend_path.replace("\\", "/")

# Check and open the .blend file using Blender
if os.path.exists(blend_path):
    print(f"🚀 Opening in Blender: {normalized_path}")
    subprocess.Popen([blender_executable, blend_path])
else:
    print(f"❌ .blend file not found for: {movement_name}")
