import requests
import os
import time
import json

# Wait for ngrok_url.txt to exist
while not os.path.exists("ngrok_url.txt"):
    print("⏳ Waiting for server to start...")
    time.sleep(2)

# Read URL
with open("ngrok_url.txt", "r") as f:
    url = f.read().strip()

# Send POST request
data = {
    "input_size": 16,
    "glove_size": 22,
    "time_steps": 50
}

try:
    res = requests.post(f"{url}/predict", data=data)
    result = res.json()
    print("✅ Prediction:", result)

    with open("movement.json", "w") as f_out:
        json.dump(result, f_out)

except Exception as e:
    print("❌ Failed to connect or predict:", str(e))
