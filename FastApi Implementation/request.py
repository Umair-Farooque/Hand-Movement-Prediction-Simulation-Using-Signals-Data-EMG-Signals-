import requests

url = "https://8298-2407-d000-b-eb63-d17f-4a6d-a03-6611.ngrok-free.app//predict"  # Replace with your actual tunnel URL
data = {
    'input_size': 16,
    'glove_size': 22,
    'time_steps': 100
}

response = requests.post(url, data=data)

print(response.json())
