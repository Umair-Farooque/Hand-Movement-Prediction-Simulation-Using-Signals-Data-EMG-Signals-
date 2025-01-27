from fastapi import FastAPI, Form
import torch
import torch.nn as nn
import uvicorn
from pyngrok import ngrok
import threading

# Initialize FastAPI app
app = FastAPI()

# Define the LSTM Model
class EMGLSTM(nn.Module):
    def __init__(self, input_size=16, glove_size=22, hidden_size=256, output_size=52, num_layers=3, dropout=0.3):
        super(EMGLSTM, self).__init__()
        self.lstm_emg = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.lstm_glove = nn.LSTM(glove_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Concatenate EMG and glove outputs

    def forward(self, emg, glove):
        emg_out, _ = self.lstm_emg(emg)
        glove_out, _ = self.lstm_glove(glove)
        emg_out = emg_out[:, -1, :]
        glove_out = glove_out[:, -1, :]
        combined = torch.cat((emg_out, glove_out), dim=1)
        combined = self.dropout(combined)
        out = self.fc(combined)
        return out

# Load the model
model = EMGLSTM(input_size=16, glove_size=22, hidden_size=256, output_size=52)
model.load_state_dict(torch.load("D:/yo/Lstm_Model.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

# Movement types mapping
movement_types = {
    0: "Rest",
    1: "Index Flexion",
    2: "Index Extension",
    3: "Middle Flexion",
    4: "Middle Extension",
    5: "Ring Flexion",
    6: "Ring Extension",
    7: "Pinky Flexion",
    8: "Pinky Extension",
    9: "Thumb Adduction",
    10: "Thumb Abduction",
    11: "Thumb Flexion",
    12: "Thumb Extension",
    13: "Thumbs Up",
    14: "Extension of index and middle, flexion of others",
    15: "Flexion of ring and pinky, extension of others",
    16: "Thumb opposing base of little finger",
    17: "Abduction of all fingers",
    18: "Fingers flexed together in fist",
    19: "Pointing Index",
    20: "Wrist flexion",
    21: "Wrist Extension",
    22: "Wrist Extension with closed hand",
    23: "Ring Grasp"
    # Add other movements as needed
}

@app.post("/predict")
async def make_prediction(input_size: int = Form(...), glove_size: int = Form(...), time_steps: int = Form(...)):
    emg_input = torch.randn(1, time_steps, input_size)  # EMG data
    glove_input = torch.randn(1, time_steps, glove_size)  # Glove data

    # Perform prediction
    with torch.no_grad():
        output = model(emg_input, glove_input)
        predicted_class = torch.argmax(output, dim=1).item()  # Get predicted class

    # Get corresponding movement type
    movement_type = movement_types.get(predicted_class, "Unknown Movement")

    # Return prediction and movement type
    return {"predicted_class": predicted_class, "movement_type": movement_type}

# Run the app with Uvicorn using ngrok
def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Create a tunnel to expose the FastAPI app to the web
public_url = ngrok.connect(8000)

# Start FastAPI app in a separate thread to allow other code to run
thread = threading.Thread(target=run)
thread.start()

print(f"FastAPI app is running at: {public_url}")
