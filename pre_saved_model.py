from fastapi import FastAPI, Form
import torch
import torch.nn as nn
import uvicorn
from pyngrok import ngrok
import threading
import time

# Initialize FastAPI app
app = FastAPI()

# Define the LSTM Model
class EMGLSTM(nn.Module):
    def __init__(self, input_size=16, glove_size=22, hidden_size=256, output_size=52, num_layers=3, dropout=0.3):
        super(EMGLSTM, self).__init__()
        self.lstm_emg = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.lstm_glove = nn.LSTM(glove_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, emg, glove):
        emg_out, _ = self.lstm_emg(emg)
        glove_out, _ = self.lstm_glove(glove)
        emg_out = emg_out[:, -1, :]
        glove_out = glove_out[:, -1, :]
        combined = torch.cat((emg_out, glove_out), dim=1)
        combined = self.dropout(combined)
        out = self.fc(combined)
        return out

# Load model
model = EMGLSTM()
model.load_state_dict(torch.load("D:/FastApi Implementation/saved_model.pth", map_location=torch.device('cpu')))
model.eval()

# Movement mapping
movement_types = {
    0: "Rest", 1: "Index Flexion", 2: "Index Extension", 3: "Middle Flexion", 4: "Middle Extension",
    5: "Ring Flexion", 6: "Ring Extension", 7: "Pinky Flexion", 8: "Pinky Extension", 9: "Thumb Adduction",
    10: "Thumb Abduction", 11: "Thumb Flexion", 12: "Thumb Extension", 13: "Thumbs Up",
    14: "Extension of index and middle, flexion of others", 15: "Flexion of ring and pinky, extension of others",
    16: "Thumb opposing base of little finger", 17: "Abduction of all fingers", 18: "Fingers flexed together in fist",
    19: "Pointing Index", 20: "Wrist flexion", 21: "Wrist Extension", 22: "Wrist Extension with closed hand",
    23: "Ring Grasp"
}

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

@app.post("/predict")
async def make_prediction(input_size: int = Form(...), glove_size: int = Form(...), time_steps: int = Form(...)):
    emg_input = torch.randn(1, time_steps, input_size)
    glove_input = torch.randn(1, time_steps, glove_size)

    with torch.no_grad():
        output = model(emg_input, glove_input)
        predicted_class = torch.argmax(output, dim=1).item()

    movement_type = movement_types.get(predicted_class, "Unknown Movement")
    return {"predicted_class": predicted_class, "movement_type": movement_type}

# Start server
def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    public_url = ngrok.connect(8000, bind_tls=True).public_url
    print(f"✅ FastAPI server running at: {public_url}")
    print(f"🔍 Swagger docs: {public_url}/docs")

    with open("ngrok_url.txt", "w") as f:
        f.write(public_url)

    # Run FastAPI in a thread
    threading.Thread(target=start_server).start()

    # Keep main thread alive
    while True:
        time.sleep(100)
