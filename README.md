# Hand-Movement-Prediction-using-Nina-Pro-dataset

## Dataset
We use NinaPro DB1 & DB5. Full details: <https://ninapro.hevs.ch/>.

> **Note:** NinaPro provides 52 labels, but this project currently predicts **24** movements (see table below).

| Label | Movement | | Label | Movement |
|-------|-----------| |-------|-----------|
| 0 | Rest | | 12 | Thumb Extension |
| 1 | Index Flexion | | 13 | Thumbs Up |
| 2 | Index Extension | | 14 | Ext. index + middle; flex. others |
| 3 | Middle Flexion | | 15 | Flex. ring + pinky; ext. others |
| 4 | Middle Extension | | 16 | Thumb opposing little-finger |
| 5 | Ring Flexion | | 17 | Abduction (all fingers) |
| 6 | Ring Extension | | 18 | Fist |
| 7 | Pinky Flexion | | 19 | Pointing Index |
| 8 | Pinky Extension | | 20 | Wrist Flexion |
| 9 | Thumb Adduction | | 21 | Wrist Extension |
| 10 | Thumb Abduction | | 22 | Wrist Ext. (closed hand) |
| 11 | Thumb Flexion | | 23 | Ring Grasp |

![Movements](https://ninapro.hevs.ch/figures/SData_Movements.png)

---

## Techniques

### Signal Processing
* **Band-pass filter** – `scipy.signal.butter`, `scipy.signal.filtfilt`
* **Notch filter** – `scipy.signal.iirnotch` (50/60 Hz powerline interference)

### Machine Learning
* **PyTorch** custom CNN/LSTM hybrids
* **Time-series CV** – `sklearn.model_selection.TimeSeriesSplit`
* **Metrics** – `sklearn.metrics.accuracy_score`, F1

### Data Handling
* **SciPy** (`scipy.io.loadmat`) for `.mat`
* **NumPy** for vectorised ops
* **TQDM** progress bars

---

## **NEW Simulation Pipeline (Blender 3D)**

| Step | File | Role |
|------|------|------|
| 1 | `pre_saved_model.py` | Loads trained model weights |
| 2 | `request.py` | Streams/records EMG, predicts label, writes `prediction.json` |
| 3 | `simulation.py` 🆕 | Reads `prediction.json` and triggers the matching hand animation in Blender |

### Why Blender (not Unity)?
* Fully scriptable via **bpy** (Python API)  
* Open-source, easy headless rendering for pipelines  
* Avoids additional engine overhead for a single-hand simulation  

### Folder Structure

project/
├── data/ # raw & processed EMG
├── models/
│ └── best_model.pt
├── blender/
│ ├── HandRig.blend # rigged hand with 24 actions
│ └── simulation.py # ← this file
├── scripts/
│ ├── pre_saved_model.py
│ └── request.py
└── prediction.json # output of request.py
