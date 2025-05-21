# Hand‑Movement Prediction Using NinaPro Dataset

A complete pipeline for **EMG‑based hand‑movement recognition** (24 gestures) **and real‑time 3‑D visualisation in Blender**.

---

## Dataset

We use **NinaPro DB‑1** and **DB‑5**&nbsp;⇢ <https://ninapro.hevs.ch/>.

> NinaPro defines 52 gestures; **this repo currently targets 24** (listed below).

### Movement‑label map

| Label | Movement | Label | Movement |
|:----:|-------------------------------|:----:|----------------------------------------------|
| 0  | Rest                                          | 12 | Thumb Extension |
| 1  | Index Flexion                                | 13 | Thumbs Up |
| 2  | Index Extension                              | 14 | Extension of index + middle; flex others |
| 3  | Middle Flexion                               | 15 | Flex ring + pinky; extend others |
| 4  | Middle Extension                             | 16 | Thumb opposing base of little finger |
| 5  | Ring Flexion                                 | 17 | Abduction of all fingers |
| 6  | Ring Extension                               | 18 | Fist (all fingers flexed together) |
| 7  | Pinky Flexion                                | 19 | Pointing Index |
| 8  | Pinky Extension                              | 20 | Wrist Flexion |
| 9  | Thumb Adduction                              | 21 | Wrist Extension |
| 10 | Thumb Abduction                              | 22 | Wrist Extension with closed hand |
| 11 | Thumb Flexion                                | 23 | Ring Grasp |

![NinaPro gestures](https://ninapro.hevs.ch/figures/SData_Movements.png)

---

## Techniques

### Signal Processing (`scipy`)
* **Band‑pass filter** – `signal.butter` + `signal.filtfilt`
* **Notch filter** – `signal.iirnotch` (50/60 Hz mains)

### Machine Learning (PyTorch + sklearn)
* CNN / LSTM hybrids
* **Time‑series cross‑validation** – `TimeSeriesSplit`
* **Metrics** – accuracy, macro‑F1

### Data Handling
* **SciPy I/O** for `.mat`
* **NumPy** vectorisation
* **TQDM** progress bars

---

## 🔧 Simulation Pipeline (Blender 3D)

| # | Script | Purpose |
|:-:|--------|---------|
| 1 | `pre_saved_model.py` | Load pretrained weights |
| 2 | `request.py` | Stream/record EMG → predict label → write `prediction.json` |
| 3 | `blender/simulation.py` | Read JSON → play matching **Action** on rigged hand |

<details>
<summary><strong>Why Blender (instead of Unity)?</strong></summary>

* Fully scriptable via Python API (`bpy`)
* Easy headless rendering/CI integration
* Lightweight for a single‑hand scene
</details>

### Folder structure

```text
project/
├── data/                     # raw & processed EMG
├── models/
│   └── best_model.pt
├── blender/
│   ├── HandRig.blend         # rigged hand with 24 actions
│   └── simulation.py
├── scripts/
│   ├── pre_saved_model.py
│   └── request.py
└── prediction.json           # generated each inference cycle
```

### Quick‑start

```bash
# 1. Predict from incoming EMG and save result
python scripts/request.py --model models/best_model.pt --out prediction.json

# 2. Animate inside Blender
blender blender/HandRig.blend --background        --python blender/simulation.py -- --json prediction.json --render
```

> Add `--render` to export an MP4; omit it for live viewport playback.

---

## Known Issues & Troubleshooting

| Symptom | Likely Cause / Fix |
|---------|--------------------|
| **❌  `Armature 'HandRig' not found`** | The armature in `HandRig.blend` must be named **exactly** `HandRig`. |
| **❌  `Action '<name>' not found`** | Ensure each of the 24 Blender Actions matches the movement names **verbatim** (see table above). |
| **Blender can’t import `bpy`** | Run the script *inside* Blender (`blender … --python …`). |
| **Python module errors** | Install requirements: `pip install -r requirements.txt` (SciPy, NumPy, PyTorch, scikit‑learn, tqdm). |
| **Lag during live streaming** | Use a lower EMG sampling rate or batch predictions. |
| **Render is blank/black** | Add a Camera & light, or switch viewport render engine to Eevee/Cycles. |

---

## Contributing

1. **Fork** the repo  
2. **Create** a feature branch  
3. **Commit** clear, atomic changes  
4. **Open** a PR – please describe *what* & *why*  

We welcome PRs for:
* Additional NinaPro gestures (25–52)
* Real‑time OSC / WebSocket streaming
* Rig/animation improvements
* Performance profiling & quantisation

---

*Happy coding & animating! 🎉*
