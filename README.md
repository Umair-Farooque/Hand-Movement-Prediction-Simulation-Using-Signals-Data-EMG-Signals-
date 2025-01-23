# Hand-Movement-Prediction-using-Nina-Pro-dataset
This project is about training a model which predicts hand movements on basis of EMG signals.

# Dataset
We have used dataset of Nina Pro DB1 and DB5 for movements prediction. You can find all the information about the dataset from this website
<https://ninapro.hevs.ch/> <br>
Dataset includes 52 output labels i.e there are total 52 hand movemnts. But in this project we are only predicting 24.
These are the labels along with their corresponding Movement.<br>
| Label | Prediction |
|-------|------------|
| 0     | Rest       |
| 1     | Index Flexion |
| 2     | Index Extension |
| 3     | Middle Flexion |
| 4     | Middle Extension |
| 5     | Ring Flexion |
| 6     | Ring Extension |
| 7     | Pinky Flexion |
| 8     | Pinky Extension |
| 9     | Thumb Adduction |
| 10    | Thumb Abduction |
| 11    | Thumb Flexion |
| 12    | Thumb Extension |
| 13    | Thumbs Up |
| 14    | Extension of index and middle, flexion of others |
| 15    | Flexion of ring and pinky, extension of others |
| 16    | Thumb opposing base of little finger |
| 17    | Abduction of all fingers |
| 18    | Fingers flexed together in fist |
| 19    | Pointing Index |
| 20    | Wrist flexion |
| 21    | Wrist Extension |
| 22    | Wrist Extension with closed hand |
| 23    | Ring Grasp |

![GitHub Logo](https://ninapro.hevs.ch/figures/SData_Movements.png)<br>

# Techniques Used

## Signal Processing

+ ### Band-pass Filtering:
+ Implemented using scipy.signal.butter and scipy.signal.filter.Filters EMG signals within a specified frequency range to remove noise.

+ ### Notch Filtering:
+ Removes powerline interference (e.g., 50 Hz or 60 Hz) using scipy.signal.notch.

## Machine Learning

+ ### PyTorch:
+ Used for creating neural network models.Includes custom datasets and dataloaders for handling EMG data.

+ ### Cross-validation:
+ Time-series cross-validation implemented with sklearn.model_selection.TimeSeriesSplit.

+ ### Performance Metrics:
+ Evaluates model accuracy using sklearn.metrics.accuracy_score.

## Data Handling

+ ### SciPy:
+ Loads .mat files containing EMG data.

+ ### NumPy:
+ Performs numerical operations on the data.

## Efficiency Tools

+ ### TQDM:
+ Displays progress bars for iterative processes.

## Frameworks and Libraries

The project utilizes the following frameworks and libraries:

+ ### Google Colab:
+ For cloud-based computation and easy file access.<br>

+ ### PyTorch:
+ For machine learning and deep learning tasks.<br>

+ ### SciPy:
+ For signal processing.<br>

+ ### NumPy:
+ For efficient numerical computations.<br>

+ ### Scikit-learn:
+ For cross-validation and evaluation metrics.<br>

+ ### TQDM:
+ For progress tracking.<br>

# Contributing

*Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes.*
