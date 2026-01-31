# Face Recognition System using DeepFace (ArcFace)

## Project Overview
This project is a **Face Recognition System** built using **Python**, **DeepFace**, and **TensorFlow**.  
It compares a test image with known face images and identifies whether the person is already present in the system.

The project uses the **ArcFace** model for face recognition and **OpenCV** as the face detector to ensure compatibility and stability.

## Objective
- Detect and recognize faces from images
- Compare a test image with stored known faces
- Display whether a match is found or not

## Technologies Used
- Python 3.10
- TensorFlow 2.10.0
- DeepFace
- Keras
- NumPy
- OpenCV
- Anaconda (Environment Management)

## Project Structure

- `known_faces/` → Contains images of known individuals  
- `test.jpg` → Image to be recognized  
- `face_recognition.py` → Main Python program  

## Environment Setup

### Step 1: Create and Activate Conda Environment

conda create -n tfclean python=3.10 -y
conda activate tfclean

### Step 2: Install Required Packages

pip install numpy==1.23.5
pip install tensorflow-cpu==2.10.0
pip install keras==2.10.0
pip install opencv-python==4.8.1.78
pip install deepface==0.0.79

### Step 3: Verify Installation

python -c "import numpy as np; print(np.__version__)"
python -c "import tensorflow as tf; print(tf.__version__)"

### Expected Output:

1.23.5
2.10.0

## How to Run the Project

### Step 1: Activate the environment

conda activate tfclean

### Step 2: Navigate to the project folder

cd "C:\Users\HP\OneDrive\Desktop\internship codsoft\TASK 5"

### Step 3: Run the program

python face_recognition.py

## Sample Output

### If a match is found:
Loading known faces...
Known people: ['person1', 'person2']

Analyzing test image...
✅ Match found: person1

### If no match is found:
❌ No match found

## Execution Video
Watch the execution of the tic_tac_toe AI here:
Click here to view execution video : https://drive.google.com/file/d/188Z1jHuZxb6s7tRMR9kl0MRKhvPD3gxs/view?usp=sharing

## Author
Yuvedha Dhandapani
