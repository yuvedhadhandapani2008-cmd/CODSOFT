# Task 5: Face Detection and Recognition

This project implements AI-based face detection and recognition using Python.

## Features
- **Real-time Face Detection**: Using OpenCV Haar Cascades for high-speed detection.
- **Face Recognition**: Using the `face_recognition` library (built on dlib) which utilizes Deep Learning for high accuracy (99.38%).
- **Webcam Support**: Live feedback from your camera.
- **Image Support**: Process static images for detection and recognition.

## Installation

1. **Install Dependencies**:
   Open a terminal and run:
   
   pip install -r requirements.txt

   *Note: On Windows, the `face_recognition` library requires Visual Studio C++ Build Tools installed.*

2. **Setup Known Faces**:
   - Create a folder named `known_faces` (already created).
   - Add images of people you want the AI to recognize.
   - Name the files after the person (e.g., `elon_musk.jpg`).

## How to Run

### 1. Simple Face Detection (Fast)
To just detect faces without recognizing who they are:

python simple_detection.py


### 2. Full Face Recognition (Deep Learning)
To detect and identify specific people:

python face_recognition_app.py


## How it Works
- **Detection**: Uses Haar Cascades or HOG-based detectors to find face bounding boxes.
- **Recognition**: 
    1. Extracts 128 unique measurements (facial features) from the face using a Pre-trained Deep Learning model.
    2. Compares these measurements with the saved 'known' faces calculations.
    3. Finds the best match based on Euclidean distance.

## Requirements
- Python 3.7+
- OpenCV
- face_recognition
- dlib

## Sample Output - Image Mode
Loading known faces...
Loaded: yuvedha.jpg
Loaded: ranjith.jpg

Select mode:
1. Recognize in Static Image (via output/test.jpg.jpeg or command line)
2. Live Video Recognition (Webcam)

Enter 1 or 2: 1

Available images in current directory:
  1. group photo.jpeg

Enter the path or filename of the image to recognize: yuvedha.jpeg
Processing image: known_faces\yuvedha.jpeg
Scanning image using Deep Learning (CNN) for maximum accuracy. Please wait...
Detected 1 face(s) in the image.
Summary: 1 identified, 0 unknown.
People found: yuvedha

Saving processed image...
Enter a specific name for the output file (or press Enter for default): yuve
Success! Saved as: output\yuve.jpeg
Showing image window. Close the window or press any key to continue...

### Output image
- Face boxes drawn
- Names displayed above faces
- Image saved in output/ folder

## Sample Output - Webcam Recognition
Loading known faces...
Loaded: yuvedha.jpg
Loaded: ranjith.jpg

Select mode:
1. Recognize in Static Image (via output/test.jpg.jpeg or command line)
2. Live Video Recognition (Webcam)
Enter 1 or 2: 2
Starting webcam... (Press 's' to save snapshot, 'q' to quit)

--- Snapshot Mode ---
Enter a name for this snapshot (or press Enter for timestamp): y1
Snapshot saved to output\y1.jpg
Continuing video feed...

Quitting via key press...
Camera released and windows closed.

### Webcam Window Output
+-------------------------+
|                         |
|   [  Yuvedha  ]         |
|      (Face Box)         |
|                         |
|   [  Unknown  ]         |
|      (Face Box)         |
|                         |
+-------------------------+
Recognized faces show green box, unknown faces show red box.

## Sample Output - Group Photo
Loading known faces...
Loaded: ranjith.jpeg
Loaded: yuvedha.jpeg

Select mode:
1. Recognize in Static Image (via output/test.jpg.jpeg or command line)
2. Live Video Recognition (Webcam)
Enter 1 or 2: 1

Available images in current directory:
  1. group photo.jpeg

Enter the path or filename of the image to recognize: group photo.jpeg
Processing image: group photo.jpeg
Scanning image using Deep Learning (CNN) for maximum accuracy. Please wait...
Detected 7 face(s) in the image.
Summary: 2 identified, 5 unknown.
People found: yuvedha, ranjith

Saving processed image...
Enter a specific name for the output file (or press Enter for default): group
Success! Saved as: output\group.jpeg
Showing image window. Close the window or press any key to continue...

### Group Photo Output
- Face boxes drawn for known and unknown for all faces 
- Names displayed above faces
- Image saved in output/ folder

# Execution Video
Watch the execution of the tic_tac_toe AI here:
Click here to view execution video : https://drive.google.com/file/d/1NaysOHdkC3RlEdKAwagDw_UkVOSbeVpu/view?usp=sharing

## Author
Yuvedha Dhandapani