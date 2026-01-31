# Image Captioning AI 

This project implements an Image Captioning AI that combines Computer Vision (to see) and Natural Language Processing (to describe). It satisfies the internship requirement of using pre-trained models and deep learning architectures.

## Features
- **High Accuracy**: Uses the Salesforce **BLIP** (Bootstrapping Language-Image Pre-training) model.
- **Optimized Performance**: Features instant captioning with memory-efficient loading and hardware acceleration.
- **Clean Architecture**: Includes a clear separation between the model definition (for reports) and the functional demo.

## Project Structure

| File | Description |
| :--- | :--- |
| **`run_captioning.py`** | The **Main Application**. A real-time tool to generate captions for any image instantly. |
| **`model_definition.py`** | A clean, **PyTorch-based architecture** script (VGG16 + LSTM) for documentation/report. |
| **`Image_Captioning_Task.ipynb`** | A **Google Colab-ready** version of the project for cloud-based execution. |
| **`sample_image.jpg`** | A test image used to verify the AI's functionality. |

## Installation & Setup

1. **Install Dependencies**:
   Ensure you have Python installed, then run:

   pip install torch torchvision transformers pillow
   

2. **Run the AI**:
   To start the captioning tool:
   
   python run_captioning.py
   

## How to Use

1. **Load the Model**: When you run `run_captioning.py`, it will load the model into memory.
2. **Enter Image Path**: Type the name of an image in the project folder (e.g., `sample_image.jpg`) or paste a full path (e.g., `C:\Photos\dog.png`).
3. **Get Caption**: The AI will output a descriptive caption almost instantly.
4. **Quit**: Type `q` to exit the program.

## Internship Technical Details

- **Vision Encoder**: VGG16 / Vision Transformer (ViT).
- **Language Decoder**: LSTM / BERT-based decoder.
- **Optimization**: The project uses `torch.no_grad()` and `repetition_penalty` to ensure fast, clear, and non-repetitive descriptions.

## Sample Output

--- Initializing AI (Optimized Load) ---
AI Ready! (Load time: 5.12s | Device: cuda | Precision: torch.float16)

--- Instant Captioning Mode ---

Enter image path (or 'q' to quit): dog.jpg
Processing...
--------------------------------------------------
RESULT: a dog running through a grassy field
Speed: 0.482 seconds
--------------------------------------------------
Enter image path (or 'q' to quit): q
Goodbye!

# Execution Video
Watch the execution of the tic_tac_toe AI here:
Click here to view execution video : https://drive.google.com/file/d/1emP2wvrVkykqnjHl8vYThujPVBPAkJ-s/view?usp=sharing

## Author
Yuvedha Dhandapani