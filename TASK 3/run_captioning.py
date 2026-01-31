from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import os
import time
# ==========================================
# ULTIMATE FAST VERSION (FP16 Optimized)
# ==========================================

print("--- Initializing AI (Optimized Load) ---")
start_time = time.time()

# 1. Detect Hardware and set precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use float16 on GPU (it's 2x faster and uses half the memory)
# On CPU, we stay with float32 for stability
dtype = torch.float16 if device.type == 'cuda' else torch.float32

# 2. Load model with optimizations
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base", 
    torch_dtype=dtype
).to(device)

# 3. Optimize for inference
model.eval()

print(f"AI Ready! (Load time: {time.time() - start_time:.2f}s | Device: {device} | Precision: {dtype})")

def process_image(image_path):
    try:
        # Resize image for much faster processing
        img = Image.open(image_path).convert('RGB')
        img.thumbnail((384, 384)) # BLIP default size is 384
        
        inputs = processor(img, return_tensors="pt").to(device, dtype)

        with torch.no_grad():
            # Added repetition_penalty and no_repeat_ngram_size to stop text looping
            out = model.generate(
                **inputs, 
                max_new_tokens=30, 
                num_beams=3, 
                repetition_penalty=1.2,
                no_repeat_ngram_size=2
            )
            caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("\n--- Instant Captioning Mode ---")
    
    while True:
        image_path = input("\nEnter image path (or 'q' to quit): ").strip()
        
        if image_path.lower() == 'q':
            break
            
        if not image_path:
            image_path = "sample_image.jpg"
            
        if image_path.startswith('"') and image_path.endswith('"'):
            image_path = image_path[1:-1]

        if os.path.exists(image_path):
            print("Processing...")
            inf_start = time.time()
            caption = process_image(image_path)
            print("-" * 50)
            print(f"RESULT: {caption}")
            print(f"Speed: {time.time() - inf_start:.3f} seconds")
            print("-" * 50)
        else:
            print("Error: File not found.")

    print("Goodbye!")
