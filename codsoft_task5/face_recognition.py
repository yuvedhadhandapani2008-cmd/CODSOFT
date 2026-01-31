from deepface import DeepFace
import os

KNOWN_FACES_DIR = "known_faces"
TEST_IMAGE = "test.jpg.jpeg"

print("Loading known faces...")

known_faces = []
known_names = []

# Load known face images
for file in os.listdir(KNOWN_FACES_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(KNOWN_FACES_DIR, file)
        known_faces.append(path)
        known_names.append(os.path.splitext(file)[0])

print("Known people:", known_names)

print("\nAnalyzing test image...")

match_found = False

for face, name in zip(known_faces, known_names):
    try:
        result = DeepFace.verify(
            img1_path=face,
            img2_path=TEST_IMAGE,
            model_name="ArcFace",
            detector_backend="opencv",   # ✅ FIX for tf_keras error
            enforce_detection=False
        )

        if result["verified"]:
            print(f"✅ Match found: {name}")
            match_found = True
            break

    except Exception as e:
        print(f"⚠️ Skipping {name} due to error")

if not match_found:
    print("❌ No match found")

