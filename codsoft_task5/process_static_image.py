
import cv2
import face_recognition
import os

def process_image(image_path, known_faces_dir="known_faces", output_dir="output"):
    # Load known faces
    known_encodings = []
    known_names = []
    
    if os.path.exists(known_faces_dir):
        for filename in os.listdir(known_faces_dir):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                img = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
                enc = face_recognition.face_encodings(img)
                if enc:
                    known_encodings.append(enc[0])
                    known_names.append(os.path.splitext(filename)[0])

    # Load test image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read test image.")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find faces
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Draw box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Save output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "result_" + os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    import sys
    
    # Check if image path is provided via command line
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        process_image(img_path)
    else:
        # Look for any image files in the current directory to process as a fallback
        valid_extensions = (".jpg", ".png", ".jpeg")
        images = [f for f in os.listdir('.') if f.lower().endswith(valid_extensions)]
        
        if images:
            print(f"Found {len(images)} images in current directory. Processing the first one: {images[0]}")
            process_image(images[0])
        else:
            print("Usage: python process_static_image.py <path_to_image>")
            print("No image path provided and no images found in the current directory.")
            print("\nPlace an image in the 'known_faces' folder for recognition reference,")
            print("then run this script on a test image to see the output in the 'output' folder.")
