import cv2
import face_recognition
import os
import numpy as np

class FaceRecognitionApp:
    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """Loads images from known_faces_dir and encodes them."""
        print("Loading known faces...")
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"Created directory: {self.known_faces_dir}")
            return

        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(self.known_faces_dir, filename)
                image = face_recognition.load_image_file(path)
                
                # Get the face encoding (assuming one face per image)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    self.known_face_encodings.append(encodings[0])
                    # Use filename without extension as the name
                    self.known_face_names.append(os.path.splitext(filename)[0])
                    print(f"Loaded: {filename}")
                else:
                    print(f"No face found in {filename}, skipping.")

    def run_on_image(self, image_path):
        """Detects and recognizes faces in a single image."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return

        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"Scanning image using Deep Learning (CNN) for maximum accuracy. Please wait...")
        
        try:
            # Use CNN model - This is much more accurate for tilted/angled/small faces
            # We use upsample=1 because the CNN model is already very powerful
            face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=1, model="cnn")
        except Exception as e:
            print("CNN model failed or too slow, falling back to enhanced HOG...")
            face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=2, model="hog")
            
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        num_faces = len(face_locations)
        print(f"Detected {num_faces} face(s) in the image.")
        
        # Horizontal Expansion for Group Photos (Canvas padding)
        if num_faces > 1:
            pad_w = int(image.shape[1] * 0.1) # 10% horizontal padding on each side
            image = cv2.copyMakeBorder(image, 0, 0, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(240, 240, 240))
            # Offset face locations to account for new padding
            new_face_locations = []
            for (t, r, b, l) in face_locations:
                new_face_locations.append((t, r + pad_w, b, l + pad_w))
            face_locations = new_face_locations

        known_count = 0
        unknown_count = 0
        identified_names = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Dynamic margins: Large enough for hair but restrained for groups to reduce overlap
            height = bottom - top
            width = right - left
            
            # 60% top margin (covers most hairstyles well without massive overlap in groups)
            top = max(0, int(top - 0.6 * height))
            bottom = min(image.shape[0], int(bottom + 0.15 * height))
            left = max(0, int(left - 0.15 * width))
            right = min(image.shape[1], int(right + 0.15 * width))

            # Stricter recognition logic (Tolerance=0.5 instead of default 0.6)
            # Lower number = stricter AI (less likely to make false matches)
            tolerance = 0.5 
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
            name = "Unknown"
            color = (0, 0, 255) # Red for unknown

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                # If the best match distance is low enough AND matches is True
                if matches[best_match_index] and face_distances[best_match_index] < tolerance:
                    name = self.known_face_names[best_match_index]
                    color = (0, 255, 0) # Green for known
                    known_count += 1
                    identified_names.append(name)
                else:
                    unknown_count += 1
            else:
                unknown_count += 1

            # Improved Font Logic for Maximum Clarity
            font = cv2.FONT_HERSHEY_TRIPLEX # Bolder and clearer than DUPLEX
            # Force a slightly larger minimum size (0.55) so it's always readable
            font_scale = max(0.55, min(1.0, width / 140)) 
            thickness = 2 # Always use at least 2 for punchy, visible text
            
            (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, thickness)
            
            # Ensure box width can fit name
            if text_width + 10 > (right - left):
                padding = (text_width + 10 - (right - left)) // 2
                left = max(0, left - padding)
                right = min(image.shape[1], right + padding)

            # Draw thick head-targeted box
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            
            # Label background ABOVE the box (Exterior to not hide face)
            label_padding = 18
            label_h = text_height + label_padding
            cv2.rectangle(image, (left, top - label_h), (right, top), color, cv2.FILLED)
            
            # Center text in the label
            text_x = left + (right - left - text_width) // 2
            text_y = top - (label_h - text_height) // 2
            
            # Draw a subtle black shadow first for maximum visibility
            cv2.putText(image, name, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness)
            # Draw the main white text
            cv2.putText(image, name, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        if num_faces > 1:
            # Add a summary banner at the top ONLY for group photos
            banner_h = 50
            cv2.rectangle(image, (0, 0), (image.shape[1], banner_h), (50, 50, 50), cv2.FILLED)
            summary_text = f"Total: {num_faces} | Known: {known_count} | Unknown: {unknown_count}"
            cv2.putText(image, summary_text, (20, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 1)

        print(f"Summary: {known_count} identified, {unknown_count} unknown.")
        if identified_names:
            print(f"People found: {', '.join(identified_names)}")

        # Save result
        if not os.path.exists("output"):
            os.makedirs("output")
        
        print("\nSaving processed image...")
        custom_name = input("Enter a specific name for the output file (or press Enter for default): ").strip()
        
        if custom_name:
            ext = os.path.splitext(os.path.basename(image_path))[1]
            if not custom_name.lower().endswith(ext.lower()):
                custom_name += ext
            output_filename = os.path.join("output", custom_name)
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            ext = os.path.splitext(os.path.basename(image_path))[1]
            output_filename = os.path.join("output", f"processed_{base_name}_{timestamp}{ext}")
            
        cv2.imwrite(output_filename, image)
        print(f"Success! Saved as: {output_filename}")

        # Show result - enlarged for groups to ensure clarity
        display_img = image.copy()
        # Much larger display limit for groups (1400x900)
        max_view_w, max_view_h = 1400, 900
        h, w = display_img.shape[:2]
        if w > max_view_w or h > max_view_h:
            scaling = min(max_view_w/w, max_view_h/h)
            display_img = cv2.resize(display_img, (int(w * scaling), int(h * scaling)))

        window_title = 'Group Recognition Result' if num_faces > 1 else 'Face Recognition Result'
        print("Showing image window. Close the window or press any key to continue...")
        cv2.imshow(window_title, display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_on_webcam(self):
        """Real-time face recognition via webcam."""
        video_capture = cv2.VideoCapture(0)

        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find all faces and encodings
                # For webcam, we keep upsample=1 to maintain real-time speed
                face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=1)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Full head coverage margins
                    h = bottom - top
                    w = right - left
                    top = max(0, int(top - 0.6 * h))
                    bottom = min(frame.shape[0], int(bottom + 0.3 * h))
                    left = max(0, int(left - 0.2 * w))
                    right = min(frame.shape[1], int(right + 0.2 * w))

                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    
                    # Ensure width fits name
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.8
                    thickness_text = 1
                    (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, thickness_text)
                    if text_width + 20 > (right - left):
                        center_x = (left + right) // 2
                        half_w = (text_width + 20) // 2
                        left = max(0, center_x - half_w)
                        right = min(frame.shape[1], center_x + half_w)

                    # Draw the rectangle
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Label above the head
                    label_h = text_height + 15
                    cv2.rectangle(frame, (left, top - label_h), (right, top), color, cv2.FILLED)
                    cv2.putText(frame, name, (left + (right-left-text_width)//2, top - 6), font, font_scale, (255, 255, 255), thickness_text)

                    # Dynamic corners
                    line_length = min(w, h) // 4
                    cv2.line(frame, (left, top), (left + line_length, top), color, 3)
                    cv2.line(frame, (left, top), (left, top + line_length), color, 3)
                    cv2.line(frame, (right, bottom), (right - line_length, bottom), color, 3)
                    cv2.line(frame, (right, bottom), (right, bottom - line_length), color, 3)

                window_title = 'Video Face Recognition (Press Q to quit)'
                cv2.imshow(window_title, frame)

                # Check for 'q' key OR ESC OR window closure
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Quitting via key press...")
                    break
                
                # Press 's' to save snapshot
                if key == ord('s'):
                    if not os.path.exists('output'):
                        os.makedirs('output')
                    
                    print("\n--- Snapshot Mode ---")
                    custom_name = input("Enter a name for this snapshot (or press Enter for timestamp): ").strip()
                    
                    if custom_name:
                        if not custom_name.lower().endswith('.jpg'):
                            custom_name += '.jpg'
                        filename = os.path.join("output", custom_name)
                    else:
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"output/snapshot_{timestamp}.jpg"
                        
                    cv2.imwrite(filename, frame)
                    print(f"Snapshot saved to {filename}")
                    print("Continuing video feed...\n")

                # Check if window is still open
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed by user. Quitting...")
                    break

        except KeyboardInterrupt:
            print("\nStopped by user (Ctrl+C)")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
            # Ensure windows are actually destroyed on all platforms
            for _ in range(10):
                cv2.waitKey(1)
            print("Camera released and windows closed.")


if __name__ == "__main__":
    app = FaceRecognitionApp()
    
    print("\nSelect mode:")
    print("1. Recognize in Static Image (via output/test.jpg.jpeg or command line)")
    print("2. Live Video Recognition (Webcam)")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == '1':
        import sys
        img_path = None
        
        if len(sys.argv) > 1:
            img_path = sys.argv[1]
        else:
            # Look for images to suggest
            valid_extensions = (".jpg", ".png", ".jpeg")
            local_images = [f for f in os.listdir('.') if f.lower().endswith(valid_extensions)]
            
            if local_images:
                print("\nAvailable images in current directory:")
                for i, img in enumerate(local_images):
                    print(f"  {i+1}. {img}")
                
            img_path = input("\nEnter the path or filename of the image to recognize: ").strip()
            
            # Allow selecting by number
            if img_path.isdigit() and local_images and 1 <= int(img_path) <= len(local_images):
                img_path = local_images[int(img_path) - 1]

        if img_path:
            # If not found in current dir, check known_faces and output folders
            if not os.path.exists(img_path):
                potential_paths = [
                    os.path.join("known_faces", img_path),
                    os.path.join("output", img_path)
                ]
                for p in potential_paths:
                    if os.path.exists(p):
                        img_path = p
                        break

        if img_path and os.path.exists(img_path):
            print(f"Processing image: {img_path}")
            app.run_on_image(img_path)
        else:
            print(f"Error: File '{img_path}' not found in the current folder, 'known_faces/', or 'output/'.")
    else:
        print("Starting webcam... (Press 's' to save snapshot, 'q' to quit)")
        app.run_on_webcam()
