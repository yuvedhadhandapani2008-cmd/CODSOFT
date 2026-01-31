import cv2

def detect_faces():
    # Load pre-trained Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit.")

    try:
        window_name = 'Face Detection (Haar Cascades)'
        cv2.namedWindow(window_name)

        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            # Convert to grayscale (Haar Cascades work better on grayscale)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Display output
            cv2.imshow(window_name, frame)

            # Check for 'q' key OR if the window was closed via 'X' button
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 'q' or ESC
                print("Quitting via key press...")
                break
            
            # Press 's' to save a snapshot to the output directory
            if key == ord('s'):
                import os
                if not os.path.exists('output'):
                    os.makedirs('output')
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"output/snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Snapshot saved to {filename}")

            # Check if window is still open (works on most platforms)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user. Quitting...")
                break

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # On some systems, destroyAllWindows needs a bit of time or extra waitKey calls to actually close
        for _ in range(10):
            cv2.waitKey(1)
        print("Camera released and windows closed.")


if __name__ == "__main__":
    detect_faces()
