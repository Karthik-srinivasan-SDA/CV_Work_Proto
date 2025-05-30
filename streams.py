'''

from picamera2 import Picamera2
import cv2
import os
import time
import numpy as np

# Setup output directory
output_dir = os.path.expanduser("~/Desktop/Cv_work/output_Stream")
os.makedirs(output_dir, exist_ok=True)

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

# Allow camera to warm up
time.sleep(2)

frame_count = 0

# Define preprocessing function
def preprocess_image(image):
    # Resize to 640x640 (if required by your model)
    image = cv2.resize(image, (640, 640))

    # Apply denoising
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Convert to LAB for CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)

    return image

try:
    while True:
        frame = picam2.capture_array()
        
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Apply preprocessing
        processed_frame = preprocess_image(frame)

        # Display processed frame
        cv2.imshow("Preprocessed PiCam Feed", processed_frame)

        # Save frame
        filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(filename, processed_frame)
        print(f"Saved {filename}")
        
        frame_count += 1

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Frame capture ended.")
'''

from picamera2 import Picamera2
import cv2
import os
import time
import numpy as np

# Setup output directory
output_dir = os.path.expanduser("~/Desktop/Cv_work/output_Stream")
os.makedirs(output_dir, exist_ok=True)

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

# Allow camera to warm up
time.sleep(6)

frame_count = 0

try:
    while True:
        frame = picam2.capture_array()

        # Convert RGB to BGR for OpenCV compatibility
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize to 640x640
        frame = cv2.resize(frame, (640, 640))

        # Denoising with slightly stronger effect
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 15, 15, 7, 21)

        # CLAHE (stronger contrast enhancement)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Sharpening (milder kernel to avoid over-sharpening)
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        frame = cv2.filter2D(frame, -1, kernel)

        # Display processed frame
        cv2.imshow("Preprocessed PiCam Feed", frame)

        # Save frame
        filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

        frame_count += 1

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Frame capture ended.")
