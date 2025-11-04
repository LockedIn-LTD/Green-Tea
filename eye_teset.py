from transformers import pipeline
import cv2
import numpy as np
from PIL import Image  # Import PIL for image conversion



def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


# Load the pre-trained eye state classification model from Hugging Face
eye_state_pipeline = pipeline(model="MichalMlodawski/open-closed-eye-classification-mobilev2")

# Define label mapping (replace with actual labels from the model if available)
label_mapping = {
    "LABEL_0": "Closed",
    "LABEL_1": "Open"
}

# Open the webcam (use 0 for the default webcam)
print(gstreamer_pipeline(flip_method=0))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

# Set a delay between frames (in milliseconds)
frame_delay = 30  # Adjust this value to control playback speed

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame from webcam.")
        break

    # Convert the OpenCV frame (BGR) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the RGB frame to a PIL image
    pil_image = Image.fromarray(rgb_frame)

    # Perform eye state detection
    results = eye_state_pipeline(pil_image)

    # Debug: Print raw model output
    print(results)
    
    # Use only the top prediction (highest confidence)
    if results:
        top_result = results[0]  # Get the first result (highest confidence)
        label = top_result["label"]
        confidence = top_result["score"]

        # Map the label to a human-readable description
        if label in label_mapping:
            eye_state = label_mapping[label]
        else:
            eye_state = label  # Fallback to the raw label if mapping is not found

        # Display the eye state and confidence score
        text = f"Eye State: {eye_state} ({confidence:.2f})"
        print(text)
        # cv2.putText(frame, text, (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the output frame
    #cv2.imshow("Eye State Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()