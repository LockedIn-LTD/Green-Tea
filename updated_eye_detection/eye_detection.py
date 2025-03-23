import cv2
import dlib
import numpy as np

# initialize the face/eye detector from dlib
eye_detector = dlib.get_frontal_face_detector()

# load pre-trained facial landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# method to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # calculate vertical distances between eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    
    # calculate horizontal distance between eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])
    
    # calculate EAR
    ear = (A + B) / (2.0 * C)
    return ear

# define the indices for only the left and right eye
left_eye_indices = list(range(36, 42))
right_eye_indices = list(range(42, 48))

# enable webcam for live capture
cap = cv2.VideoCapture(0)

# state variables for each eye
left_eye_state_buffer = "Open"
right_eye_state_buffer = "Open"

# loop to process frames
while True:
    # read each frame
    ret, frame = cap.read()
    
    # break loop if no frame is found
    if not ret:
        break

    # convert frame to grey scale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # find faces in frame
    faces = eye_detector(gray)

    # loop through each face
    for face in faces:
        # predict the landmarks on each face
        landmarks = predictor(gray, face)

        # get the landmarks for each eye
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_indices])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_indices])
        
        # call method to calculate EAR
        left_eye_ear = eye_aspect_ratio(left_eye)
        right_eye_ear = eye_aspect_ratio(right_eye)

        # boundry to determine when the eye is closed 
        boundry = 0.26 
        
        # determine the state of the left and right eyes
        if left_eye_ear < boundry:
            left_eye_state = "Closed"
        else:
            left_eye_state = "Open"
            
        if right_eye_ear < boundry:
            right_eye_state = "Closed"
        else:
            right_eye_state = "Open"   
            
        #left_eye_state = "Closed" if left_eye_ear < boundry else "Open"
        #right_eye_state = "Closed" if right_eye_ear < boundry else "Open"

        # update state variables
        if left_eye_state != left_eye_state_buffer:
            left_eye_state_buffer = left_eye_state
            
        if right_eye_state != right_eye_state_buffer:
            right_eye_state_buffer = right_eye_state

        # display the eye state and EAR value on live feed
        cv2.putText(frame, f"Left: {left_eye_state_buffer} ({left_eye_ear:.2f})", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right: {right_eye_state_buffer} ({right_eye_ear:.2f})", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # draw markers on the landmarks for visualization
        for point in left_eye:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

        for point in right_eye:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

    # display frame with live eye detection
    cv2.imshow("Eye Detection", frame)

    # detect when user wants to exit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # print EAR values
    print(f"Left EAR: {left_eye_ear:.3f}, Right EAR: {right_eye_ear:.3f}")

# disable webcam and close the window when done
cap.release()
cv2.destroyAllWindows()