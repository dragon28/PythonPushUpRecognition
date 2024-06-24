import cv2
import mediapipe as mp
import numpy as np
import imutils
import sys


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def main(file_path):
    
    # Push up video file location
    pushup_video_file = file_path
        
    # Mediapipe pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Open video file
    cap = cv2.VideoCapture(pushup_video_file)

    
    # Initiate the pose detector
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        pushup_counter = 0
        pushup_stage = None

        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame if needed for faster processing
            frame = imutils.resize(frame, width=700)
            
            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Revert image to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates of the relevant landmarks
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Push-up detection logic
                if angle > 160:
                    pushup_stage = 'up'
                if angle < 90 and pushup_stage == 'up':
                    pushup_stage = 'down'
                    pushup_counter += 1

                # Render push-up counter
                cv2.putText(image, 'Push-ups: {}'.format(pushup_counter), 
                            (10,40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA
                        )
                
            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the resulting frame
            cv2.imshow('Push-Up Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    
    if len(sys.argv) > 1 and sys.argv[1] != '':
        main(sys.argv[1])
    
    else:
        
        print('Please provide the path to the video file')
        sys.exit(1)