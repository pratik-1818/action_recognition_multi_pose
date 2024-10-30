import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO('yolov8n-pose.pt')

# Initialize video capture
cap = cv2.VideoCapture('/home/vmukti/pose_multi/faling_PCFWWblu (online-video-cutter.com)(1).mp4')


# Placeholder for storing keypoints data
keypoints_list = []

# Number of frames to collect
num_frames = 5000

frame_count = 0

while cap.isOpened() and frame_count < num_frames:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    # Perform pose estimation
    results = model(frame)

    
    keypoints = results[0].keypoints


    if keypoints.has_visible:
        
        keypoints_flat = keypoints.xy.flatten().tolist()
        keypoints_list.append(keypoints_flat)
    
    frame_count += 1

    # Display the annotated frame
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Pose Inference", annotated_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Convert the collected keypoints to a numpy array
keypoints_array = np.array(keypoints_list)

# Save the keypoints data to a file for later use
np.savetxt('fall_downs.txt', keypoints_array)
