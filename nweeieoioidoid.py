import cv2
from ultralytics import YOLO
import numpy as np


model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture('')  


keypoints_list = []


num_frames = 5000

frame_count = 0

while cap.isOpened() and frame_count < num_frames:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    
    results = model(frame)
    
    keypoints = results[0].keypoints

    if keypoints.has_visible:
        keypoints_flat = keypoints.xy.flatten().tolist()
        
        
        if len(keypoints_flat) == 34:  
            keypoints_list.append(keypoints_flat)
        else:
            print(f"Skipping frame {frame_count} due to inconsistent keypoints length")

    frame_count += 1

    
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Pose Inference", annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

if keypoints_list:
    keypoints_array = np.array(keypoints_list)
    
    np.savetxt('neutral', keypoints_array)
else:
    print("No valid keypoints data collected.")
