import cv2
from ultralytics import YOLO


MODEL = YOLO('yolov8n-pose.pt')


CAP = cv2.VideoCapture('rtmp://media5.ambicam.com:1938/live/viprepaidroming')


file_path = 'keypoints_data.txt'


with open(file_path, 'w') as file:
    frame_count = 0

    while CAP.isOpened() and frame_count < 5000:
        success, frame = CAP.read()
        if not success:
            print("Failed to grab frame")
            break
        
        
        results = MODEL(frame)
        
        
        keypoints = results[0].keypoints
        
        if keypoints.has_visible:
            for person_keypoints in keypoints.xy:
                
                keypoints_str = ' '.join(f"{x:.2f},{y:.2f}" for x, y in person_keypoints)
                
                file.write(f"{keypoints_str}\n")
                
                
                for keypoint in person_keypoints:
                    x, y = keypoint
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        
        cv2.imshow("YOLOv8 Pose Inference", frame)
        
        
        frame_count += 1
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


CAP.release()
cv2.destroyAllWindows()
