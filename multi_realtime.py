import cv2
import numpy as np
import tensorflow as tf
import threading
from ultralytics import YOLO


model_yolo = YOLO('/home/vmukti/pose_multi/yolov8n-pose.pt')


model_lstm = tf.keras.models.load_model("/home/vmukti/pose_multi/sort/mukti.h5")


cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.86:554/ch0_0.264')
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

if not cap.isOpened():
    print("Error: Could not open video stream. Please check the video file path.")
    exit()


desired_width = 640
desired_height = 480


keypoints_dict = {}  
confidence_threshold = 0.8
person_confidence_threshold = 0.8

def draw_class_on_image(label, bbox, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (int(bbox[0]), int(bbox[1]) - 10)
    fontScale = 1
    fontColor = (0, 255, 0) if label == "StandWalk" else (0, 0, 255)
    thickness = 2
    lineType = 2
    cv2.putText(img, str(label), bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
    return img

def detect(person_id, keypoints_list):
    keypoints_array = np.expand_dims(np.array(keypoints_list), axis=0)
    result = model_lstm.predict(keypoints_array)
    
    label = "neutral"  
    if result[0][1] > confidence_threshold:
        label = "StandWalk"
    elif result[0][2] > 0.9:
        label = "fall_down"
    
    return label

def detect_and_draw(frame, person_id, results):
    keypoints_list = keypoints_dict[person_id]
    
    if len(keypoints_list) < 20:
        label = "neutral"  
    else:
        label = detect(person_id, keypoints_list)
    
    if len(results[0].boxes) > person_id:
        bbox = results[0].boxes[person_id].xyxy.cpu().numpy().astype(int) 
        if bbox.size > 0:
            bbox = bbox[0]  
            frame = draw_class_on_image(label, bbox, frame)
    
    return frame

i = 0
warm_up_frames = 60
threads = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab a frame. Stream may not be open.")
        break

    frame = cv2.resize(frame, (desired_width, desired_height))

    
    results = model_yolo(frame)

    if len(results[0].keypoints) == 0:
        print("No detections found.")
        continue

    
    for person_id, keypoints in enumerate(results[0].keypoints):
        if person_id >= len(results[0].boxes):
            continue  
        
        person_confidence = results[0].boxes[person_id].conf.cpu().numpy()[0]

        if person_confidence >= person_confidence_threshold and keypoints.has_visible:
            keypoints_flat = keypoints.xy.flatten().tolist()
            if person_id not in keypoints_dict:
                keypoints_dict[person_id] = []

            keypoints_dict[person_id].append(keypoints_flat)
            if len(keypoints_dict[person_id]) > 20:  
                keypoints_dict[person_id].pop(0)  

            
            if len(keypoints_dict[person_id]) == 20:
                if person_id in threads:
                    threads[person_id].join()
                
                t = threading.Thread(target=lambda pid=person_id: detect_and_draw(frame, pid, results))
                threads[person_id] = t
                t.start()

    
    for person_id in keypoints_dict:
        if len(keypoints_dict[person_id]) == 20:
            frame = detect_and_draw(frame, person_id, results)


    annotated_frame = results[0].plot()  
    frame = cv2.addWeighted(frame, 1.0, annotated_frame, 1.0, 0)

    
    cv2.imshow("YOLOv8 Pose Inference", frame)

    i += 1

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if threads:
    for t in threads.values():
        t.join()
