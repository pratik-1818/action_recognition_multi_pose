import cv2
import numpy as np
import tensorflow as tf
import threading
from ultralytics import YOLO

# Load the YOLO model
model_yolo = YOLO('yolov8n-pose.pt')

# Load the pre-trained LSTM model
model_lstm = tf.keras.models.load_model("/home/vmukti/pose_multi/vmukti.h5")


cap = cv2.VideoCapture('/home/vmukti/pose_multi/faling_PCFWWblu.mp4')
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

if not cap.isOpened():
    print("Error: Could not open video stream. Please check the RTMP URL and network connection.")
    exit()

# Define the dimensions of the frames
desired_width = 640
desired_height = 480

# Initialize variables for keypoints and labels
keypoints_list = []
label = "neutral"
neutral_label = "neutral"

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0) if label == neutral_label else (0, 0, 255)
    thickness = 2
    lineType = 2
    cv2.putText(img, str(label), bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
    return img

def detect(model, keypoints_list):
    global label
    keypoints_array = np.expand_dims(np.array(keypoints_list), axis=0)
    result = model.predict(keypoints_array)
    if result[0][0] > 0.7:
        label = "neutral"
    elif result[0][1] > 0.7:
        label = "satnding"
    elif result[0][2] > 0.7:
        label = "walking"
    elif result[0][3] > 0.7:
        label = "fall_down"
   
    return str(label)

i = 0
warm_up_frames = 60
t1 = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab a frame. Stream may not be open.")
        break

    frame = cv2.resize(frame, (desired_width, desired_height))

    # Perform YOLO pose estimation
    results = model_yolo(frame)

    # Extract keypoints
    keypoints = results[0].keypoints

    if i > warm_up_frames:
        if keypoints.has_visible:
            keypoints_flat = keypoints.xy.flatten().tolist()
            keypoints_list.append(keypoints_flat)
            if len(keypoints_list) == 20:  # Assuming 20 timesteps
                if t1 is not None:
                    t1.join()
                t1 = threading.Thread(target=detect, args=(model_lstm, keypoints_list))
                t1.start()
                keypoints_list = []

        # Draw the YOLO pose estimation on the frame
        annotated_frame = results[0].plot()
        frame = annotated_frame

    # Display the label on the frame
    frame = draw_class_on_image(label, frame)
    cv2.imshow("YOLOv8 Pose Inference", frame)

    i += 1

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if t1 is not None:
    t1.join()
