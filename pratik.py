from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load the YOLOv8 model with custom weights
model = YOLO("/home/vmukti/pose_multi/best(1).pt")

# Parameters
loitering_time_seconds = 30   # 30 seconds for loitering
next_id = 0  # To assign unique IDs to each person

# Data structure to store object positions and track counts
trackers = {}

# Define the desired width and height for the resized frame
resize_width = 640  # Set your desired width
resize_height = 480  # Set your desired height
tracker_config = "/home/vmukti/pose_multi/bytetrack.yaml"

# Open the video file or capture from webcam
cap = cv2.VideoCapture("rtsp://admin:admin@192.168.1.41:554/ch0_0.264")  # Replace with your video source

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture frame")
        break

    # Resize the frame
    frame = cv2.resize(frame, (resize_width, resize_height))

    # Run the model on the current frame with tracking
    results = model.track(source=frame, show=False, tracker=tracker_config, imgsz=(resize_width, resize_height))

    # Get the current timestamp
    current_time = time.time()

    # Temporary dictionary to hold current frame's detections
    new_trackers = {}

    for result in results:
        for box in result.boxes:
            # Extract information from each detected box
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            # Check if the ID is not None
            if box.id is not None:
                obj_id = int(box.id)  # Tracker ID assigned by YOLO
            else:
                continue  # Skip this detection if ID is None

            if obj_id not in trackers:
                trackers[obj_id] = {'start_time': current_time}

            # Update tracking information
            is_loitering = (current_time - trackers[obj_id]['start_time']) >= loitering_time_seconds
            new_trackers[obj_id] = {
                'start_time': trackers[obj_id]['start_time'],
                'is_loitering': is_loitering
            }

            # Display the bounding box and label
            if is_loitering:
                label = f"Loitering: ID {obj_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Print alert message
                print(f"Alert! Person with ID {obj_id} is loitering for more than {loitering_time_seconds} seconds!")
            else:
                # Draw the bounding box on the frame (normal detection)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID: {obj_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update trackers with new positions and times
    trackers = new_trackers

    # Display the raw frame for debugging
    cv2.imshow("Raw Frame", frame)

    # Display the frame with detections
    cv2.imshow("YOLOv8 Loitering Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
