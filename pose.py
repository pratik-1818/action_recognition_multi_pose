from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-pose.yaml")  
model = YOLO("yolov8n-pose.pt")  
model = YOLO("yolov8n-pose.yaml").load("yolov8n-pose.pt") 


results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)