import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # Replace 'yolov8n.pt' with your model file for YOLOv11 if applicable

# Path to the input image
image_path = 'test_yolo_image2.jpg'  # Replace with your image path

# Load the image
image = cv2.imread(image_path)

# perform object detection
# the results object contains detailed information about the detections,
# such as bounding boxes, class labels, and confidence scores
results = model(image)

# Annotate the image with detection results
detections = results[0].plot()

cv2.imshow('YOLO Detection', detections)
cv2.waitKey(0)
cv2.destroyAllWindows()