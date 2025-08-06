import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # Replace 'yolov8n.pt' with your model file for YOLOv11 if applicable
class_names = model.names

# Path to the input image
image_path = 'test_yolo_image2.jpg'  # Replace with your image path

# Load the image
image = cv2.imread(image_path)

# perform object detection
# the results object contains detailed information about the detections,
# such as bounding boxes, class labels, and confidence scores
results = model(image)

# Annotate the image with detection results
detections = results[0]

for result in results:
    boxes = result.boxes  # This contains all detections
    for box in boxes:
        # Coordinates of the bounding box
        # x1, y1, x2, y2 = box.xyxy[0]  # Top-left (x1, y1), bottom-right (x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Confidence level
        confidence = box.conf[0]
        # Class label
        class_label = int(box.cls[0])
        # Get the class name from the model's class names mapping
        class_name = class_names[class_label]

        # Draw bounding box if confidence > 0.5
        if confidence > 0.5:
            # Draw the rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (252, 3, 127), 2)
            # Add label and confidence score
            label = f"{class_name}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (252, 3, 127), 2)

# Display the image with annotations
cv2.imshow('YOLO Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()