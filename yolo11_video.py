import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # 'n' denotes the nano version; choose according to your needs

# Open the input video
video_path = 'yolo_test.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
output_path = 'output_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the frame (press 'q' to quit)
    cv2.imshow('YOLOv11 Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()