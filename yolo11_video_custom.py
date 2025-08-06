import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # 'n' denotes the nano version; choose according to your needs

model.train(data='D:\\lioness-detection.v1i.yolov11\\data.yaml', epochs=50, save=False)

# Open the input video
video_path = 'lion_video2.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# get the original video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
output_path = "output_lion1.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Display the frame (press 'q' to quit)
    cv2.imshow('YOLOv11 Detection', annotated_frame)
    # Write the annotated frame to the output video
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
