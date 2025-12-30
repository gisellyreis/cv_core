import cv2
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n-seg.pt')

# Open video file
video_path = "road.mp4"
cap = cv2.VideoCapture(video_path)

# Video writing module
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
slow_fps = fps / 2

output_path = "YOLO-segmentation.avi"
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), slow_fps, (w, h))

# Loop through video frames
while cap.isOpened():
  success, frame = cap.read()

  if success:
    # Run inference on frame
    results = model(frame)
    # Visualize results on frame
    annotated_frame = results[0].plot()
    # Display annotated frame
    cv2.imshow("YOLOv8 Detection Inference", annotated_frame)
    # Write annotated frame on the video
    writer.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break
  else: 
    break

cap.release()
writer.release()
cv2.destroyAllWindows()
