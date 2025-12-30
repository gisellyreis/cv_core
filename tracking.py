import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


class Identify:
  """
  YOLO Interactive Object Tracker with Live Crop Display. Click on any detected object to track
  it and view a real-time cropped view in the corner.
  """

  def __init__(self, model="yolov11s.pt", source="path/to/video.mp4", crop_size=(400, 400)):
    
    # Model initialization and classes names
    self.model = YOLO(model)
    self.names = self.model.names

    # Video capturing module
    self.cap = cv2.VideoCapture(source)
    assert self.cap.isOpened(), "Error reading video file"

    # Video writing module
    w, h, fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                                cv2.CAP_PROP_FRAME_HEIGHT,
                                                cv2.CAP_PROP_FPS))
    self.writer = cv2.VideoWriter("YOLO-tracking.avi",
                                  cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, (w, h)) 
    
    # Display settings
    self.crop_size = crop_size
    self.crop_margin = self.crop_pad = 5

    # Window setup
    self.window_name = "YOLO Tracking"
    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(self.window_name, self.mouse_callback)

    self.current_data = None # for mouse event
    self.selected_id = None  # tracking state
    self.ann = None  # annotation utils

  def mouse_callback(self, event, x, y, flags, param):
    """ Handle mouse clicks for object selection """
    if event == cv2.EVENT_LBUTTONDOWN and hasattr(self, "current_data"):
      boxes, ids = self.current_data
      self.selected_id = None

      if boxes is not None and ids is not None:
        for i, box in enumerate(boxes):
          x1, y1, x2, y2 = box
          if x1 <= x <= x2 and y1 <= y <= y2:
            self.selected_id = int(ids[i])
            break
  
  def crop_object_with_pad_and_resize(self, im0, box):
    """ Extract and resize object crop with padding """
    h, w = im0.shape[:2]
    x1, y1, x2, y2 = box.astype(int)

    # Crop with padding and bounds checking
    crop = im0[max(0, y1 - self.crop_pad):min(h, y2 + self.crop_pad),
               max(0, x1 - self.crop_pad):min(w, x2 + self.crop_pad)]
    
    # Return None if crop is invalid
    if crop.size == 0:
      return None
    
    # Resize maintaining aspect ratio
    ch, cw = crop.shape[:2]
    scale = min(self.crop_size[0] / cw, self.crop_size[1] / ch)
    return cv2.resize(crop, (int(cw * scale), int(ch * scale)))
  
  def add_crop_as_overlay(self, im0, crop):
    """ Display crop in top-right corner of frame """
    if crop is None:
      return im0
    
    h, w = im0.shape[:2]

    # Calculate position (top-right)
    y1, x1 = self.crop_margin, w - crop.shape[1] - self.crop_margin
    y2, x2 = y1 + crop.shape[0], x1 + crop.shape[1]

    # Bounds check and overlay
    if y2 <= h and x2 <= w and y1 >= 0 and x1 >= 0:
      cv2.rectangle(im0, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (68, 243, 0), 5)
      im0[y1:y2, x1:x2] = crop
    
    return im0
  
  def process_seleted_object(self, im0, boxes, ids):
    """ Handle cropping and display of selected object """
    if self.selected_id is None:
      return im0
    
    # Find selected object box
    for i, obj_id in enumerate(ids):
      if int(obj_id) == self.selected_id:
        selected_box = boxes[i].numpy()
        crop_resized = self.crop_object_with_pad_and_resize(im0.copy(), selected_box)
        return self.add_crop_as_overlay(im0, crop_resized)
      
    # Object lost from tracking
    print(f"Object ID {self.selected_id} lost from tracking")
    self.selected_id = None
    return im0
  
  def run(self):
    print("- Click on any detected object to select it")
    print("- Press 'q' to quit")
    print("- Press 'c' to clear selection")

    while self.cap.isOpened():
      success, im0 = self.cap.read()

      if not success:
        print("End of video or failed to read image.")
        break

      results = self.model.track(im0, persist=True) # Object tracking
      self.ann = Annotator(im0, line_width=4)

      if results and len(results) > 0:
        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
          boxes = result.boxes.xyxy.cpu()
          ids = result.boxes.id.cpu()
          clss = result.boxes.cls.tolist()

          self.current_data = (boxes, ids) # Store data for mouse callback

          im0 = self.process_seleted_object(im0, boxes, ids)

          if boxes is not None or ids is not None:
            for box, obj_id, cls in zip(boxes, ids.tolist(), clss):
              self.ann.box_label(box, label=self.names[cls],
                                 color=colors(6 if cls == 2 else cls, True))
      
      self.writer.write(im0)
      cv2.imshow(self.window_name, im0) # Display and handle input

      key = cv2.waitKey(50) & 0xFF
      if key == ord('q'):
        break
      elif key == ord('c'):
        self.selected_id = None
        print("Selection cleared")
    
    # Cleanup
    self.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  # Initialize and run tracker
  tracker = Identify(
    model="yolov10s.pt",
    source="road.mp4"
  )
  tracker.run()
