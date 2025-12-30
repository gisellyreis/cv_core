# cv_core

**cv_core** is a modular computer vision sandbox exploring real-time Object Detection, Instance Segmentation, and Multi-Object Tracking (MOT) using state-of-the-art YOLO models.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)

---

## Demo

The core pipeline processes raw video feeds to extract semantic information. Below are the results of applying different computer vision tasks to the same input.

### Input Source
> **Context:** Traffic aerial footage used for testing inference speed and accuracy.

| Original Feed |
| :---: |
| ![Original Feed](assets/road.gif) |

### Model Outputs

| 1. Object Detection | 2. Instance Segmentation | 3. Object Tracking |
| :---: | :---: | :---: |
| *Identifies objects with bounding boxes.* | *Pixel-level classification for shape analysis.* | *Assigns unique IDs to maintain identity over time.* |
| ![Detection](assets/YOLO-detection.gif) | ![Segmentation](assets/YOLO-segmentation.gif) | ![Tracking](assets/YOLO-tracking.gif) |

---

## Installation

1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/cv_core.git](https://github.com/yourusername/cv_core.git)
   cd cv_core

2. **Install dependencies** \
  *Ensure you have Python installed, then run:*
   ```bash
    pip install -r requirements.txt

--- 

## Usage
The project is split into modular scripts for each task. You can run them individually:

1. **Detection** \
  Runs standard bounding box detection (YOLOv8n).
   ```bash
    python detection.py

2. **Segmentation** \
   Runs instance segmentation to separate objects from the background (YOLOv8n-seg).
    ```bash
    python segmentation.py

3. **Tracking** \
   Implementation of BoT-SORT/ByteTrack for persistent object identification.
   ```bash
   python tracking.py

---