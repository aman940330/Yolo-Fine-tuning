# YOLOv11 Object Detection (Image & Video Inference + Training)

This repository contains code to train a YOLOv11 model using the Ultralytics framework, and perform object detection on images and videos using OpenCV. It uses the `ultralytics` Python package, which simplifies the training and inference pipeline for the YOLO family of models.

---

## Requirements

Install the required libraries:

```bash
pip install ultralytics opencv-python
```

## Model

* `yolo11n.pt`: Pretrained YOLOv11 nano model. Replace this with your custom model if needed.
* You can download it or train your own using the training script provided below.

---

## Image Inference

### `image_inference.py`

This script:

* Loads a YOLOv11 model
* Detects objects in an input image
* Draws bounding boxes and class labels

```bash
python image_inference.py
```

Modify the image path in the script:

```python
image_path = 'test_yolo_image2.jpg'
```

### Output

* Opens a window displaying the image with bounding boxes and labels
* Confidence threshold is set at `0.5`

---

## Simple Image Inference

### `image_inference_simple.py`

A simpler version using the `.plot()` method from the Ultralytics results object for fast annotation.

```bash
python image_inference_simple.py
```

---

## Video Inference

### `video_inference.py`

* Loads a video file
* Performs object detection frame-by-frame
* Annotates and saves the result to `output_lion1.mp4`

```bash
python video_inference.py
```

Update the input video path in the script:

```python
video_path = 'lion_video2.mp4'
```

### `video_inference_simple.py`

A compact version of the video inference pipeline.

---

## Model Training

### `train_yolov11.py`

Trains the YOLOv11 model on a custom dataset using the Ultralytics format:

```bash
python train_yolov11.py
```

Update this line with the correct dataset YAML path:

```python
model.train(data='D:\\lioness-detection.v1i.yolov11\\data.yaml', epochs=50, save=False)
```

You can customize:

* `epochs`: Number of training iterations
* `save`: Set to `True` to save the model weights
* `data`: Path to your dataset YAML file

---

## Dataset Format

Ensure your custom dataset follows the YOLOv5/8/11 format, including:

* `images/`
* `labels/`
* `data.yaml` containing:

  ```yaml
  train: path/to/train/images
  val: path/to/val/images
  nc: <number of classes>
  names: [ 'class1', 'class2', ... ]
  ```

---

## Notes

* The `.plot()` method automatically applies bounding boxes and labels.
* You can switch from the nano version (`yolo11n.pt`) to other variants like `yolo11s.pt`, `yolo11m.pt`, etc., based on resource availability.

---

## Resources

* [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
* [YOLOv11 (if official release)](https://github.com/ultralytics/ultralytics)

---
