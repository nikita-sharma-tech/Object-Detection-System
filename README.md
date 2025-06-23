#  Real-Time Object Detection using SSD MobileNet & OpenCV

Welcome to this exciting computer vision project!  
Here, we use **SSD MobileNet**, a highly efficient deep learning model, to detect multiple real-world objects like people, cars, bottles, and more — in real-time using OpenCV.

##  Project Highlights

-  Real-time object detection on images, video, or webcam
-  Uses **SSD MobileNet v3**, a fast and lightweight model ideal for edge devices
-  Detects objects with confidence scores and draws bounding boxes
-  Built with **OpenCV DNN module** (no need for TensorFlow/PyTorch setup)

  ##  Tech Stack

| Tool         | Purpose                        |
|--------------|--------------------------------|
| Python       | Core programming language      |
| OpenCV       | Image & video processing       |
| SSD MobileNet| Pre-trained object detection model |
| NumPy        | Numerical operations           |

###  Clone the Repository


git clone https://github.com/yourusername/ssd-object-detection.git
--
cd ssd-object-detection
---

### How It Works
1. Loads SSD MobileNet model and label file.

2. Captures frames from webcam or video.

3. Preprocesses each frame (resizes, normalizes, etc.).

4. Feeds frame into model using OpenCV's DNN module.

5. Extracts detections and draws labeled boxes.

### Future Enhancements
- Add object tracking (using SORT or Deep SORT)

- Deploy on mobile or embedded devices

- Add GUI using Tkinter or Streamlit

Author
--
Nikita Sharma
 --
📧 nikitasharmaearthling@gmail.com
--
🚀 AI & Computer Vision Enthusiast


