# Green Tea: Embedded Vision for Drowsiness Detection 🍵

**Green Tea** is the computer vision subsystem of the **LockedIn** driver safety platform. Optimized specifically for the **NVIDIA Jetson Orin Nano**, this project leverages a custom-compiled software stack to perform high-speed, low-latency fatigue detection on the edge.

This module acts as the "Eye" of the system, processing video streams and transmitting inference data via ZeroMQ to the [Hardware Controller (HW-Main)](https://github.com/LockedIn-LTD/HW-main), which manages physical alerts and sensors (IMU, Pulse Oximeter, Pressure sensor, Speakers, LCD)

## ⚡ System Architecture

The system is split into two asynchronous nodes communicating over TCP (ZMQ):

1. **Vision Node (Green Tea):** Handles heavy ML inference (YOLOv8 + OpenCV DNN).
2. **Controller Node (HW-Main):** Handles GPIO, sensors, watchdog timers, and physical alerts.

## 🚀 Key Performance Features

* **Jetson Orin Nano Optimized:** Built to run on ARM64 architecture with NVIDIA JetPack.
* **Custom Software Stack:** Runs on a self-compiled version of **OpenCV** linked against **CUDA 12.6** and cuDNN, enabling `cv2.cuda.GpuMat` hardware acceleration for image preprocessing.
* **Hybrid Inference Engine:**
* **Face Detection:** GPU-accelerated OpenCV DNN (ResNet10 SSD).
* **State Classification:** Ultralytics YOLOv8 optimized for drowsiness classification.

* **Smart Decimation:** Implements logic to throttle expensive inference calls (running every Nth frame) while maintaining high-frequency tracking, ensuring the Jetson does not overheat or lag.

## 🛠️ Hardware & Tech Stack

* **Platform:** NVIDIA Jetson Orin Nano (Developer Kit)
* **Language:** Python 3.x
* **Vision:** OpenCV (Custom Build with CUDA 12.6), PyTorch (GPU), Ultralytics YOLO
* **IPC:** ZeroMQ (Pub/Sub)

## 📦 Installation & Setup

### Prerequisites

* **NVIDIA JetPack 6.x** (or compatible version supporting CUDA 12.6).
* **OpenCV with CUDA:** This project requires OpenCV to be built from source with `OPENCV_DNN_CUDA=ON` and `WITH_CUDA=ON`. Standard `pip install opencv-python` **will not work** as it lacks GPU support.

### Clone the Repository

```bash
git clone https://github.com/LockedIn-LTD/Green-Tea.git
cd Green-Tea

```

*Note: The required caffe models (`res10_300x300_ssd_iter_140000.caffemodel`) and helper classes (`zmq_video_client.py`) are included in this repository.*

## 💻 Usage

This module is designed to be launched by the main hardware controller, but can be run independently for debugging.

1. **Ensure ZMQ Streams are Active:**
The model expects a raw video frame stream on `tcp://127.0.0.1:5555`.
2. **Run the Vision Node:**
```bash
python mainOpenCV.py

```

3. **Output:**
The model publishes JSON status payloads to `tcp://127.0.0.1:5557`:
```json
{
  "perclos_time_s": 2.5,
  "status": "FATIGUE_DROWSY"
}

```

## 🔗 Integration

This repository works in tandem with **[HW-Main](https://github.com/LockedIn-LTD/HW-main)**, which consumes the output from this model to trigger:

* **Auditory Alarms:** Active sound warnings.
* **Visual Status:** LCD updates live.
* **Cloud connection:** Data and events are stored in cloud for review afterward.

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
