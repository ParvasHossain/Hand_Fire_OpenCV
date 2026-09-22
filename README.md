# 💥 Hand Gesture Fire Controller

An interactive Python application that uses **MediaPipe Hand Landmarking** and **OpenCV** to detect open palm gestures via webcam and overlay a dynamic, rotating flame GIF directly onto the user's hand in real time.

---

## ✨ Features

- **Real-Time Hand Tracking:** Powered by Google MediaPipe Tasks API for precise multi-hand detection.
- **Gesture Activation:** Fire effect triggers automatically when an **Open Palm** gesture is detected.
- **Dynamic Rotation & Scaling:** Flame rotates based on hand orientation and scales dynamically according to hand distance from the camera.
- **Transparent GIF Overlay:** Smooth alpha blending for RGBA GIF animations.
- **Resizable & Fullscreen Mode:** Toggle borderless fullscreen output using keyboard shortcuts.

---

## 📁 Project Structure

```text
 hand-fire-controller/
├── main.py                  # Primary application script
├── hand_landmarker.task     # MediaPipe Hand Landmarker model file
├── fire.gif                 # Fire animation GIF with transparent background
├── requirements.txt         # Dependency declarations
└── README.md                # Project documentation
