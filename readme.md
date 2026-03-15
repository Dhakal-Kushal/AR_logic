# OpenAR: Feature-Based 3D Object Projection

A custom Augmented Reality (AR) engine built from scratch using OpenCV. This application detects a 2D reference image in a live video stream and overlays a 3D wireframe model onto it by calculating a real-time homography-to-projection matrix.

![AR Project Demo](./ardemo.gif)

## Overview
This project uses **natural feature tracking**. It extracts unique keypoints from a reference image and maps them to the 3D world space, allowing any high-contrast image to act as an AR trigger.

## Technical Stack
* **Language:** Python
* **Computer Vision:** OpenCV (SIFT/ORB)
* **Math & Geometry:** NumPy, Linear Algebra (Homography, SVD)
* **3D Modeling:** Custom wavefront `.obj` parser

## Technical Deep-Dive

### 1. Robust Feature Matching
The system implements a dual-mode feature detector:
* **SIFT (Scale-Invariant Feature Transform):** Used as the primary engine for its superior accuracy and rotation invariance.
* **ORB (Oriented FAST and Rotated BRIEF):** Implemented as a fallback for higher-performance real-time tracking on lower-end hardware.
* **Flann-Based Matcher:** Utilizes K-D Trees for high-speed nearest-neighbor matching between the model and the live scene.

### 2. 3D Geometry & Homography
The core of this project is the **Projection Matrix Calculation**. Instead of using pre-built AR libraries, the transformation is handled manually:
* **Homography Estimation:** Uses `RANSAC` to find a robust mapping between the reference image and the camera frame, effectively filtering out outlier matches.
* **Matrix Decomposition:** Decomposes the homography matrix into rotation and translation vectors using the camera’s intrinsic parameters.
* **Normalizing Orthogonality:** Implements cross-product logic to ensure the rotation matrix columns remain orthonormal, preventing 3D model warping.

### 3. Stability & Optimization
* **Homography Smoothing:** Implements a `deque` buffer to average the homography matrix across multiple frames, significantly reducing "jitter" in the 3D overlay.
* **Custom OBJ Parser:** A lightweight implementation to read vertex and face data from standard 3D files without the overhead of heavy 3D engines.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Dhakal-Kushal/AR_logic.git
    cd AR_logic
    ```

2.  **Dependencies:**
    ```bash
    pip install opencv-contrib-python numpy
    ```
    *Note: `opencv-contrib` is required for SIFT.*

3.  **Setup:**
    * Place your reference image in `models/ref.jpg`.
    * Place your 3D model in `3dObject.obj`.

4.  **Run:**
    ```bash
    python main.py
    ```

## Challenges Overcome
* **The "Perspective Distortion" Problem:** Solved the issue of model warping when viewing the reference image at steep angles by refining the camera calibration matrix ($K$) and implementing specific projection constraints.
* **Tracking Jitter:** Introduced a rolling average smoothing algorithm to maintain a stable 3D render even when feature matches fluctuate frame-to-frame.