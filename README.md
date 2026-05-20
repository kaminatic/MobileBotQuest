# MobileBotQuest: Vision-Based Autonomous Navigation

An autonomous, purely vision-based navigation system designed for a mobile robot chassis. This system uses a top-mounted RGB camera to detect unknown obstacle configurations, map their positions relative to the robot using homography transformations, and safely navigate through them without relying on LiDAR or odometry.

Developed as an academic challenge for the **Robotics 2025** course at LIACS, Universiteit Leiden.

## 👥 Collaborators & Team Credits
This project was developed through a collaborative team effort by **Team Spigot**:
* **Nataliia Kaminskaia**
* **Michael Olthof**
* **Robert C. Weber**
* **Amber van der Tuin**
* **Abdolrahim Tooranian**

Course taught by: *Erwin Bakker*

---


## 🛠️ Repository Structure

```text
├── n8_capture.py               # Background capturing and color tracking pipeline
├── n8_ch2.2.py                 # Structural Similarity (SSIM) detection and route planning
├── .gitignore                  # Git tracking rules
├── LICENSE                     # MIT open-source license
└── requirements.txt            # Python dependencies

```

*Note: Calibrated homography matrices (`.npy`) and temporary map/image outputs have been omitted to adhere to university assignment distribution guidelines.*

---
## 🚀 Pipeline & Technical Methods

### 1. Foreground Detection via SSIM
[cite_start]Rather than using naive frame differencing, the pipeline takes a baseline frame of an empty arena and compares it to the active environment using **Structural Similarity Index Measure (SSIM)** tracking[cite: 9, 73, 74]. [cite_start]This isolates new objects robustly despite minor shifts in environment lighting[cite: 74, 88]:

**Difference = (1 - SSIM(Scene, Background)) · 255**

[cite_start]The difference map undergoes a Gaussian filter, binary thresholding, and morphological opening/closing operations to eliminate shadow artifacts and specular floor reflections[cite: 70, 75].

### 2. Homography & Coordinate Mapping
[cite_start]Pixel positions corresponding to the baseline contact point of the target objects are extracted using contour area filters[cite: 77, 78]. [cite_start]These are mapped into real-world workspace coordinates via a calibrated inverse homography matrix (H_inv)[cite: 81]:

**P_world = H_inv · P_pixel**

### 3. Triangle Route Navigation Strategy
[cite_start]Path planning calculates a target trajectory utilizing an equilateral triangle geometry over the object positions[cite: 84]:
1. [cite_start]Locate the centroids of the two obstacles (A and B)[cite: 78].
2. [cite_start]Compute their geographic midpoint and baseline distance[cite: 78].
3. [cite_start]Establish a safe steering target (the **Triangle Apex**) placed perpendicular to the midpoint[cite: 84].
4. [cite_start]Drive smoothly to the apex using proportional steering, align the chassis angle toward the midpoint vector, and execute a timed straight-line thrust to navigate safely through the clearing[cite: 84, 85].
---

## ⚙️ Getting Started

### Prerequisites

* Python 3.8+
* Target robot system equipped with a compatible RGB camera framework (`picamera2` support or tailored OpenCV video stream)

### Installation

1. Clone the repository:
```bash
git clone [https://github.com/kaminatic/MobileBotQuest.git](https://github.com/kaminatic/MobileBotQuest.git)
cd MobileBotQuest

```


2. Install python package requirements:
```bash
pip install -r requirements.txt

```

### Execution Protocol

1. **Scene Profiling:** Execute `n8_capture.py` to capture a baseline snapshot of your empty workspace arena. Follow terminal prompts to confirm background registration and configure localized HSV bounds for object tracking targets.
2. **Autonomous Trajectory:** Ensure your tracking camera is running and your target matrix file (`homography_og.npy`) is accessible in the working folder. Place your target objects into the arena and run the primary pilot module:
```bash
python n8_ch2.2.py

```