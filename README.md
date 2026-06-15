# Visual Navigation in GPS-Denied Environments (UAVs)

This repository contains the final project for autonomous UAV visual navigation. The system is designed to estimate the real-time location and flight trajectory of a drone in a GPS-denied environment by relying solely on visual inputs (camera feed), reference mapping, and advanced filtering algorithms.

**Developed at:** Ariel University, Department of Computer Science.

## 🌍 Trajectory Results

Here is the final offline-fusion "View-Center" trajectory running accurately on the 2D map:

<img width="934" height="567" alt="trajectory_result png" src="https://github.com/user-attachments/assets/4ff0d052-bd6f-4f85-aa8e-8b289a3c0577" />

## 🎯 Project Overview

The core objective is to calculate the precise "View-Center" trajectory...

The core objective is to calculate the precise "View-Center" trajectory (the exact ground location the drone's camera is pointing at) using a hybrid architecture. The system compares a raw, telemetry-less query video against previously recorded reference flights and a 2D satellite map.

The architecture is divided into two main execution pipelines:
1. **Offline Fusion Optimizer (High Precision):** A batch-processing pipeline that extracts features, reranks matches, and solves a robust factor graph to generate a perfectly smoothed 3D/2D trajectory.
2. **Online Real-Time Engine (High Speed):** A lightweight `ConstantVelocityKalman3D` filter designed for edge computing, demonstrating sub-millisecond noise reduction for live camera feeds.

## ✨ Key Features
* **Multi-Source Localization:** Integrates Visual Odometry (VO), Satellite Map Priors, and Reference Flight Anchors.
* **Robust Factor Graph Optimization:** Eliminates trajectory drift and physical anomalies (e.g., teleportation jumps) using adaptive robust loss functions.
* **Real-Time Kalman Filtering:** Smooths violent visual tracking errors (false positives) in real-time.
* **KML Export:** Automatically generates Google Earth compatible `.kml` files for immediate 3D visual validation.

## 📁 Repository Structure
* `Main.py` - The primary entry point for the offline batch-processing pipeline.
* `RealTime_Demo.py` - Proof of Concept (PoC) demonstrating the real-time Kalman filter engine.
* `config.yaml` - The master configuration file containing weights, thresholds, and operational parameters.
* `src/` - Core source code (Kalman filter, Geo-alignment, KML export, etc.).
* `pipeline/` - Scripts for sequential data processing (Anchor matching, VO integration, Fusion).
* `requirements.txt` - Python dependencies.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sagigile/NavigationDrone.git
   cd NavigationDrone

## 🚀 Usage

### 1. Offline Trajectory Generation (Max Precision)
To generate the highly accurate, smoothed flight path using the pre-computed feature matching data:
```bash
python Main.py --config config.yaml --only-fusion
python RealTime_Demo.py

