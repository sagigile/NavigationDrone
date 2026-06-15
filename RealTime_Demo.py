import time
import numpy as np
from src.kalman import ConstantVelocityKalman3D


def run_realtime_demo():
    print("🚀 Loading Real-Time Navigation Engine (Kalman Filter)...")

    # 1. Load the Kalman engine with settings from the config.yaml
    kalman_filter = ConstantVelocityKalman3D(process_noise=2.0, measurement_noise=8.0)

    # 2. Simulation of noisy data arriving in real-time from computer vision
    # Sample data in [X, Y, Z] meters format
    noisy_camera_detections = [
        (0.0, np.array([0.0, 0.0, 50.0])),  # Smooth start
        (0.5, np.array([5.0, 0.2, 49.5])),  # Forward movement
        (1.0, np.array([10.0, -0.1, 50.1])),
        (1.5, np.array([15.5, 8.0, 50.0])),  # Violent noise jump from the camera (false detection)
        (2.0, np.array([20.0, 0.0, 50.0])),  # Back on track
        (2.5, np.array([25.0, 0.1, 49.8])),
    ]


    for timestamp, noisy_measurement in noisy_camera_detections:
        # Here happens the magic: updating the filter taking a fraction of a second
        start_time = time.perf_counter()
        clean_position = kalman_filter.update(t=timestamp, z=noisy_measurement)
        calc_time_ms = (time.perf_counter() - start_time) * 1000  # Calculate time in milliseconds

        print(f"⏱️ Time {timestamp}s | "
              f"Computer Vision (Noisy): {np.round(noisy_measurement, 1)} | "
              f"Kalman Filter (Clean): {np.round(clean_position, 1)}")
        print(f"   ⚡ Frame processing time: {calc_time_ms:.4f} ms (Ready for Real-Time!)\n")

        # Wait to simulate actual flight frame rate
        time.sleep(0.5)


if __name__ == "__main__":
    run_realtime_demo()