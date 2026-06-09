from __future__ import annotations

import numpy as np


class ConstantVelocityKalman3D:
    """Simple 3D constant-velocity Kalman filter over ENU coordinates."""

    def __init__(self, process_noise: float = 2.0, measurement_noise: float = 8.0):
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.x: np.ndarray | None = None
        self.p: np.ndarray | None = None
        self.last_t: float | None = None

    def update(self, t: float, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(3)

        if self.x is None:
            self.x = np.array([z[0], z[1], z[2], 0.0, 0.0, 0.0], dtype=float)
            self.p = np.eye(6) * 10.0
            self.last_t = float(t)
            return z

        dt = max(1e-3, float(t) - float(self.last_t))
        self.last_t = float(t)

        f = np.eye(6)
        f[0, 3] = dt
        f[1, 4] = dt
        f[2, 5] = dt

        q = np.eye(6) * self.q
        q[0, 0] *= dt * dt
        q[1, 1] *= dt * dt
        q[2, 2] *= dt * dt

        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q

        h = np.zeros((3, 6))
        h[0, 0] = h[1, 1] = h[2, 2] = 1.0
        r = np.eye(3) * self.r

        y = z - h @ self.x
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.inv(s)

        self.x = self.x + k @ y
        self.p = (np.eye(6) - k @ h) @ self.p
        return self.x[:3].copy()
