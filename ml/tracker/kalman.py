"""2D constant-velocity Kalman filter for golf ball tracking.

State x = [a, b, u, v]: position (a, b) and velocity (u, v) in pixels/frame.
Measurement z = [a, b]: the detected ball position.

Matches the formulation in the paper (Section III-B): time update (a priori
estimate) predicts the next position before detection runs on the cropped
ROI; measurement update (a posteriori estimate) corrects that prediction
once the detector returns a result.
"""

import numpy as np


class KalmanFilter2D:
    # Q/R matches the reference implementation (golf_ball_reference/test_net.py)
    # rather than a generic default: Q/R ~= 1000 means the filter trusts a
    # fresh detection almost completely instead of smoothing it against the
    # prediction. That matters specifically at impact, where true velocity
    # jumps from ~0 to 100+mph in a single frame — a filter that smooths
    # heavily (low Q/R) structurally cannot react to that in time and lags
    # behind the real ball, which is what we saw in testing.
    def __init__(self, initial_x, initial_y, process_noise=0.1, measurement_noise=0.0001):
        self.x = np.array([initial_x, initial_y, 0.0, 0.0], dtype=np.float64)

        self.A = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        # Zero initial uncertainty: like the reference implementation, frame 0
        # comes from a known-good point (their hardcoded ground truth, our
        # user tap) rather than a noisy estimate.
        self.P = np.zeros((4, 4))

    def predict(self):
        """Time update — a priori estimate. Returns predicted (x, y)."""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[0], self.x[1]

    def update(self, measured_x, measured_y):
        """Measurement update — a posteriori estimate given a detection."""
        z = np.array([measured_x, measured_y], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[0], self.x[1]

    @property
    def position(self):
        return self.x[0], self.x[1]
