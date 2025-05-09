from filterpy.kalman import KalmanFilter
import numpy as np


def init_KF(kf):
    # Initialize Kalman Filter for 2D tracking
    # kf = KalmanFilter(dim_x=4, dim_z=2)

    # Initial state: [x, y, vx, vy]
    # kf.x = np.array([0., 0., 0., 0.])

    # State transition matrix (assuming constant velocity model)
    dt = 1.0  # time step
    kf.F = np.array([[1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Measurement function: we observe only position
    kf.H = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])

    # Measurement noise covariance
    kf.R = np.array([[1, 0],
                    [0, 1]])

    # Initial estimate error covariance
    kf.P *= 20.

    # Process noise covariance (tune this for smoother or faster response)
    kf.Q = np.eye(4) * 0.1

    # Sample measurements: [x, y] positions from sensor (e.g., noisy detection)
    measurements = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]

def predict_KF(kf, x, measurements):
    kf.x = x
    for z in measurements:
            kf.predict()
            kf.update(z)
            print(f"Estimated Position: x={kf.x[0]:.2f}, y={kf.x[1]:.2f}, "
                f"vx={kf.x[2]:.2f}, vy={kf.x[3]:.2f}")
