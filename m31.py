import numpy as np

def kalman_filter_constant_velocity(x, P, F, Q, H, R, z):
    # Prediction
    x_pred = np.dot(F, x)
    P_pred = np.dot(F, np.dot(P, F.T)) + Q

    # Kalman gain
    S = np.dot(H, np.dot(P_pred, H.T)) + R
    K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(S)))

    # Update
    y = z - np.dot(H, x_pred)
    x_updated = x_pred + np.dot(K, y)
    P_updated = P_pred - np.dot(K, np.dot(H, P_pred))

    return x_updated, P_updated

# Measurement data
MR = 94779.54
MA = 217.0574
ME = 2.7189
MT = 21486.916

# Initial state
x = np.array([[MR], [MA], [ME], [MT]])  # State vector [range, azimuth, elevation, time]

# Initial covariance matrix
P = np.diag([1, 1, 1, 1])  # Covariance matrix

# Constant velocity motion model
dt = 1.0  # Time step
F = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])  # State transition matrix

# Process noise covariance
Q = np.diag([0.01, 0.01, 0.01, 0.01])  # Process noise covariance matrix

# Measurement matrix
H = np.eye(4)  # Identity matrix

# Measurement noise covariance
R = np.diag([0.01, 0.01, 0.01, 0.01])  # Measurement noise covariance matrix

# Measurement
z = np.array([[94805.44], [217.89], [2.0831], [21486.916]])  # Measurement vector

# Kalman filter prediction
x_pred, _ = kalman_filter_constant_velocity(x, P, F, Q, H, R, z)

# Extracting predicted values
predicted_range = x_pred[0, 0]
predicted_azimuth = x_pred[1, 0]
predicted_elevation = x_pred[2, 0]
predicted_time = x_pred[3, 0]

print("Predicted Range:", predicted_range)
print("Predicted Azimuth:", predicted_azimuth)
print("Predicted Elevation:", predicted_elevation)
print("Predicted Time:", predicted_time)
