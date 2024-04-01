import numpy as np
import matplotlib.pyplot as plt

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

# Measurement data without the first column
data = [
    [94805.44, 217.89, 2.0831, 21486.916],
    [27177.54, 153.5201, 2.086, 21487.166],
    [85834.72, 226.6823, 4.7109, 21487.189],
    [26591.4, 120.7162, 1.3585, 21487.24],
    [67521.98, 295.1252, 2.8341, 21487.256],
    [64726.5, 341.2639, 4.6564, 21487.332],
    [24220.79, 89.6023, 3.1265, 21487.369],
    [3768.37, 129.798, 12.6818, 21487.381],
    [20474.44, 27.3968, 0.6826, 21487.557],
    [94854.33, 217.6161, 2.4473, 21487.693],
    [27184.91, 153.4323, 1.7675, 21487.863],
    [64715.11, 341.1737, 4.6514, 21487.971],
    [70434.91, 325.4155, 3.0297, 21488.012],
    [26844.95, 301.2844, 4.9459, 21488.039],
    [80301.8, 352.2547, 4.7756, 21488.08],
    [87872.73, 46.1141, 6.5272, 21488.119],
    [66776.26, 104.3781, 3.9765, 21488.057]
]

# Initial state adjusted to match the first measurement
x = np.array([[data[0][0]], [data[0][1]], [data[0][2]], [data[0][3]]])  # State vector [range, azimuth, elevation, time]

# Initial covariance matrix adjusted to smaller values
P = np.diag([0.1, 0.1, 0.1, 0.1])  # Covariance matrix

# Constant velocity motion model
dt = 1.0  # Time step
F = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])  # State transition matrix

# Process noise covariance adjusted to smaller values
Q = np.diag([0.001, 0.001, 0.001, 0.001])  # Process noise covariance matrix

# Measurement matrix
H = np.eye(4)  # Identity matrix

# Measurement noise covariance adjusted to smaller values
R = np.diag([0.001, 0.001, 0.001, 0.001])  # Measurement noise covariance matrix

# Lists to store predicted values
predicted_ranges = []
predicted_azimuths = []
predicted_elevations = []
predicted_times = []

# Perform Kalman filter prediction for each data point
for measurement in data:
    z = np.array([[measurement[0]], [measurement[1]], [measurement[2]], [measurement[3]]])  # Measurement vector
    
    # Kalman filter prediction
    x_pred, _ = kalman_filter_constant_velocity(x, P, F, Q, H, R, z)

    # Extracting predicted values
    predicted_range = x_pred[0, 0]
    predicted_azimuth = x_pred[1, 0]
    predicted_elevation = x_pred[2, 0]
    predicted_time = x_pred[3, 0]
    
    # Append predicted values to lists
    predicted_ranges.append(predicted_range)
    predicted_azimuths.append(predicted_azimuth)
    predicted_elevations.append(predicted_elevation)
    predicted_times.append(predicted_time)

    # Update state for the next iteration
    x = x_pred
    
    # Print predicted values for each measurement
    print("Predicted values for measurement at time", measurement[3])
    print("Predicted range:", predicted_range)
    print("Predicted azimuth:", predicted_azimuth)
    print("Predicted elevation:", predicted_elevation)
    print("Predicted time:", predicted_time)
    print()

# Plotting
plt.figure(figsize=(12, 8))

# Plot measured and predicted range
plt.subplot(2, 2, 1)
plt.plot([measurement[3] for measurement in data], [measurement[0] for measurement in data], label='Measured', marker='o')
plt.plot([measurement[3] for measurement in data], predicted_ranges, label='Predicted', marker='o')
plt.xlabel('Time')
plt.ylabel('Range')
plt.title('Range')
plt.legend()

# Plot measured and predicted azimuth
plt.subplot(2, 2, 2)
plt.plot([measurement[3] for measurement in data], [measurement[1] for measurement in data], label='Measured', marker='o')
plt.plot([measurement[3] for measurement in data], predicted_azimuths, label='Predicted', marker='o')
plt.xlabel('Time')
plt.ylabel('Azimuth')
plt.title('Azimuth')
plt.legend()

# Plot measured and predicted elevation
plt.subplot(2, 2, 3)
plt.plot([measurement[3] for measurement in data], [measurement[2] for measurement in data], label='Measured', marker='o')
plt.plot([measurement[3] for measurement in data], predicted_elevations, label='Predicted', marker='o')
plt.xlabel('Time')
plt.ylabel('Elevation')
plt.title('Elevation')
plt.legend()

# Plot measured and predicted time
plt.subplot(2, 2, 4)
plt.plot([measurement[3] for measurement in data], [measurement[3] for measurement in data], label='Measured', marker='o')
plt.plot([measurement[3] for measurement in data], predicted_times, label='Predicted', marker='o')
plt.xlabel('Time')
plt.ylabel('Time')
plt.title('Time')
plt.legend()

plt.tight_layout()
plt.show()
