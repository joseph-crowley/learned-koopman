# Simple numerical integration technique to generate the time-series data.
# Could instead use measured data from a real-world dynamical system.

import numpy as np
import matplotlib.pyplot as plt
import torch


# Parameters
dt = 0.01  # Time step
T = 100  # Total time
n_steps = int(T / dt)  # Number of time steps

# Bounds for initial conditions
theta_init_bounds = (-np.pi, np.pi)
theta_dot_init_bounds = (-0.1, 0.1)
#theta_init_bounds = (-0.1, 0.1)
#theta_dot_init_bounds = (-0.01, 0.01)

# Number of different initial conditions
num_conditions = 500

# Initialize empty lists to store data
all_data_pairs = []
all_next_data_pairs = []

# Loop over different initial conditions
for _ in range(num_conditions):
    # Random initial conditions within the bounds
    theta_init = np.random.uniform(*theta_init_bounds)
    theta_dot_init = np.random.uniform(*theta_dot_init_bounds)
    
    # Initialize arrays to store theta and theta_dot values
    theta_values = np.zeros(n_steps)
    theta_dot_values = np.zeros(n_steps)
    theta_values[0] = theta_init
    theta_dot_values[0] = theta_dot_init

    # Time-stepping loop
    for t in range(1, n_steps):
        # Calculate acceleration (theta'') using the governing equation theta'' = -sin(theta)

        # exact 
        theta_double_dot = -np.sin(theta_values[t-1])

        ## linearized
        #theta_double_dot = -theta_values[t-1]
        
        # Update velocity and position using simple Euler integration
        theta_dot_values[t] = theta_dot_values[t-1] + dt * theta_double_dot
        theta_values[t] = theta_values[t-1] + dt * theta_dot_values[t]

    # Create pairs of (theta, theta_dot) for each time step
    data_pairs = np.stack([theta_values[:-1], theta_dot_values[:-1]], axis=1)
    next_data_pairs = np.stack([theta_values[1:], theta_dot_values[1:]], axis=1)

    # Append to the list
    all_data_pairs.append(data_pairs)
    all_next_data_pairs.append(next_data_pairs)

# Concatenate all the data
all_data_pairs = np.concatenate(all_data_pairs, axis=0)
all_next_data_pairs = np.concatenate(all_next_data_pairs, axis=0)

# Save numpy arrays to file
np.save('all_data_pairs.npy', all_data_pairs)
np.save('all_next_data_pairs.npy', all_next_data_pairs)

print("Shape of all_data_pairs:", all_data_pairs.shape)
print("Shape of all_next_data_pairs:", all_next_data_pairs.shape)

## Convert to PyTorch tensors
#all_data_tensor = torch.tensor(all_data_pairs, dtype=torch.float32)
#all_next_data_tensor = torch.tensor(all_next_data_pairs, dtype=torch.float32)
#
## Save PyTorch tensors to file
#torch.save(all_data_tensor, 'all_data_tensor.pt')
#torch.save(all_next_data_tensor, 'all_next_data_tensor.pt')


# Plot the generated data for a single initial condition
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Theta values over time")
plt.plot(np.arange(0, T, dt), theta_values)
plt.xlabel("Time")
plt.ylabel("Theta")

plt.subplot(2, 1, 2)
plt.title("Theta_dot values over time")
plt.plot(np.arange(0, T, dt), theta_dot_values)
plt.xlabel("Time")
plt.ylabel("Theta_dot")

plt.tight_layout()
plt.show()
