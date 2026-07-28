import torch
import numpy as np
import matplotlib.pyplot as plt

from cVAE import cVAE

from constants import HIDDEN_DIM, LATENT_DIM, INPUT_DIM, N_STEPS_TEST, DT_TEST

if __name__ == "__main__":
    cVAE_model = cVAE(HIDDEN_DIM, LATENT_DIM, INPUT_DIM)
    cVAE_model.load_state_dict(torch.load('cVAE_model.pth'))

    # Step 1: Generate Test Data with different initial conditions
    theta_init_test = 0.05  # New initial position (angle)
    theta_dot_init_test = 0.0  # New initial velocity (angular velocity)

    # Initialize arrays to store theta and theta_dot values
    theta_values_test = np.zeros(N_STEPS_TEST)
    theta_dot_values_test = np.zeros(N_STEPS_TEST)
    theta_values_test[0] = theta_init_test
    theta_dot_values_test[0] = theta_dot_init_test

    # Time-stepping loop to generate test data
    for t in range(1, N_STEPS_TEST):
        # Calculate acceleration (theta'') using the governing equation theta'' = -sin(theta)

        # exact
        theta_double_dot_test = -np.sin(theta_values_test[t-1])

        ## linearized
        #ktheta_double_dot_test = -theta_values_test[t-1]

        # Update velocity and position using simple Euler integration
        theta_dot_values_test[t] = theta_dot_values_test[t-1] + DT_TEST * theta_double_dot_test
        theta_values_test[t] = theta_values_test[t-1] + DT_TEST * theta_dot_values_test[t]

    # Create pairs of (theta, theta_dot) for each time step
    test_data_pairs = np.stack([theta_values_test[:-1], theta_dot_values_test[:-1]], axis=1)
    next_test_data_pairs = np.stack([theta_values_test[1:], theta_dot_values_test[1:]], axis=1)

    # Convert to PyTorch tensors
    test_data_tensor = torch.tensor(test_data_pairs, dtype=torch.float32)
    next_test_data_tensor = torch.tensor(next_test_data_pairs, dtype=torch.float32)

    # Step 2: Forward Prediction and Evaluation
    with torch.no_grad():
        # Encode and predict using the cVAE model
        test_pred, _, _ = cVAE_model.forward(test_data_tensor)
        test_pred = test_pred.numpy()

    # Calculate Error Metrics (RMSE)
    rmse = np.sqrt(np.mean((test_pred - next_test_data_pairs)**2))
    print(f"Root Mean Square Error (RMSE) on test data: {rmse}")

    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title("Theta values: Actual vs Predicted")
    plt.plot(next_test_data_pairs[:, 0], label='Actual')
    plt.plot(test_pred[:, 0], label='Predicted')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Theta_dot values: Actual vs Predicted")
    plt.plot(next_test_data_pairs[:, 1], label='Actual')
    plt.plot(test_pred[:, 1], label='Predicted')
    plt.legend()

    plt.tight_layout()
    plt.show()


