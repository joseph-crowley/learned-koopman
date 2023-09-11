import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np

from cVAE import cVAE

# Define the loss function
def oscillator_loss(y_true, y_pred, z_logits, z_log_var, lambda1=1):
    # Reconstruction loss
    reconstruction_loss = F.mse_loss(y_pred, y_true, reduction='none').sum(dim=-1)
    
    # KL-divergence loss
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_logits**2 - z_log_var.exp(), dim=-1)
    
    # Combined loss
    return reconstruction_loss.mean() + lambda1 * kl_loss.mean()

if __name__ == "__main__":
    from torch.optim import Adam
    from constants import HIDDEN_DIM, LATENT_DIM, INPUT_DIM

    cVAE_model = cVAE(HIDDEN_DIM, LATENT_DIM, INPUT_DIM)

    # check if cVAE has been trained
    try:
        with open('cVAE_model.pth', 'rb') as f:
            cVAE_model.load_state_dict(torch.load(f))
    except FileNotFoundError:
        print("cVAE model not found, initializing a new model.")
    except RuntimeError:
        print("cVAE model parameters mismatch, initializing a new model.")

    optimizer = Adam(cVAE_model.parameters())

    print("Complete cVAE Model for Harmonic Oscillator")
    print(cVAE_model)

    # Load data
    data_pairs = np.load('data_pairs.npy')
    next_data_pairs = np.load('next_data_pairs.npy')

    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data_pairs, dtype=torch.float32)
    next_data_tensor = torch.tensor(next_data_pairs, dtype=torch.float32)

    # Create a PyTorch dataset
    dataset = TensorDataset(data_tensor, next_data_tensor)

    # calculate the batch size
    batch_size = len(dataset) // 10

    # Create a PyTorch data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    n_epochs = 600  # Number of epochs

    # Loss history for plotting
    loss_history = []

    try:
        # Training loop
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for x_t_batch, x_t_plus_1_batch in data_loader:
                optimizer.zero_grad()

                # Encode x_t and x_{t+1} to get z_t and z_{t+1}
                z_t_logits, z_t_log_var = cVAE_model.encoder(x_t_batch).chunk(2, dim=-1)
                z_t = cVAE_model.sample_from_latent(z_t_logits)

                # Apply Koopman layer on z_t to get z'
                z_t_prime = cVAE_model.koopman_layer(z_t)

                # Decode z' to get the predicted x_{t+1}
                x_t_plus_1_pred, _, _ = cVAE_model.forward(x_t_batch)

                # Compute loss
                loss = oscillator_loss(x_t_plus_1_batch, x_t_plus_1_pred, z_t_logits, z_t_log_var)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Average epoch loss
            epoch_loss /= len(data_loader)
            loss_history.append(epoch_loss)

            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    
    except KeyboardInterrupt:
        print("Interrupted by user. Saving model in current state...")

    # Save the trained model
    torch.save(cVAE_model.state_dict(), 'cVAE_model.pth')

    print("Training complete!")

    # Plot loss history
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.show()

    # plot log(difference(loss)) on the same plot with axis on the right
    plt.plot(np.log(np.abs(np.diff(loss_history))))
    plt.xlabel('Epoch')
    plt.ylabel('Log(Difference(Loss))')
    plt.title('Log(Difference(Loss)) History')
    plt.show()
