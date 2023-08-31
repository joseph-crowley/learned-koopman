from torch import nn

# Custom conditional Variational Autoencoder (cVAE) class for dynamical system 
class cVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim=2):
        super(cVAE, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # Adding the Koopman layer for linear dynamics in latent space
        self.koopman_layer = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

    # Encoder network
    def build_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2*self.latent_dim), 
        )

    # Decoder network
    def build_decoder(self):
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
    
    # Sampling from latent space using Gumbel-Softmax (you can also use Gaussian sampling)
    def sample_from_latent(self, z_logits):
        gumbel_noise = nn.functional.gumbel_softmax(z_logits, tau=1, hard=False)
        return gumbel_noise

    # Full forward pass
    def forward(self, inputs):
        x = self.encoder(inputs)
        z_logits, z_log_var = x.chunk(2, dim=-1)
        z = self.sample_from_latent(z_logits)

        # Apply Koopman layer to enforce linear dynamics
        z_next = self.koopman_layer(z)

        reconstructed = self.decoder(z_next)
        return reconstructed, z_logits, z_log_var


