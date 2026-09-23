import torch
import torch.nn as nn

class VAE(nn.Module):

    def __init__(self, input_dims, hidden_dims, latent_dims):

        super(VAE,self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.latent_dims = latent_dims

        self.linear_encoder = nn.Linear(input_dims,hidden_dims)
        self.linear_mu = nn.Linear(hidden_dims, latent_dims)
        self.linear_logsigma = nn.Linear(hidden_dims,latent_dims)

        self.linear_decoder = nn.Linear(latent_dims,hidden_dims)
        # Skapa lager för likelihood eller för decode_mu, decode_sigma?
        # self.likelihood = nn.Linear(hidden_dims,input_dims)

    def encode(self, x):
        h = torch.tanh(self.linear_encoder(x))
        return self.linear_mu(h), self.linear_logsigma(h)

    def reparametrize(self, mu, logsigma):
        std = torch.exp(0.5*logsigma)
        eps = torch.randn_like(std)
        return mu + eps*std # = z
    
    def decode(self,z):
        pass # output x eller mu_x, logsigma_x?

    def forward(self, x):
        mu, logsigma = self.encode(x)
        z = self.reparametrize(mu,logsigma)
        pass
        # Decode part?

    def loss_function(self):
        pass
