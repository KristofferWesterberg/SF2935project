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
        self.linear_mu_x = nn.Linear(hidden_dims, input_dims)
        self.linear_logsigma_x = nn.Linear(hidden_dims, input_dims)

    def encode(self, x):
        h = torch.tanh(self.linear_encoder(x))
        return self.linear_mu(h), self.linear_logsigma(h)

    def reparametrize(self, mu, logsigma):
        std = torch.exp(0.5*logsigma)
        eps = torch.randn_like(std)
        return mu + eps*std # = z
    
    def decode(self,z):
        h = torch.tanh(self.linear_decoder(z))
        mu_x = torch.sigmoid(self.linear_mu_x(h))
        logsigma_x = torch.sigmoid(self.linear_logsigma_x(h))
        return mu_x, logsigma_x
    
    def forward(self, x):
        mu_z, logsigma_z = self.encode(x)
        z = self.reparametrize(mu,logsigma)
        mu_x,logsigma_x = decode(z)
        return mu_x,logsigma_x,mu_z,logsigma_z

    def loss_function(self,x):
        mu_x,logsigma_x,mu_z,logsigma_z = self.forward(x)
        kl = 1/2*(torch.exp(logsigma_z) + mu_z**2 -1-logsigma_z).sum(dim = 1)
        ev = 1/2*(torch.log(2*torch.pi)+logsigma_x+(x-mu_x)/np.exp(logsigma_x)).sum(dim = 1)
        loss = -kl - ev
        return loss
