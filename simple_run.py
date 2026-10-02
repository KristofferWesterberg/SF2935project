import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import Optimizer
import math

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
        logsigma_x = self.linear_logsigma_x(h)
        return mu_x, logsigma_x
    
    def forward(self, x):
        mu_z, logsigma_z = self.encode(x)
        z = self.reparametrize(mu_z,logsigma_z)
        mu_x,logsigma_x = self.decode(z)
        return mu_x,logsigma_x,mu_z,logsigma_z

    def elbo(self,x):
        """
        Return the positive ELBO value
        """
        mu_x,logsigma_x,mu_z,logsigma_z = self.forward(x)
        kl = -1/2*(1+ logsigma_z - torch.exp(logsigma_z) - mu_z**2 ).sum(dim = 1)
        ev = -1/2*(math.log(2*math.pi)+logsigma_x+(x-mu_x)**2/torch.exp(logsigma_x)).sum(dim = 1)
        loss = (ev - kl).mean()
        return loss

class Adagrad(Optimizer):
    ''' 
    We use the same notation as in "https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adagrad.html"
    '''
    def __init__(self, params, gamma, lambd, tau, eta,eps):
        super().__init__(params)
        self.gamma = gamma
        self.lambd = lambd
        self.tau = tau
        self.eta = eta
        self.eps = eps

    def step(self):
        t = 1
        state_sum = self.tau
        for p in self.params:
            g = p.grad
            gamma_tilde = self.gamma/(1+(t-1)*self.eta)
            if self.lambd != 0:
                g = g + self.lambd*p 
            state_sum += g**2
            p = p - gamma_tilde*gamma_tilde*g/(np.sqrt(state_sum)+self.eps)

class Adam(Optimizer):
    '''
    See https://arxiv.org/pdf/1412.6980 for lambd = 0 and for lambda != 0 (i.e weight decay) see https://arxiv.org/pdf/1711.05101 (we dont test the L_2 reg)
    '''

    def __init__(self,params, alpha, beta1, beta2, eps, lambd = 0):
        super().__init__(params)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lambd = lambd
        self.t = 0 
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self):
        self.t += 1 
        idx = 0
        for p in self.params:
            g = p.grad
            self.m[idx] = self.beta1*self.m[idx] + (1-self.beta1)*g 
            self.v[idx] = self.beta2*self.v[idx] + (1-self.beta2)*g**2
            m_hat = self.m[idx]/(1-self.beta1**self.t)
            v_hat = self.v[idx]/(1-self.beta2**self.t)
            if self.lambd != 0: # Weight decay
                p = p - self.alpha*self.lambd*p
            p = p - self.alpha*m_hat/(np.sqrt(v_hat)+self.eps)
            idx +=1 

# class Muon: Implement later


def train_one_step(optimizer,model,data):

    optimizer.zero_grad()
    # Forward pass in loss function, we minimize so must return negative value
    loss = -model.elbo(data) 
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad
def evaluate_elbo(data_loader, model):
    """
    Evaluate the ELBO for current epoch
    """
    tot_elbo = 0.0
    tot_samples = 0
    for (data,) in data_loader:
        # Rescale to sum of batch elbo loss in case all batches not the same size
        tot_elbo += model.elbo(data).item() * data.size(0)
        tot_samples += data.size(0)
    avg_elbo = tot_elbo / tot_samples
    return avg_elbo

def train(train_loader, test_loader, epcs, model, optimizer):

    training_samples = len(train_loader.dataset)
    samples = []
    elbo_train = []
    elbo_test = []
    for epc in range(epcs):

        print(f'EPOCH: {epc+1} / {epcs}')

        for i, data in enumerate(train_loader):
            loss = train_one_step(optimizer, model, data)
            #print(f' batch: {i+1}, loss: {loss}')

        avg_train_elbo = evaluate_elbo(train_loader, model)
        avg_test_elbo = evaluate_elbo(test_loader, model)
        samples_evaluated = (epc+1) * training_samples

        samples.append(samples_evaluated)
        elbo_train.append(avg_train_elbo)
        elbo_test.append(avg_test_elbo)

    return elbo_train, elbo_test, samples


def load_data(data_path):

    dataset = loadmat(data_path)
    faces = dataset["ff"]
    faces = torch.tensor(faces, dtype=torch.float32).T
    faces = faces /255.0 #transfrom to [0,1] pixel vals
    dataset = TensorDataset(faces)
    return dataset

def parse_data(train_size, dataset, batch_size):       

    train_size = int(train_size*len(dataset))
    test_size = len(dataset)- train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    return train_loader, test_loader


def main():

    # input_dims = 
    # hidden_dims = 
    # latent_dims =







if __name__ == "__main__":
    main()