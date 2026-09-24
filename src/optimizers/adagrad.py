from optim import Optimizer
import numpy as np

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
