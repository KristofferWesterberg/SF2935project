from torch.optim import Optimizer
import numpy as np

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
            self.m[p] = self.beta1*self.m[p] + (1-self.beta1)*g 
            self.v[p] = self.beta2*self.v[p] + (1-self.beta2)*g**2
            m_hat = self.m[p]/(1-self.beta1**self.t)
            v_hat = self.v[p]/(1-self.v_hat**self.t)
            if self.lambd != 0: # Weight decay
                p = p - self.alpha*self.lambd*p
            p = p - self.alpha*m_hat/(np.sqrt(v_hat)+self.eps)
            idx +=1 