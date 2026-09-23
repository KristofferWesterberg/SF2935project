from torch.optim import Optimizer

class Muon(Optimizer):

    def __init__(self, params): #learning rate + mu
        super().__init__(params)


    def step(self):
        pass