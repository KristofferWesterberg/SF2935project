from torch.optim import Optimizer

class Muon(Optimizer):

    def __init__(self, params):
        super().__init__(params)


    def step(self):
        pass