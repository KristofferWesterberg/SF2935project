class Optimizer():
    '''
    Base class for all optimizers
    '''

    def __init__(self, params, ):
        self.params = params

    def zero_grad(self):
        for p in params:
            p.grad = 0

    def step(self):
        raise NotImplementedError