class OptimizerManager:
    ''''
        https://github.com/thuml/Separate_to_Adapt/blob/fe22775665bfacc809a4367b6451b2c5c5fdfbc6/utilities.py#L115
    '''
    def __init__(self, optims):
        self.optims = optims

    def __enter__(self):
        for op in self.optims:
            op.zero_grad()

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()

        self.optims = None

        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True
    
class OptimWithSheduler:
    '''
        https://github.com/thuml/Separate_to_Adapt/blob/fe22775665bfacc809a4367b6451b2c5c5fdfbc6/utilities.py#L100
    '''
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr = g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1