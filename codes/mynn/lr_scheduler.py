from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    """
    Multiply optimizer learning rate by gamma when step_count reaches each milestone.
    Milestones are compared after incrementing step_count (first call -> step_count==1).
    """
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.milestones = sorted(int(m) for m in milestones)
        self.gamma = gamma
        self._milestone_idx = 0

    def step(self) -> None:
        self.step_count += 1
        while (
            self._milestone_idx < len(self.milestones)
            and self.step_count >= self.milestones[self._milestone_idx]
        ):
            self.optimizer.init_lr *= self.gamma
            self._milestone_idx += 1

class ExponentialLR(scheduler):
    pass