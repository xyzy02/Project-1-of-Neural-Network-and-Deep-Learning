from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= 1 - self.init_lr * layer.weight_decay_lambda
                    # 必须原地更新：若用 `params = params - lr*g` 会新建数组，
                    # 与 Linear.conv2D 里 self.W / self.W 与 params['W'] 的引用脱钩，前向仍用旧权重。
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    """
    SGD with momentum: v <- mu * v + grad; param <- param - lr * v
    (same weight_decay multiplicative step on params as SGD, applied before the update).
    """
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self._velocity = {}

    def step(self):
        for layer in self.model.layers:
            if not layer.optimizable:
                continue
            for key in layer.params.keys():
                pid = (id(layer), key)
                if pid not in self._velocity:
                    self._velocity[pid] = np.zeros_like(layer.params[key], dtype=np.float64)
                if layer.weight_decay:
                    layer.params[key] *= 1 - self.init_lr * layer.weight_decay_lambda
                g = layer.grads[key].astype(np.float64, copy=False)
                self._velocity[pid] = self.mu * self._velocity[pid] + g
                layer.params[key] -= self.init_lr * self._velocity[pid]