from .op import *
import pickle
import numpy as np

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    hidden_dropout: if > 0, inserts inverted Dropout after each hidden ReLU (not after the last block).
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, hidden_dropout=0.0):
        self.size_list = size_list
        self.act_func = act_func
        self.hidden_dropout = float(hidden_dropout) if hidden_dropout else 0.0

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)
                    if self.hidden_dropout > 0:
                        self.layers.append(Dropout(self.hidden_dropout))

    def set_training(self, training=True):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = bool(training)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        if isinstance(param_list, str):
            with open(param_list, 'rb') as f:
                param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        # Legacy: [sizes, act, {W,b,...}, {W,b,...}, ...] — index 2 is first Linear dict
        if len(param_list) > 2 and isinstance(param_list[2], dict) and 'W' in param_list[2]:
            self.hidden_dropout = 0.0
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i + 2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
            return

        # New typed format: [sizes, act, hidden_dropout, {type:...}, ...]
        self.hidden_dropout = float(param_list[2])
        for item in param_list[3:]:
            t = item.get('type')
            if t == 'Linear':
                layer = Linear(in_dim=item['W'].shape[0], out_dim=item['W'].shape[1])
                layer.W = item['W']
                layer.b = item['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = item.get('weight_decay', False)
                layer.weight_decay_lambda = item.get('lambda', 1e-8)
                self.layers.append(layer)
            elif t == 'ReLU':
                self.layers.append(ReLU())
            elif t == 'Dropout':
                self.layers.append(Dropout(p=item.get('p', 0.5)))
            else:
                continue
        
    def save_model(self, save_path):
        uses_dropout = any(isinstance(L, Dropout) for L in self.layers)
        if uses_dropout or self.hidden_dropout > 0:
            param_list = [self.size_list, self.act_func, self.hidden_dropout]
            for layer in self.layers:
                if isinstance(layer, Linear):
                    param_list.append({
                        'type': 'Linear',
                        'W': layer.params['W'], 'b': layer.params['b'],
                        'weight_decay': layer.weight_decay, 'lambda': layer.weight_decay_lambda,
                    })
                elif isinstance(layer, ReLU):
                    param_list.append({'type': 'ReLU'})
                elif isinstance(layer, Dropout):
                    param_list.append({'type': 'Dropout', 'p': layer.p})
        else:
            param_list = [self.size_list, self.act_func]
            for layer in self.layers:
                if layer.optimizable:
                    param_list.append({'W': layer.params['W'], 'b': layer.params['b'],
                                       'weight_decay': layer.weight_decay, 'lambda': layer.weight_decay_lambda})
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

def build_mnist_cnn(weight_decay_lambda=1e-4):
    """
    Simple MNIST CNN: two strided conv blocks (implicit downsampling) + FC.
    Spatial sizes with k=5, s=2, p=0: 28x28 -> 12x12 -> 4x4 (see conv2D output formula).
    First conv: 8 filters; second conv: 16 filters (28->12->4 spatial, k=5, s=2).
    """
    layers = [
        conv2D(1, 8, kernel_size=5, stride=2, padding=0),
        ReLU(),
        conv2D(8, 16, kernel_size=5, stride=2, padding=0),
        ReLU(),
        Flatten(),
        Linear(16 * 4 * 4, 10),
    ]
    # 分类头：与 256 维特征匹配的量级（全 N(0,1) 时 logits 方差过大，softmax 易饱和/难优化）
    fc = layers[-1]
    fin, fout = fc.W.shape
    std_fc = np.sqrt(2.0 / (fin + fout))
    fc.W = np.random.normal(0.0, std_fc, size=(fin, fout)).astype(np.float64)
    fc.b = np.zeros((1, fout), dtype=np.float64)
    fc.params["W"] = fc.W
    fc.params["b"] = fc.b
    for layer in layers:
        if getattr(layer, 'optimizable', False):
            layer.weight_decay = True
            layer.weight_decay_lambda = weight_decay_lambda
    return layers


class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, layers=None):
        # layers: a list of layer instances (conv2D, ReLU, Linear, Flatten, ...)
        if layers is None:
            self.layers = build_mnist_cnn()
        else:
            self.layers = layers

    def set_training(self, training=True):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = bool(training)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        # param_list can be a filepath or a loaded list
        if isinstance(param_list, str):
            with open(param_list, 'rb') as f:
                param_list = pickle.load(f)

        self.layers = []
        for item in param_list:
            t = item.get('type', None)
            if t == 'conv2D':
                layer = conv2D(in_channels=item['in_channels'], out_channels=item['out_channels'], kernel_size=item['kernel_size'], stride=item.get('stride',1), padding=item.get('padding',0))
                layer.W = item['W']
                layer.b = item['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = item.get('weight_decay', False)
                layer.weight_decay_lambda = item.get('lambda', 1e-8)
                self.layers.append(layer)
            elif t == 'Linear':
                W = item['W']
                in_dim, out_dim = W.shape[0], W.shape[1]
                layer = Linear(in_dim=in_dim, out_dim=out_dim)
                layer.W = item['W']
                layer.b = item['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = item.get('weight_decay', False)
                layer.weight_decay_lambda = item.get('lambda', 1e-8)
                self.layers.append(layer)
            elif t == 'ReLU':
                self.layers.append(ReLU())
            elif t == 'Flatten':
                self.layers.append(Flatten())
            elif t == 'Dropout':
                self.layers.append(Dropout(p=item.get('p', 0.5)))
            else:
                # unknown type, skip
                continue
        return self
        
    def save_model(self, save_path):
        # Save a serializable description of the layer sequence
        param_list = []
        for layer in self.layers:
            if isinstance(layer, conv2D):
                param_list.append({'type':'conv2D', 'in_channels':layer.in_channels, 'out_channels':layer.out_channels, 'kernel_size':layer.kernel_size, 'stride':layer.stride, 'padding':layer.padding, 'W':layer.params['W'], 'b':layer.params['b'], 'weight_decay':layer.weight_decay, 'lambda':layer.weight_decay_lambda})
            elif isinstance(layer, Linear):
                param_list.append({'type':'Linear', 'W':layer.params['W'], 'b':layer.params['b'], 'weight_decay':layer.weight_decay, 'lambda':layer.weight_decay_lambda})
            elif isinstance(layer, ReLU):
                param_list.append({'type':'ReLU'})
            elif isinstance(layer, Flatten):
                param_list.append({'type': 'Flatten'})
            elif isinstance(layer, Dropout):
                param_list.append({'type': 'Dropout', 'p': layer.p})
            else:
                # skip unknown layers
                continue

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)