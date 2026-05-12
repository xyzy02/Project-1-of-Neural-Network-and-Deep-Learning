from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass

class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        # save input for backward
        self.input = X
        # linear transform
        output = X.dot(self.W) + self.b
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # grad: [batch, out_dim] = dL/d(logits), already (p - y)/B for mean CE over batch B
        # dL/dW = X^T @ grad, dL/db = sum_n grad_n (no extra /B)
        self.grads['W'] = self.input.T.dot(grad)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)

        # gradient wrt input to pass to previous layer
        grad_input = grad.dot(self.W.T)
        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=None, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # weights: [out, in, k, k]
        # Default He (Kaiming) normal for ReLU stacks; std=1 的卷积在归一化输入上极易数值过大、
        # ReLU 后特征尺度失控，配合较大 lr 时 logits 难学、loss 易卡在 ln(C) 附近。
        fan_in = max(in_channels * kernel_size * kernel_size, 1)
        if initialize_method is None:
            std = np.sqrt(2.0 / fan_in)
            self.W = np.random.normal(0.0, std, size=(out_channels, in_channels, kernel_size, kernel_size)).astype(np.float64)
            self.b = np.zeros((1, out_channels), dtype=np.float64)
        else:
            self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
            self.b = initialize_method(size=(1, out_channels))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.input_padded = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X
        batch, in_ch, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1

        # pad input
        if p > 0:
            X_padded = np.pad(X, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
        else:
            X_padded = X
        self.input_padded = X_padded

        out = np.zeros((batch, self.out_channels, H_out, W_out))

        for n in range(batch):
            for oc in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * s
                        w_start = j * s
                        patch = X_padded[n, :, h_start:h_start+k, w_start:w_start+k]
                        out[n, oc, i, j] = np.sum(patch * self.W[oc]) + self.b[0, oc]

        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X_padded = self.input_padded
        batch, in_ch, H_p, W_p = X_padded.shape
        _, out_ch, H_out, W_out = grads.shape
        k = self.kernel_size
        s = self.stride

        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros((1, self.out_channels))
        grad_input_padded = np.zeros_like(X_padded)

        for n in range(batch):
            for oc in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * s
                        w_start = j * s
                        g = grads[n, oc, i, j]
                        patch = X_padded[n, :, h_start:h_start+k, w_start:w_start+k]
                        grad_W[oc] += patch * g
                        grad_input_padded[n, :, h_start:h_start+k, w_start:w_start+k] += self.W[oc] * g
                        grad_b[0, oc] += g

        # grad_W / grad_b are sums over batch & spatial positions; upstream grad is already
        # scaled for mean loss (dL/d(conv_out)), so do not divide by batch again.
        self.grads['W'] = grad_W
        self.grads['b'] = grad_b

        # unpad grad input
        if self.padding > 0:
            pad = self.padding
            grad_input = grad_input_padded[:, :, pad:-pad, pad:-pad]
        else:
            grad_input = grad_input_padded

        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


class Flatten(Layer):
    """[batch, C, H, W] -> [batch, C*H*W] for fully-connected layers."""
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        batch = X.shape[0]
        return X.reshape(batch, -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)

    def clear_grad(self):
        pass


class Dropout(Layer):
    """Inverted dropout: forward scales by 1/(1-p); eval forward is identity."""
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = float(p)
        self.mask = None
        self.training = True
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if not self.training or self.p <= 0.0:
            return X
        if self.p >= 1.0:
            return np.zeros_like(X)
        self.mask = (np.random.rand(*X.shape) > self.p).astype(X.dtype)
        scale = 1.0 / (1.0 - self.p)
        return X * self.mask * scale

    def backward(self, grads):
        if not self.training or self.p <= 0.0:
            return grads
        if self.p >= 1.0:
            return np.zeros_like(grads)
        return grads * self.mask / (1.0 - self.p)

    def clear_grad(self):
        pass


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.probs = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        assert predicts.shape[0] == labels.shape[0]
        batch = predicts.shape[0]
        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts

        # numerical stability
        probs_clipped = np.clip(probs, 1e-12, 1.0)
        # negative log likelihood
        correct = probs_clipped[np.arange(batch), labels]
        loss = -np.mean(np.log(correct))

        self.probs = probs
        self.labels = labels

        # compute gradient
        if self.has_softmax:
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(batch), labels] = 1
            self.grads = (probs - one_hot) / batch
        else:
            # predicts are probabilities (softmax already applied in model)
            grads = np.zeros_like(probs)
            grads[np.arange(batch), labels] = -1.0 / (probs_clipped[np.arange(batch), labels] * batch)
            self.grads = grads

        return loss
    
    def backward(self):
        """
        Gradient of softmax + cross-entropy w.r.t. logits is (p - y_one_hot) / batch,
        already stored in self.grads by forward(). Propagate through the model.
        """
        assert self.model is not None and self.grads is not None
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition