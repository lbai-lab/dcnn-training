"""
This source code is adapted from DeepXDE:
https://github.com/lululxvi/deepxde
under the GNU Lesser General Public License found in:
https://github.com/lululxvi/deepxde/blob/master/LICENSE
"""

import torch
from deepxde import config
from deepxde.nn import activations
from deepxde.nn import initializers

# The parent class for MLP in deepXDE
class NN(torch.nn.Module):
    """Base class for all neural network modules."""

    def __init__(self):
        super().__init__()
        self.regularizer = None
        self._input_transform = None
        self._output_transform = None

    def apply_feature_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features).
        """
        self._input_transform = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """
        self._output_transform = transform

    def num_trainable_parameters(self):
        """Evaluate the number of trainable parameters for the NN."""
        return sum(v.numel() for v in self.parameters() if v.requires_grad)
    
# The class for MLP in deepXDE with option of enabling batch normalization
class DeepXDE_FNN(NN):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer, batchnorm=False):
        super().__init__()
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.batchnorms = None
        if batchnorm: self.batchnorms = [ torch.nn.BatchNorm1d(layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for j, linear in enumerate(self.linears[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
            if self.batchnorms: x = self.batchnorms[j](x)
        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
    
if __name__ == "__main__":
    a = DeepXDE_FNN([3] + [40] * 6 + [2], "tanh", "Glorot normal", batchnorm=True)