import torch
# IAF ###################################################################################
class relu_int(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g
        return
    
    def forward(self, x):
        return torch.where(x <= 0, torch.zeros_like(x), 0.5*torch.pow(x, 2))

class elu_int(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g
        return
    
    def forward(self, x):
        return torch.where(x <= 0, self.g*(torch.exp(x)-x), 0.5*torch.pow(x, 2))
    
class leaky_relu_int(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g
        return
    
    def forward(self, x):
        return torch.where(x <= 0, self.g*torch.pow(x, 2), 0.5*torch.pow(x, 2))