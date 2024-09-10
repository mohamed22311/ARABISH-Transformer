import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """
    Applies Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in the paper *Layer Normalization*

    """
    def __init__(self,
                 eps: float = 10**-6):
        
        """
        Applies Layer Normalization over a mini-batch of inputs.

        This layer implements the operation as described in the paper *Layer Normalization*

        Args:
            eps: float
                a value added to the denominator for numerical stability. Default: 1e-6

        Variabels:
            alpha: the learnable weights of the module
            bias: the learnable bias of the module

        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multliplicative 
        self.bias = nn.Parameter(torch.zeros(1)) # addatitive 

    
    def forward(self,x: torch.Tensor):
        """
        Apply Layer Normalization by calculating the mean and STD over the layer, 
        and apply the normalization equation:
        ```bash
        layer_norm = Alpha * X` / (std + bais + eps)
        ``` 
        """
        mean = x.mean(dim= -1, keepdim=True) # keepdim=True -> the mean function cancel dim that's applied to but keepdim doesn't 
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias
