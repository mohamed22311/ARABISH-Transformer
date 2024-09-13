import torch
import torch.nn as nn
from model.LayerNormalization import LayerNormalization

class Encoder(nn.Module):
    def __init__(self,
                 layers: nn.ModuleList) -> None:
        
        """
        Class that creates a number of Encoder blocks. 
        
        Args:
            layers: nn.ModuleList
                list of encoder blocks
        """
        
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
