import torch
import torch.nn as nn
from model.LayerNormalization import LayerNormalization

class ResidualConnection(nn.Module):
    """
    Class to create a Residual Connection to add the input to previous layers to thier outputs.
    
    The idea of Residual Connection came from ResNets. 

    ResNet networks are characterized by skip connections, or shortcuts to jump over some layers, 
    this trick gives the ability to train really deep networks without caring about 
    The problem of gradient vanishing.
    """
    def __init__(self,
                 dropout: float)->None:
        """
        Class to create a Residual Connection to add the input to previous layers to thier outputs.
    
        The idea of Residual Connection came from ResNets. 

        ResNet networks are characterized by skip connections, or shortcuts to jump over some layers, 
        this trick gives the ability to train really deep networks without caring about 
        The problem of gradient vanishing.

        Args: 
            dropout: float
                the dropout precentage to avoid overfitting
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """
        Function makes skip connection and apply layer normlaization. 
        """
        return x + self.dropout(sublayer(self.norm(x)))
