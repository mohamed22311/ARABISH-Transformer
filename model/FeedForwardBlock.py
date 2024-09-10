import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dropout: float) -> None:
        """
        Class the creates Feed Forward Netowrk, just simple sturcture of two Linear layers and some dropout.

        Args:
            d_model: int 
                the length of the embeding vector for each token
            dropout: float
                the dropout precentage to avoid overfitting
            d_ff: int
                number of neurons in the first layer.             
        """
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_ff, out_features=d_model)
        )

    def forward(self,x):
        # (batch, seq_len, d_model) -> (batch, seq_len, dff) --> (batch, seq_len, d_model)
        return self.layer_stack(x)
