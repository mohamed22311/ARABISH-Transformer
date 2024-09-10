import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 vocab_size: int) -> None:
        """
        Class of projection layer -> a linear layer followed by a softmax to 
        output the probability of each token. 

        Args:
            d_model: int
                The length of the vector to represnt each token
            vocab_size: int
                the number of tokens to embedded in the matrix

        """

        super().__init__()
        self.proj = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self,x):
        # (Batch, seq_len, d_model) ---> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
