
import math
import torch 
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self,
                 d_model: int,
                 vocab_size: int):
        """
        Class to create an embedding matrix of the size 
        of the vocabulary and the dimension vector for each token.
        
        Args:
            d_model: int
                The length of the vector to represnt each token
            vocab_size: int
                the number of tokens to embedded in the matrix

        Example: 
            embed = InputEmbeddings(512, 10000)

        Returns: None
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self,x):
        """
        function to embed each token passed through, it uses torch.nn.Embedding
        Args:
            x: torch.tensor
                the token should be embeded
        Returns:
            the embeding vector 
            torch.tensor
        """
        return self.embedding(x) * math.sqrt(self.d_model)
