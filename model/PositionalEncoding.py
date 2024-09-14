import math
import torch 
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 seq_len: int,
                 dropout: float) -> None:
        """
        Class to create positional encodings and add them to the input sequance embedings.
        The class creates a positional encoding matrix using the *Sinusoidal Positional Embedding Function*
        ```bash
        Embedding[i, 2k] = sin(position / (10000^(2k / d_model)))

        Embedding[i, 2k+1] = cos(position / (10000^(2k / d_model)))
        ```
        then it takes a sequance that have been tokenized and retrived its embedings, and add positional encoding to the sequance (sequance only without padding).

        Args:
            d_model: int
                the length of the embeding vector for each token
            seq_len: int
                the maximum allowable sequance length
            dropout: float
                the dropout precentage to avoid overfitting
        """
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # positional encoding matrix in the shape of (seq_len, d_model)
        pe_matrix = torch.zeros(seq_len, d_model)
        
        # create vector of shape (seq_len,1)
        positions = torch.arange(0,seq_len, dtype=torch.float).unsqueeze_(1)
        denominators = torch.pow(self=10000.0, 
                                 exponent= (2 * torch.arange(0, d_model//2) ) / d_model) # 10000^(2i/d_model), i is the index of embedding
        # apply sin to even pos
        pe_matrix[:,0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
        # apply cos to odd pos
        pe_matrix[:,1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

        # add batch dimenshion  (1,seq_len,d_model) same dimension as the sequance embeddings (to be added over)
        pe_matrix = pe_matrix.unsqueeze(0)

        self.register_buffer('pe_matrix',pe_matrix)

    def forward(self,x):
        """
        Function that adds positional encodings to the input embeddings only without the padding.
        """
        x = x + (self.pe_matrix[:,:x.shape[1],:]).requires_grad_(False) 
        return self.dropout(x)
