import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self,
                 d_model: int,
                 vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.d_model)
        
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 seq_len: int,
                 dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # positional encoding matrix in the shape of (seq_len, d_model)
        self.pe = torch.zeros(self.seq_len, self.d_model)
        
        # create vector of shape (seq_len,1)
        self.position = torch.arange(0,self.seq_len, dtype=torch.float).unsqueeze(dim=1)
        self.div_term = torch.exp(torch.arange(0,self.d_model,2).float() * (-math.log(10000.0) / self.d_model))
        # apply sin to even pos
        self.pe[:,0::2] = torch.sin(self.position * self.div_term)
        # apply cos to odd pos
        self.pe[:,1::2] = torch.cos(self.position * self.div_term)

        # add batch dimenshion  (1,seq_len,d_model)
        self.pe = self.pe.unsqueeze(dim=0)

        self.register_buffer('pe',self.pe)

    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self,
                 eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multliplicative 
        self.bias = nn.Parameter(torch.zeros(1)) # addatitive 

    
    def forward(self,x: torch.Tensor):
        mean = x.mean(dim= -1, keepdim=True) # mean cancel dim that's applied but keepdim doesn't 
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dropout: float) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(in_features=d_ff, out_features=d_model)
    
    def forward(self,x):
        # (batch, seq_len, d_model) -> (batch, seq_len, dff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

