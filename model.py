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
    


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 h: int,
                 dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.h = h
        assert self.d_model % self.h == 0, 'd_model is not divisble by h'
        self.d_k = self.d_model // self.h
        self.w_q = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.w_k = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.w_v = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.w_o = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        #(batch, h, seq_len, d_k) --- > (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) // math.sqrt(d_k)
        if mask is not None: 
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)# (batch, h, seq_len, seq_len)
        if dropout is not None: 
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor, 
                v: torch.Tensor, 
                mask: torch.Tensor):
        query = self.w_q(q) # batch, seq_len, d_model same dim in and out
        key = self.w_k(k)
        value = self.w_v(v)

        #(batch, seq_len, d_model) ---> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # (batch, h, seq_len, d_k) ---> (batch, seq_len, h, d_k)---> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # (batch, seq_len, d_model) ---> (batch, seq_len, d_model)
        return self.w_o(x)
    
class RisdualConnection(nn.Module):
    def __init__(self,
                 dropout: float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.risdual_connection = nn.ModuleList([RisdualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.risdual_connection[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.risdual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self,
                 layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

    