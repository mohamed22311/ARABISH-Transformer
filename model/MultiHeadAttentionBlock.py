import torch
import torch.nn as nn
import math
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 h: int,
                 dropout: float) -> None:
        
        """
        Class that creates a Multihead Attention explained in the paper `Attention is all you need`
        
        The math function for it is:
        
        ```bash
        Head(i) = Attention(QW_q, KW_k, VW_v)

        MultiHead(Q,K,V) = Concat(head(1), head(2), ...., head(h))W_o
        ```

        self attention is being computed (i.e., query, key, and value are the same tensor).
        inputs are batched (3D) with batch_first==True

        Args:
            d_model: int 
                the length of the embeding vector for each token
            h: int
                Number of heads 
            dropout: float
                the dropout precentage to avoid overfitting
            
        Returns: 
            out: torch.Tensor
                The Multihead Attention
        """
        
        super().__init__()

        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, 'd_model is not divisble by h'
        self.d_k = d_model // h  # d_k is the length of each head

        self.w_q = nn.Linear(d_model, d_model, bias=False) # Query weight matrix
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Key weigth matrix
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Value weight matrix
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Output weight matrix 

        self.dropout = nn.Dropout(dropout) # Dropout layer to avoid overfitting

    @staticmethod
    def attention(query: torch.Tensor, 
                  key: torch.Tensor, 
                  value: torch.Tensor, 
                  mask: torch.Tensor,
                  dropout: nn.Dropout):
        """
        Function to calculate the attnetion process.

        Args:
            query: torch.Tensor
                The Query embeddings of shape (batch, h, seq_len, d_k)
                Queries are compared against key-value pairs to produce the output. 
                See “Attention Is All You Need” for more details.
            
            key: torch.Tensor
                Key embeddings of shape (batch, h, seq_len, d_k)

            value: torch.Tensor
                Value embeddings of shape  (batch, h, seq_len, d_k)

            mask: torch.Tensor
                 If specified, a mask of shape (batch, h, seq_len) indicating which elements within key to ignore 
                 for the purpose of attention (i.e. treat as “padding”). 
                 Binary masks are supported. For a binary mask, a True value indicates that 
                 the corresponding key value will be ignored for the purpose of attention.

            dropout: nn.Dropout 
                The dropout layer to drop some weights randomly from calculations.

        """

        d_k = query.shape[-1] 

        # Attention_scores shape:  (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        
        if mask is not None: 
            attention_scores.masked_fill_(mask==0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len)

        if dropout is not None: 
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor, 
                v: torch.Tensor, 
                mask: torch.Tensor):
        """
        Fucntion calculates the multihead attention.
        """
        # (batch, seq_len, d_model)  same dimension in and out
        query = self.w_q(q) 
        key = self.w_k(k)
        value = self.w_v(v)

        # Devide it into heads where the length of each head is d_k = d_model//h
        # (batch, seq_len, d_model) ---> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1], self.h, self.d_k).transpose(1,2)

        head_attention, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # (batch, h, seq_len, d_k) ---> (batch, seq_len, h, d_k)---> (batch, seq_len, d_model)
        head_attention = head_attention.transpose(1,2).contiguous().view(head_attention.shape[0], -1, self.h * self.d_k)
        
        return self.w_o(head_attention)
