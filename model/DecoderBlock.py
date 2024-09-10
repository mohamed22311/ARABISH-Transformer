import torch
import torch.nn as nn

from MultiHeadAttentionBlock import MultiHeadAttentionBlock
from FeedForwardBlock import FeedForwardBlock
from ResidualConnection import ResidualConnection

class DecoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        
        """
        Class to define an Decoder Block. 
        The architecture prposed in the paper *Attention is all you need*

        The Decoder block contains a self_attention block, cross_attention block, 
        Residual Connection, LayerNormalization, and FeedForward.

        Args:
            self_attention_block: MultiHeadAttentionBlock
                Multihead Attetion to calculate the attention in the decoder input sequance.

            cross_attention_block: MultiHeadAttentionBlock
                Multihead Attetion to calculate the attention between the Encoder output and Decoder input

            feed_forward_block: FeedForwardBlock
                Linear Network

            dropout: float
                the dropout precentage to avoid overfitting

        """
        
        super().__init__() 

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)]) # 3 Residual Connections 
    
    def forward(self, x, econder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x,econder_output, econder_output,src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
