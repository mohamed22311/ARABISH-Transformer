import torch
import torch.nn as nn

from model.ResidualConnection import ResidualConnection
from model.MultiHeadAttentionBlock import MultiHeadAttentionBlock
from model.FeedForwardBlock import FeedForwardBlock

class EncoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        """
        Class to define an Encoder Block. 
        The architecture prposed in the paper *Attention is all you need*

        The Encoder block contains a self_attention block, Residual Connection, LayerNormalization, and FeedForward.

        Args:
            self_attention_block: MultiHeadAttentionBlock
                Block that calculates the attention scores of each to token to the other tokens in the sequance.

            feed_forward_block: FeedForwardBlock
                Linear Network

            dropout: float
                the dropout precentage to avoid overfitting
        """

        super().__init__()
        
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # 2 Residual Connections


    def forward(self, x, src_mask):

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
