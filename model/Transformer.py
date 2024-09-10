import torch
import torch.nn as nn

from Encoder import Encoder
from Decoder import Decoder
from InputEmbeddings import InputEmbeddings
from PositionalEncoding import PositionalEncoding
from ProjectionLayer import ProjectionLayer

class Transformer(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 trg_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 trg_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        
        """
        A transformer model.

        User is able to modify the attributes as needed. 
        The architecture is based on the paper “Attention Is All You Need”. 
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin.
        
        Args:
            encoder: Encoder
                Encoder that encodes the input tokens 
            decoder: Decoder
                Decoder to decode the encoder output into tokens 
            src_embed: InputEmbeddings
                The embedding matrix for the source inputs 
            trg_embed: InputEmbeddings
                the embeding matrix for the target inputs 
            src_pos: PositionalEncoding
                the source positional encodings 
            trg_pos: PositionalEncoding
                the target positional encodings 
            projection_layer: ProjectionLayer
                layer to project the decoder output into tokens 
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        Function that calculates the input embeddings then encodes them and add positional encoding.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, trg, trg_mask):
        """
        Function to decode the encoder output
        """
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg, encoder_output, src_mask, trg_mask)
    
    def project(self, x):
        """
        Function that project the decoder output to tokens 
        """
        return self.projection_layer(x)
