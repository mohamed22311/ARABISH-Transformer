import torch
import torch.nn as nn

from model.InputEmbeddings import InputEmbeddings
from model.Transformer import Transformer
from model.PositionalEncoding import PositionalEncoding
from model.MultiHeadAttentionBlock import MultiHeadAttentionBlock
from model.DecoderBlock import DecoderBlock
from model.EncoderBlock import EncoderBlock
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.FeedForwardBlock import FeedForwardBlock
from model.ProjectionLayer import ProjectionLayer


def build_transformer(src_vocab_size: int,
                      trg_vocab_size: int,
                      src_seq_len: int,
                      trg_seq_len: int,
                      d_model: int = 512,
                      N: int = 6,
                      h: int = 8,
                      dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:
    
    """
    Function that build a transformer model. 

    Args:
        src_vocab_size: int
            Size of the source vocab 
        
        trg_vocab_size: int
            Size of the target vocab

        src_seq_len: int
            Maximum sequance length for the source inputs 
        
        trg_seq_len: int
            Maximum sequance length for the target inputs 

        d_model: int = 512
            Size of the embedding vector for each token in the embedding matrix
        
        N: int = 6
            Number of blocks in the Encoder and Decoder 
        
        h: int = 8
            Number of Heads in the Multihead attetion blocks

        dropout: float = 0.1
            Dropout precentage to drop while calculaations randomly to avoid overfitting
                     
        d_ff: int = 2048
            Number of hidden neourns in the projection layer
    
    Example:
        transformer = build_transformer(1000, 1000, 300, 300)

    Returns:
        transformer: Transformer
    """
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    trg_embed = InputEmbeddings(d_model, trg_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = PositionalEncoding(d_model, trg_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the deocder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder 
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, trg_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, projection_layer)

    # Intialize the parameters using Xavier intialization 
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
