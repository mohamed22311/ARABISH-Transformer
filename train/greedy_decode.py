import torch
import torch.nn as nn

from tokenizers import Tokenizer

from model.Transformer import Transformer
from dataset.causal_mask import causal_mask

def greedy_decode(model: Transformer, 
                  source_tokens: torch.Tensor,
                  source_mask: torch.Tensor,
                  tokenizer_src: Tokenizer,
                  tokenizer_trg: Tokenizer,
                  max_len: int,
                  device: torch.device):

    """
    Function that calculates the output of the transformer in greedy way.
    (output the hieghts probability only)

    Args:
        model: Transformer
            Model that should used for inference 
        
        source_tokens: torch.Tensor
            the input sequance ids 
        
        source_mask: torch.Tensor
            Mask for the input size to avoid calculations for paddings 
        
        tokenizer_src: Tokenizer
            the tokenizer used in the source language
        
        tokenizer_trg: Tokenizer
            the tokenizer used in the target language
        
        max_len: int
            the maximum sequance length allowed

        device: torch.device
            the hardware device that's used in the compuations
        
    Example:
        model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

    Returns:
        out: torch.Tensor
            a sequance of the highest probabities. 
    """
    
    sos_idx = tokenizer_trg.token_to_id('[SOS]') # Start of sentence id (each token has id in the tokenizer)
    eos_idx = tokenizer_trg.token_to_id('[EOS]') # End of sentence id

    # Precompute the encoder output and reuse it for every output predication
    encoder_output = model.encode(source_tokens, source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source_tokens).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])

        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat(
            tensors=[
                decoder_input, 
                torch.empty(1, 1).type_as(source_tokens).fill_(next_word.item()).to(device)
            ], 
            dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)
