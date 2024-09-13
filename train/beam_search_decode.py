import torch
import torch.nn as nn

from tokenizers import Tokenizer

from model.Transformer import Transformer
from dataset.causal_mask import causal_mask

def beam_search_decode(model: Transformer,
                       beam_size: int,
                       source_tokens: torch.Tensor,
                       source_mask: torch.Tensor,
                       tokenizer_src: Tokenizer,
                       tokenizer_trg: Tokenizer,
                       max_len: int,
                       device: torch.device):
    
    """
    Function that calculates the multible candidate output of the transformer to choose from.
    (output top `beam_size` hieghts probabilities)

    Args:
        model: Transformer
            Model that should used for inference 
        
        beam_size: int
            Number to indicate how many candidates to consider     

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
        out: List
            List of sequances that are candidates. 
    """
    
    sos_idx = tokenizer_trg.token_to_id('[SOS]')
    eos_idx = tokenizer_trg.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source_tokens, source_mask)

    # Initialize the decoder input with the sos token
    decoder_initial_input = torch.empty(1, 1).fill_(sos_idx).type_as(source_tokens).to(device)

    # Create a candidate list
    candidates = [(decoder_initial_input, 1)]

    while True:

        # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand, _ in candidates]):
            break

        # Create a new list of candidates
        new_candidates = []

        for candidate, score in candidates:

            # Do not expand candidates that have reached the eos token
            if candidate[0][-1].item() == eos_idx:
                continue

            # Build the candidate's mask
            candidate_mask = causal_mask(candidate.size(1)).type_as(source_mask).to(device)
            # calculate output
            out = model.decode(encoder_output, source_mask, candidate, candidate_mask)
            # get next token probabilities
            prob = model.project(out[:, -1])
            # get the top k candidates
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
            for i in range(beam_size):
                # for each of the top k candidates, get the token and its probability
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                # create a new candidate by appending the token to the current candidate
                new_candidate = torch.cat([candidate, token], dim=1)
                # We sum the log probabilities because the probabilities are in log space
                new_candidates.append((new_candidate, score + token_prob))

        # Sort the new candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # Keep only the top k candidates
        candidates = candidates[:beam_size]

        # If all the candidates have reached the eos token, stop
        if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
            break

    # Return the best candidate
    return candidates[0][0].squeeze()
