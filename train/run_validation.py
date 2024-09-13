import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torch.utils import tensorboard

from tokenizers import Tokenizer

from model.Transformer import Transformer
from train.greedy_decode import greedy_decode
from train.beam_search_decode import beam_search_decode
from evaluate.evaluate_model_outputs import evaluate_model_outputs

def run_validation(model: Transformer,
                   validation_ds: DataLoader,
                   tokenizer_src: Tokenizer,
                   tokenizer_trg: Tokenizer,
                   max_len: int, 
                   device: torch.device,
                   print_msg,
                   global_step: int,
                   writer: tensorboard,
                   beam_size: int=1,
                   num_examples: int=2):

    """
    Function to make predictions on the validation set to test the model performance.

    The function also evaluate the preidctions useing *weights&biases* , *torchmetrics*,
    and also *tensorboard*.

    Args:
        model: Transformer
            Model that should used for inference 
        
        validation_ds: Dataloader
            the validation dataloader to be used for validation

        tokenizer_src: Tokenizer
            the tokenizer used in the source language
        
        tokenizer_trg: Tokenizer
            the tokenizer used in the target language
        
        max_len: int
            the maximum sequance length allowed

        device: torch.device
            the hardware device that's used in the compuations

        print_msg: function
            function to create a message to appear the at the TQDM bar while training the model
        
        global_step: int 
            variable used to idicate the state globally and used for resuming training

        writer: tensorboard:
            tensorboard writer used in evaluation

        beam_size: int
            if you want beam search -> Number to indicate how many candidates to consider  
            if you want greedy serach -> beam_size = 1 (default = 1)

        num_examples: int 
            Number of samples in the validation data to be tested (default=2)

    Example:
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_trg, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)

    Returns:
        None
    """
    
    model.eval() # Put the model in the evaluation mode

    count = 0 # counter to break when the num_examples reached
    source_texts = [] # List to store the source input text for each sample used in the validation 
    expected = [] # List to store the true output text for each sample used in the validation 
    predicted = [] # List to store the predicted output text for each sample used in the validation 

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad(): # stop calculating the gradients while testing

        for batch in validation_ds: # iterate over each batch in the validation set ot calculate the result
            
            count += 1
            
            encoder_input = batch["encoder_input"].to(device) # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = None

            if beam_size > 1:
                model_out = beam_search_decode(model=model,
                       beam_size=beam_size,
                       source_tokens=encoder_input,
                       source_mask=encoder_mask,
                       tokenizer_src=tokenizer_src,
                       tokenizer_trg=tokenizer_trg,
                       max_len=max_len,
                       device=device)
            else:
                model_out = greedy_decode(model=model, 
                                      source_tokens=encoder_input,
                                      source_mask=encoder_mask,
                                      tokenizer_src=tokenizer_src,
                                      tokenizer_trg=tokenizer_trg,
                                      max_len=max_len,
                                      device=device)

            source_text = batch["src_text"][0]
            target_text = batch["trg_text"][0]
            model_out_text = tokenizer_trg.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

    evaluate_model_outputs(predicted=predicted,
                           expected=expected,
                           global_step=global_step,
                           writer=writer)
    
