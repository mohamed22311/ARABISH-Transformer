import torchmetrics
import wandb
from torch.utils import tensorboard

from evaluate.char_error_rate import char_error_rate_wandb, char_error_rate_tb
from evaluate.word_error_rate import word_error_rate_wandb, word_error_rate_tb
from evaluate.belu_score import belu_score_wandb, belu_score_tb

from typing import List
def evaluate_model_outputs(predicted: List,
                           expected: List,
                           global_step: int,
                           writer: tensorboard,
                           wandb_: bool = False):
    """
    Function that evaluates model outputs through diffetent metrics:
        * Character error rate
        * Word error rate
        * BELU score
    
    the results are shown in 2 formats: 
        * tensorboard
        * weights&biases 
    
    Args:
        predicted: List
            Model output texts 
        
        expected: List
            True output texts
        
        global_step: int
            step of trainging

        wandb_: bool
            True to log in wegiths and biases
        
        writer: tensorboard
            writer to show results
            
    Example:
        evaluate_model_outputs(pred, expect, 5, writer, True)

    Returns: None
    """
    
    if wandb_:
        char_error_rate_wandb(predicted=predicted,
                              expected=expected,
                              global_step=global_step)
        
        word_error_rate_wandb(predicted=predicted,
                              expected=expected,
                              global_step=global_step)
        
        belu_score_wandb(predicted=predicted,
                              expected=expected,
                              global_step=global_step)
    if writer:
        char_error_rate_tb(writer=writer,
                           predicted=predicted,
                           expected=expected,
                           global_step=global_step)
        
        word_error_rate_tb(writer=writer,
                           predicted=predicted,
                           expected=expected,
                           global_step=global_step)
        
        belu_score_tb(writer=writer,
                           predicted=predicted,
                           expected=expected,
                           global_step=global_step)
        
        
        
