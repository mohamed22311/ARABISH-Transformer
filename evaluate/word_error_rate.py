import torchmetrics
import wandb
from torch.utils import tensorboard
from typing import List
def word_error_rate_wandb(predicted: List,
                          expected: List,
                          global_step: int):
    """
    Model output evaluation function that calculates 
    the rate of mispredicted words in the sequance.

    The results are logged in weigths and biases format.

    Args:
        predicted: List
            Model output texts 
        
        expected: List
            True output texts
        
        global_step: int
            step of trainging
    
    Example: 
        word_error_rate_wandb(pred, expect, 5)
    
    Returns: None
    """
    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})



def word_error_rate_tb(writer: tensorboard,
                       predicted: List,
                       expected: List,
                       global_step: int):
    """
    Model output evaluation function that calculates 
    the rate of mispredicted words in the sequance.

    The results are shown in tensorboard.

    Args:
        predicted: List
            Model output texts 
        
        expected: List
            True output texts
        
        global_step: int
            step of trainging
    
    Example: 
        word_error_rate_tb(writer, pred, expect, 5)
    
    Returns: None
    """
    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    writer.add_scalar('validation wer', wer, global_step)
    writer.flush()

