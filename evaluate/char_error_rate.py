import torchmetrics
import wandb
from torch.utils import tensorboard
from typing import List

def char_error_rate_wandb(predicted: List,
                          expected: List,
                          global_step: int):
    """
    Model output evaluation function that calculates 
    the rate of mispredicted cahrecters in the sequance.

    The results are logged in weigths and biases format.

    Args:
        predicted: List
            Model output texts 
        
        expected: List
            True output texts
        
        global_step: int
            step of trainging
    
    Example: 
        char_error_rate_wandb(pred, expect, 5)
    
    Returns: None
    """
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})


def char_error_rate_tb(writer: tensorboard,
                       predicted: List,
                       expected: List,
                       global_step: int):
    """
    Model output evaluation function that calculates 
    the rate of mispredicted cahrecters in the sequance.

    The results are shown in tensorboard.

    Args:
        predicted: List
            Model output texts 
        
        expected: List
            True output texts
        
        global_step: int
            step of trainging
    
    Example: 
        char_error_rate_tb(writer, pred, expect, 5)
    
    Returns: None
    """
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    writer.add_scalar('validation cer', cer, global_step)
    writer.flush()
