import torchmetrics
import wandb
from torch.utils import tensorboard
from typing import List

def belu_score_wandb(predicted: List,
                     expected: List,
                     global_step: int):
    """
    Model output evaluation function that calculates 
    the rate of mispredicted the BLUE score in the sequance.

     BLEU (Bilingual Evaluation Understudy) is 
     a score used to evaluate the translations performed by a machine translator.

    The results are logged in weigths and biases format.

    Args:
        predicted: List
            Model output texts 
        
        expected: List
            True output texts
        
        global_step: int
            step of trainging
    
    Example: 
        belu_score_wandb(pred, expect, 5)
    
    Returns: None
    """
    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})



def belu_score_tb(writer: tensorboard,
                  predicted: List,
                  expected: List,
                  global_step: int):
    """
    Model output evaluation function that calculates 
    the rate of mispredicted the BLUE score in the sequance.

     BLEU (Bilingual Evaluation Understudy) is 
     a score used to evaluate the translations performed by a machine translator.

    The results are shown in tensorboard.

    Args:
        predicted: List
            Model output texts 
        
        expected: List
            True output texts
        
        global_step: int
            step of trainging
    
    Example: 
        belu_score_tb(writer, pred, expect, 5)
    
    Returns: None
    """
    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    writer.add_scalar('validation BLEU', bleu, global_step)
    writer.flush()
