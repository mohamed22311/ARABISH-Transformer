
import torch
import torch.nn as nn
from config import get_weights_file_path
from typing import Dict
from model.Transformer import Transformer
def save_model_state(model: Transformer,
                     optimizer: torch.optim.Optimizer,
                     global_step: int,
                     config: Dict,
                     epoch: int) -> None:
    """
    function saves the model state, optimizer state, and the global_step for each epoch

    Args:
        model: Transformer
            model to save its state
        
        optimizer: torch.optim.Optimizer
            optimizer to save its state
        
        global_state: int

        config: Dict
        epoch: int
    """
     # Save the model at the end of every epoch
    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)
