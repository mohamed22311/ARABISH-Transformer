
import torch
def causal_mask(size: int) -> torch.Tensor:
    """
    Function to provide a mask that masks or covers the future inputs and keep them away from the attention calculations.
    Args:
        size: int
            the size of the mask 
    Returns: 
        mask: torch.Tensor
            mask of True only at the future input (mask is squared (size*size) and has outer batch dimension)
    """
    mask = torch.triu( torch.ones(size=(1,size,size)), diagonal=1).type(torch.int64)
    return mask == 0
