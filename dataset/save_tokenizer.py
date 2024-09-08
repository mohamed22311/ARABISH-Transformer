from tokenizers import Tokenizer
from pathlib import Path
def save_tokenizer(tokenizer: Tokenizer,
                   tokenizer_path: Path) -> None:
    """
    Function to save a tokenizer in a json file.
    The tokenizer be saved in a naming convention (tokenizername_language.json)
    Args:
        tokenizer: Tokenizer
            The tokenizer to be saved 
        tokenizer_path: Path
            the Path that the tokenizer should be saved it.
    
    Example:
        tokenizer('tokenizer', 'en) ----> tokeinzer_en.json
    
    Returns:
        None
    """
    tokenizer.save(str(tokenizer_path))
