
from tokenizers import Tokenizer
from tokenizer import tokenizer
from train_tokenizer import train_tokenizer
from save_tokenizer import save_tokenizer
from pathlib import Path
import datasets

def make_or_load_tokenizer(tokenizer_name:str, 
                           lang: str,
                           dataset: datasets.dataset_dict.DatasetDict) -> Tokenizer:
    """
    Function to build a WordLevel tokenizer, train it on a given dataset, and save it for later use.
    if it already exits, just load it.

    Args:
        tokenizer_name: str
            the name of the tokenizer file
        lang: str
            The language of the tokenizer (This variable used only for naming)
        dataset: datasets.dataset_dict.DatasetDict
            The dataset that should be tokenized
        
    Example:
        make_or_load_tokenizer('tokenizer', 'en', ds_raw)
    
    Returns:
        tokenizer: Tokenizer

        A tokenizer that already tokenized a certain dataset and saved for later use. 

    """
    tokenizer_path = Path(tokenizer_name.format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = tokenizer()

        train_tokenizer(tokenizer = tokenizer,
                        dataset=dataset,
                        lang=lang)
        save_tokenizer(tokenizer=tokenizer,tokenizer_path=tokenizer_path)
        
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
