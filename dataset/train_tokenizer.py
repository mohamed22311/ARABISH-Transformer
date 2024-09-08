from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from data_genarator import data_genarator
import datasets

def train_tokenizer(tokenizer: Tokenizer,
                    dataset: datasets.dataset_dict.DatasetDict,
                    lang: str) -> Tokenizer:
    """
    Function to tokenize a certain dataset. 
    The function creates a WordLevelTrainer and train the tokeizer to the given dataset.

    Args:
        tokenizer: Tokenizer
            the tokenizer that should be trained
        dataset: datasets.dataset_dict.DatasetDict
            The dataset that should be tokenized
        lang: str
            The language of the tokenizer (This variable used only for naming)
    
    Example:
        train_tokenizer(english_tokenizer, dataset_raw, 'en')
    
    Returns:
        tokenizer: Tokenizer

        A tokenizer that already tokenized a certain dataset. 

    """
    
    trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
    tokenizer.train_from_iterator(data_genarator(dataset,lang), trainer=trainer)

    return tokenizer
