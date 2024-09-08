
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

def tokenizer() -> Tokenizer:
    """
    Function to create WordLevel Tokenizer. 
    The function has adds 4 special tokens for the dataset:
        1- [UNK]: Unknown token for tokens that are not recognized in the dataset
        2- [PAD]: Padding token to keep the size of sequance constant
        3- [SOS]: Start Of Sentence token to indicate the sentance start
        4- [EOS]: End Of Sentence token to indicaaate the sentence end

    Example:
        toeknizer = tokenizer()
    
    Returns:
        tokenizer: Tokenizer

        A word-level tokenizer tokenizer. 

    """
    
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer =Whitespace()
    return tokenizer
