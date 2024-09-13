import datasets
from tokenizers import Tokenizer
def calculate_max_seq_len(dataset: datasets.dataset_dict.DatasetDict,
                          src_tokenizer: Tokenizer,
                          trg_tokenizer: Tokenizer,
                          lang_src: str,
                          lang_trg: str,
                          offset: int) -> int:
    """
    Function to calculate the maximum allowable sequance length in the transformer acrhitecture.
    it's calculated to be the longest sequance in the dataset + offset

    Args:
        dataset: datasets.dataset_dict.DatasetDict
            the raw dataset to search through 
        src_tokenizer: Tokenizer
            the tokenizer should be used to tokenize the source language dataset
        trg_tokenizer: Tokenizer
            the tokenizer should be used to tokenize the target language dataset 
        lang_src: str
            the name of source language in the dataset
        lang_trg: str
            the name of target language in the dataset
        offset: int
            the number of offest above the max sequance in the dataset should be added to indicate max sequance
    
    Example:
        calculate_max_seq_len(raw_ds, tokenizer_en, tokenizer_ar, 'en', 'ar', 10)

    Returns: 
        out: int
            the maximum allowable sequance length 
    """

    max_len_src = 0
    max_len_tgt = 0
        
    for item in dataset['train']:
        src_tokens = src_tokenizer.encode(item['translation'][lang_src]).ids
        trg_tokens = trg_tokenizer.encode(item['translation'][lang_trg]).ids
        max_len_src = max(max_len_src, len(src_tokens))
        max_len_tgt = max(max_len_tgt, len(trg_tokens))

    print(f'Max Length of source sentence: {max_len_src}\nMax Length of Target sentence: {max_len_tgt}')

    return max(max_len_src,max_len_tgt) + offset
