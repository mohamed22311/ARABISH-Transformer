import torch
from torch.utils.data import Dataset
import datasets
from tokenizers import Tokenizer
from causal_mask import causal_mask
class BilingualDataset(Dataset):
    """
    Class inherit from ```bash torch.utils.data.Dataset``` to create encompase a raw data into a dataset
    valid for use in dataloaders.
    
    The class has a constructor, __len__() method, and __getitem__() method.

    The BilingualDataset class is ment to take tokenized data and store them as dataloaders,
    it adds spical tokens to the raw tokens and keeps each sequance in a fixed constant length.

    """
    def __init__(self,
                 datasaet: datasets.dataset_dict.DatasetDict,
                 src_tokenizer: Tokenizer,
                 trg_tokenizer: Tokenizer, 
                 seq_len: int):
        """
        Constructor for the BilingualDataset class to create dataset instance.
        the constructor saves the attuributes to each given instance and creates some attributes to be used.

        Args:
            dataset: a raw data set of any format like ```bash datasets.dataset_dict.DatasetDict```
                the dataset should be in a format of bilingual data;
                ```bash
                {
                    "id": 1,
                    "score": 1.2498379,
                    "translation": 
                    {
                        "en": "This uncertainty was very difficult for them.”",
                        "ar": "كانت حالة عدم اليقين هذه صعبة للغاية بالنسبة لهم.”"
                    }
                }
                ``` 
            src_tokenizer: Tokenizer
                the tokenizer should be used to tokenize the source language dataset
            trg_tokenizer: Tokenizer
                the tokenizer should be used to tokenize the target language dataset
            seq_len: int
                the maximum sequance length for every input or output.
        Example:
            dataset = BilingualDataset(raw_ds, tokenizer_en, tokenizer_ar, 200)
        
        Returns: None
        """
        super().__init__()

        self.ds = datasaet
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.seq_len = seq_len

        self.sos_token = torch.tensor([src_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([src_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)
        self.eos_token = torch.tensor([src_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)

    def __len__(self):
        """
        Function to calculate the length of the dataset.
        Args: None
        Example:
            BilingualDataset.__len__(ds)
        Returns:
            out: int
                the number of rows in the dataset

        """
        return len(self.ds)
    
    def __getitem__(self,
                    index: int) -> Dict:
        """
        Function to retrive datarow from the dataset and tokenize it.

        Args:
            index: int
                the inxed of the row should be tokenized.
        
        example:
            BilingualDataset.__len__(ds, 5)

        Returns:
            out: Dict
                a dictonary the containts the `encoder input tokens`, `decoder_input_tokens`,
                `ecoder_mask` to mask the padding tokens and keep them away from computations.
                `decoder_mask` to mask the padding tokens and the future tokens form the decoder input,
                `label` the true output of the decoder, `src_text` the actual text without encoding, 
                `trg_text` the actual text after decodeing.

        """
        src_txt, trg_txt = self.ds[index]['translation']
        
        enc_input_tokens = self.src_tokenizer.encode(src_txt).ids
        dec_input_tokens = self.trg_tokenizer.encode(trg_txt).ids
         
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # 2 for SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # 1 for SOS only
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        encoder_input = torch.cat(
            tensors=[
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        decoder_input = torch.cat(
            tensors=[
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        label = torch.cat(
            tensors=[
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        self.mask = (torch.triu(torch.ones((1, decoder_input.size(0), decoder_input.size(0))), diagonal=1).type(torch.int64)) == 0

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
       
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label': label,
            'src_text': src_txt,
            'tgt_text': trg_txt
        }
