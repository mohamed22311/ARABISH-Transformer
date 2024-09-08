
import datasets

def data_genarator(dataset: datasets.dataset_dict.DatasetDict,
                     lang: str):
    """"
    Genrate all sentences in a given dataset. 
    This function pass through the whole dataset rows as a genrator to yield every row in the translation for a spcific language to be processed
    
    Examples
    Here are some examples of the inputs that are accepted::

        genrator_dataset(dataset_raw, 'en')
        genrator_dataset(dataset_raw, 'ar')


    Args
        dataset : :datasets.dataset_dict.DatasetDict
            The Raw Dataset that should be iterated over.
        lang: str
            The Language argument in the dataset fields 
    Returns
        iter(next(dataset['train]))
    """
    for item in dataset:
        yield item['translation'][lang]
