from transformers import ElectraTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
from transformers import AutoTokenizer, LongformerTokenizer, DebertaTokenizer

def load_tokenizer(system:str)->'Tokenizer':
    """ downloads and returns the relevant pretrained tokenizer from huggingface """
    if system   == 'bert-base'     : tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif system == 'bert-rand'     : tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif system == 'bert-large'    : tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")
    elif system == 'bert-tiny'     : tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    elif system == 'roberta-base'  : tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif system == 'roberta-large' : tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
    elif system == 'debert-base'   : tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
    elif system == 'deberta-large' : tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-large")
    elif system == 'deberta-xl'    : tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-xlarge")
    elif system == 'electra-base'  : tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")
    elif system == 'electra-large' : tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
    elif system == 'longformer'    : tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    else: raise ValueError(f"invalid transfomer system provided: {system}")
    return tokenizer
       
