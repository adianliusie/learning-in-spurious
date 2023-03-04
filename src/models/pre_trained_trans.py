from transformers import ElectraModel, BertModel, BertConfig, RobertaModel, DebertaModel, AutoModel, LongformerModel 
from transformers import BertForMaskedLM, RobertaForMaskedLM, DebertaForMaskedLM
from transformers import OPTModel

def load_transformer(system:str)->'Model':
    """ downloads and returns the relevant pretrained transformer from huggingface """
    if   system == 'bert-base'    : trans_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    elif system == 'bert-rand'    : trans_model = BertModel(BertConfig())
    elif system == 'bert-large'   : trans_model = BertModel.from_pretrained('bert-large-uncased', return_dict=True)
    elif system == 'bert-tiny'    : trans_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    elif system == 'roberta-base' : trans_model = RobertaModel.from_pretrained('roberta-base', return_dict=True)
    elif system == 'roberta-large': trans_model = RobertaModel.from_pretrained('roberta-large', return_dict=True)
    elif system == 'deberta-base' : trans_model = DebertaModel.from_pretrained("microsoft/deberta-base", return_dict=True)
    elif system == 'deberta-large': trans_model = DebertaModel.from_pretrained("microsoft/deberta-large", return_dict=True)
    elif system == 'deberta-xl'   : trans_model = DebertaModel.from_pretrained("microsoft/deberta-xlarge", return_dict=True)
    elif system == 'electra-base' : trans_model = ElectraModel.from_pretrained('google/electra-base-discriminator',return_dict=True)
    elif system == 'electra-large': trans_model = ElectraModel.from_pretrained('google/electra-large-discriminator', return_dict=True)
    elif system == 'longformer'   : trans_model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
    else: raise ValueError(f"invalid transfomer system provided: {system}")
    return trans_model

def load_opt_LM_transformer(system:str)->'Model':
    """ downloads and returns the relevant OPT transformer from huggingface """
    if   system == 'opt-350m'    : trans_model = OPTModel.from_pretrained("facebook/opt-350m")




def load_MLM_transformer(system:str)->'Model':
    """ downloads and returns the relevant pretrained transformer from huggingface """
    if system == 'bert-base'       : trans_model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)
    elif system == 'bert-rand'     : trans_model = BertForMaskedLM(BertConfig())
    elif system == 'bert-large'    : trans_model = BertForMaskedLM.from_pretrained('bert-large-uncased', return_dict=True)
    elif system == 'roberta-base'  : trans_model = RobertaForMaskedLM.from_pretrained('roberta-base', return_dict=True)
    elif system == 'roberta-large' : trans_model = RobertaForMaskedLM.from_pretrained('roberta-large', return_dict=True)
    elif system == 'deberta-base'  : trans_model = DebertaForMaskedLM.from_pretrained("microsoft/deberta-base", return_dict=True)
    elif system == 'deberta-large' : trans_model = DebertaForMaskedLM.from_pretrained("microsoft/deberta-large", return_dict=True)
    elif system == 'deberta-xl'    : trans_model = DebertaForMaskedLM.from_pretrained("microsoft/deberta-xlarge", return_dict=True)
    else: raise ValueError(f"invalid transfomer system provided: {system}")
    return trans_model