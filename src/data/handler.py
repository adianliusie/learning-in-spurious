import random
import os

from typing import List
from types import SimpleNamespace
from tqdm import tqdm
from copy import deepcopy
from functools import lru_cache

from .load_classification_cad import load_cad_cls_data, CAD_CLS_DATA
from .load_classification_hf import load_hf_cls_data, HF_CLS_DATA
from .load_pair_hf import load_hf_pair_data, HF_NLI_DATA, HF_PARA_DATA
from ..models.tokenizers import load_tokenizer
from ..utils.general import get_base_dir, save_pickle, load_pickle
from .biasing import create_balanced_data, create_biased_balanced_data, get_biased_data, get_unbiased_data

BASE_PATH = get_base_dir()
BASE_CACHE_PATH = f"{BASE_PATH}/tokenize-cache/"

#== Main DataHandler class ========================================================================#
class DataHandler:
    def __init__(self, trans_name:str, prompt_finetuning:bool=False):
        self.trans_name = trans_name
        self.tokenizer = load_tokenizer(trans_name)
        self.prompt_finetuning = prompt_finetuning

    #== Data processing (i.e. tokenizing text) ====================================================#
    @lru_cache (maxsize=10)
    def prep_split(self, data_name:str, mode:str, bias:str=None, lim=None):
        split = self.load_split(
            data_name=data_name, 
            mode=mode, 
            bias=bias, 
            lim=lim
        )
        if is_classification(data_name):
            data = self._prep_ids(split)
        elif is_nli(data_name):
            data = self._prep_ids_nli(split)
        elif is_paraphrasing(data_name):
            data = self._prep_ids_para(split)
        else: raise ValueError(f"invalid data set: {data_name}")
        return data

    @lru_cache(maxsize=10)
    def prep_data(self, data_name, bias:str=None, lim=None):
        train, dev, test = self.load_data(data_name=data_name, bias=bias, lim=lim)
        if is_classification(data_name):
            train, dev, test = [self._prep_ids(split) for split in [train, dev, test]]
        elif is_nli(data_name):
            train, dev, test = [self._prep_ids_nli(split) for split in [train, dev, test]]
        elif is_paraphrasing(data_name):
            train, dev, test = [self._prep_ids_para(split) for split in [train, dev, test]]
        else: raise ValueError(f"invalid data set: {data_name}")
        
        return train, dev, test
    
    #== Different tokenization methods for different task set up ==================================#
    def _prep_ids(self, split_data:List[SimpleNamespace]):
        split_data = deepcopy(split_data)
        for ex in tqdm(split_data):
            if self.prompt_finetuning:
                # add full stop to text if there's no full stop
                ex.text = ex.text.strip()
                text = ex.text + '.' if ex.text[-1] not in ('.', '!', '?') else ex.text 

                # get the prompt input ids
                ex.input_ids = self.tokenizer(f'{text} It was {self.tokenizer.mask_token}.').input_ids
                assert ex.input_ids.count(self.tokenizer.mask_token_id) == 1
                ex.mask_position = ex.input_ids.index(self.tokenizer.mask_token_id)
            else:
                ex.input_ids = self.tokenizer(ex.text).input_ids 

        return split_data

    def _prep_ids_nli(self, split_data:List[SimpleNamespace]):
        split_data = deepcopy(split_data)
        for ex in tqdm(split_data):
            if self.prompt_finetuning:
                # remove punctuation at the end of the first sentence
                text_1 = ex.text_1.strip().rstrip('?,.!')

                # get the prompt input ids
                ex.input_ids = self.tokenizer(f'{text_1} ? {self.tokenizer.mask_token}, {ex.text_2}').input_ids
                assert ex.input_ids.count(self.tokenizer.mask_token_id) == 1
                ex.mask_position = ex.input_ids.index(self.tokenizer.mask_token_id)
            else:
                text_1_ids = self.tokenizer(ex.text_1).input_ids
                text_2_ids = self.tokenizer(ex.text_2).input_ids
                ex.input_ids = text_1_ids + text_2_ids[1:]
        return split_data

    def _prep_ids_para(self, split_data:List[SimpleNamespace]):
        split_data = deepcopy(split_data)
        for ex in tqdm(split_data):
            if self.prompt_finetuning:
                # remove punctuation at the end of the first sentence
                text_1 = ex.text_1.strip().rstrip('?,.!')

                # get the prompt input ids
                ex.input_ids = self.tokenizer(f'{text_1} {self.tokenizer.mask_token}, {ex.text_2}').input_ids
                assert ex.input_ids.count(self.tokenizer.mask_token_id) == 1
                ex.mask_position = ex.input_ids.index(self.tokenizer.mask_token_id)
            else:
                text_1_ids = self.tokenizer(ex.text_1).input_ids
                text_2_ids = self.tokenizer(ex.text_2).input_ids
                ex.input_ids = text_1_ids + text_2_ids[1:]
        return split_data 

    #== Data loading utils ========================================================================#
    @classmethod
    @lru_cache(maxsize=10)
    def load_data(cls, data_name:str, bias:str=None, lim=None):
        train, dev, test = cls._load_data(data_name, lim)

        if bias is None: 
            train, dev, test = cls._load_data(data_name, lim)
        else:
            train, dev, test = cls._load_data(data_name)
            train = cls.bias_data(data=train, bias=bias, lim=lim)
            dev   = cls.bias_data(data=dev, bias=bias, lim=lim)
            test  = cls.bias_data(data=test, bias=bias, lim=lim)

        return train, dev, test
    
    @classmethod
    @lru_cache(maxsize=10)
    def load_split(cls, data_name:str, mode:str, bias:str=None, lim=None):
        split_index = {'train':0, 'dev':1, 'test':2}        
        if bias is None: 
            data = cls._load_data(data_name, lim)[split_index[mode]]
        else:
            data = cls._load_data(data_name)[split_index[mode]]
            data = cls.bias_data(
                data=data, 
                bias=bias, 
                lim=lim
            )
        return data

    @staticmethod
    def bias_data(data, bias=None, lim=None):
        # parse on whether there is 1 or 2 arguments
        if bias == 'balanced':
            output = create_balanced_data(data=data, lim=lim)

        # full bias 
        elif bias[:4] == 'all-':
            _, bias_name, *bias_bounds = bias.split('-')
            bias_bounds = [float(i) for i in bias_bounds]
            output = get_biased_data(data=data, bias_name=bias_name, bias_bounds=bias_bounds, lim=lim)

        # unbiased points
        elif bias[:4] == 'inv-':
            _, bias_name, *bias_bounds = bias.split('-')
            bias_bounds = [float(i) for i in bias_bounds]
            output = get_unbiased_data(data=data, bias_name=bias_name, bias_bounds=bias_bounds, lim=lim)
  
        # balanced, at controllable strength
        else:
            bias_name, bias_acc, *bias_bounds = bias.split('-')
            bias_acc = float(bias_acc)
            bias_bounds = [float(i) for i in bias_bounds]
        
            output = create_biased_balanced_data(
                data=data, 
                bias_name=bias_name, 
                bias_acc=bias_acc, 
                bias_bounds=bias_bounds, 
                lim=lim
            )
        return output

    @staticmethod
    @lru_cache(maxsize=10)
    def _load_data(data_name:str, lim=None):
        if   data_name in HF_CLS_DATA   : train, dev, test = load_hf_cls_data(data_name)
        elif data_name in CAD_CLS_DATA  : train, dev, test = load_cad_cls_data(data_name)
        elif data_name in HF_NLI_DATA   : train, dev, test = load_hf_pair_data(data_name)
        elif data_name in HF_PARA_DATA  : train, dev, test = load_hf_pair_data(data_name)
        else: raise ValueError(f"invalid dataset name: {data_name}")
          
        train, dev, test = to_namespace(train, dev, test)

        if lim:
            train = rand_select(train, lim)
            dev   = rand_select(dev, lim)
            test  = rand_select(test, lim)
            
        return train, dev, test
    
#== Misc utils functions ============================================================================#
def is_classification(data_name:str):
    return data_name in HF_CLS_DATA + CAD_CLS_DATA

def is_nli(data_name:str):
    return data_name in HF_NLI_DATA

def is_paraphrasing(data_name:str):
    return data_name in HF_PARA_DATA

def rand_select(data:list, lim:None):
    if data is None: return None
    random_seed = random.Random(1)
    data = data.copy()
    random_seed.shuffle(data)
    return data[:lim]

def to_namespace(*args:List):
    def _to_namespace(data:List[dict])->List[SimpleNamespace]:
        return [SimpleNamespace(ex_id=k, **ex) for k, ex in enumerate(data)]

    output = [_to_namespace(split) for split in args]
    return output if len(args)>1 else output[0]

def get_num_classes(data_name:str):
    if is_classification(data_name): output = 2
    elif is_nli(data_name)         : output = 3
    return output 
    