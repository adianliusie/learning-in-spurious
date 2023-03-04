import os
import csv
from typing import List, Dict, Tuple, TypedDict

from .download import BASE_DIR, download_cls_cad
from .load_classification_hf import _remove_html_tags

class SingleText(TypedDict):
    """Output example formatting (only here for documentation)"""
    text : str
    label : int

CAD_CLS_DATA = ['k-imdb', 'k-imdb-cad', 'k-yelp', 'k-semeval', 'k-amazon']

def load_cad_cls_data(data_name)->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    """ loading sentiment classification datsets used in Kaushik et. Al 2021,
        'Learning the Difference that Makes a Difference with Counterfactually Augmented Data' """
    if   data_name == 'k-imdb'      : train, dev, test = load_imdb()
    elif data_name == 'k-imdb-cad'  : train, dev, test = load_imdb_cad()
    elif data_name == 'k-imdb-comb' : train, dev, test = load_imdb_combine()
    elif data_name == 'k-yelp'      : train, dev, test = load_yelp()
    elif data_name == 'k-semeval'   : train, dev, test = load_semeval()
    elif data_name == 'k-amazon'    : train, dev, test = load_amazon()
    else: raise ValueError(f"invalid single text dataset name: {data_name}")
    return train, dev, test

#== Individual Data set Loader for CAD datasets ===================================================#
def load_imdb()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    # Download data if not already downloaded    
    data_path = f"{BASE_DIR}/data/cad-sentiment/orig"
    if not os.path.isdir(data_path): download_cls_cad()
    
    train, dev, test = load_tsv_datasets(data_path)
    return train, dev, test

def load_imdb_cad()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    # Download data if not already downloaded
    data_path = f"{BASE_DIR}/data/cad-sentiment/new"
    if not os.path.isdir(data_path): download_cls_cad()
    
    train, dev, test = load_tsv_datasets(data_path)
    return train, dev, test

def load_imdb_combine()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    # Download data if not already downloaded
    data_path = f"{BASE_DIR}/data/cad-sentiment/combined"
    if not os.path.isdir(data_path): download_cls_cad()
    
    train, dev, test = load_tsv_datasets(data_path)
    return train, dev, test

def load_yelp()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    data_path = f"{BASE_DIR}/data/cad-eval/yelp_balanced.tsv"
    if not os.path.isfile(data_path): raise ValueError('Need to download dataset first')
    
    test = load_tsv_file(data_path)
    return [], [], test

def load_semeval()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    data_path = f"{BASE_DIR}/data/cad-eval/semeval_balanced.tsv"    
    if not os.path.isfile(data_path): raise ValueError('Need to download dataset first')
    
    test = load_tsv_file(data_path)
    return [], [], test

def load_amazon()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    data_path = f"{BASE_DIR}/data/cad-eval/amazon_balanced.tsv"
    if not os.path.isfile(data_path): raise ValueError('Need to download dataset first')

    test = load_tsv_file(data_path)
    return [], [], test

#== Util functions that load and prepare datasets =================================================#
def load_tsv_datasets(data_path:str)->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    paths = [f'{data_path}/{split}.tsv' for split in ['train', 'dev', 'test']]
    
    train, dev, test = [load_tsv_file(path) for path in paths]
    train, dev, test = _remove_html_tags(train, dev, test)
    return train, dev, test

def load_tsv_file(path:str)->List[dict]:
    output = []
    sentiment_map = {'Negative':0, 'Positive':1}
    with open(path) as file:
        tsv_file = csv.reader(file, delimiter="\t")

        _ = next(tsv_file)
        for line in tsv_file:
            sentiment, text = line
            label = sentiment_map[sentiment]
            output.append({
                'text':text,
                'label':label
            })
    return output