import os

from ..utils.general import get_base_dir

#get the base directory of the framework
BASE_DIR = get_base_dir()
REPO = 'https://github.com/acmi-lab/counterfactually-augmented-data'

#== Automatic downloading utils ===================================================================#
def download_cls_cad():
    """ automatically downloads the classification CAD datasets in parent data folder"""
    os.system(f"svn export {REPO}/trunk/sentiment {BASE_DIR}/data/cad-sentiment")

def download_nli_cad():
    """ automatically downloads the NLI CAD datasets in parent data folder"""
    os.system(f"svn export {REPO}/trunk/NLI {BASE_DIR}/data/cad-nli")
