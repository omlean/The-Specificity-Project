import pandas as pd
import pickle


def get_text(file, path_prefix=''):
    """Returns text of input file path as string"""
    if type(file) == pd.Series:
        file = path_prefix + file.filename
    with open(file, 'r') as file:
        text = file.read()
    return text

#########################################################################################

def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    
#########################################################################################