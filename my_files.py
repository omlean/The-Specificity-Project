import pandas as pd


def get_text(file):
    """Returns text of input file path as string"""
    if type(file) == pd.Series:
        file = 'data/txt/' + file.filename
    with open(file, 'r') as file:
        text = file.read()
    return text

#########################################################################################