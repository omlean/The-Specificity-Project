import pandas as pd
import re
import nltk
from nltk.corpus import stopwords


def load_metadata(path='data/metadata.csv.gz'):
    df = pd.read_csv(path, sep='\t', compression='gzip', dtype={'year': int})
    return df

# drop unwanted documents, e.g. 'front matter'
def drop_non_research(df):
    drop_titles = ['Front Matter', 'Back Matter', 'Volume Information', 'Obituary Notices of Fellows Deceased', 'Obituary Notices']
    return df[df.title.apply(lambda x: x not in drop_titles)]


def clean_text(string, split_hyphenated=False):
    """Cleans and tokenizes input text"""
    s = string.lower() # lowercase
    
    # remove punctuation, excluding hyphens
    punc = '''!()[]{};:'"\,<>./?¿@#$%^&*_—~+=|©°•■§ß±'''
    s = ''.join([c for c in s if c not in punc])
    
    s = re.sub('\\n', ' ', s) # remove newlines
    s = re.sub(r'\b\d+\b', ' ', s) # remove numbers
    s = re.sub(r'\b\w\b', '', s) # remove single letters
    s = re.sub(r'\b[ivxlcmd]+\b', r'', s) # remove Roman numerals
    
    # join words that have exactly one hyphen and space between them, e.g. "pure- form" to "pure-form"
    s = re.sub(r'(\b- \b)|(\b -\b)', '-', s)
    # delete hyphens that don't connect two words
    s = re.sub(r'\W(-)|(-)\W', '', s)
    # if split_hyphenated, split hyphenated words into two separate words
    if split_hyphenated:
        s = re.sub(r'\b(-)\b', ' ', s)
    
#     Remove stopwords
    stop_words = stopwords.words('english')
    for word in ['pp', 'viz', 'vs', 'ie', 'eg']: # add words to stopword list
        stop_words.append(word)
    s = ' '.join([word for word in s.split() if word not in stop_words])

    s = nltk.word_tokenize(s)
    return s