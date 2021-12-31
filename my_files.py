import xml.etree.ElementTree as ET
from collections import defaultdict
import os
import pandas as pd
import pickle


def get_metadata(xml_file_path):
    """Parses metadata from input xml file and returns as a Series"""
    tree = ET.parse(xml_file_path)# parse xml file
    article_meta = tree.getroot()[0][1] # get article metadata branch
    metadata = defaultdict(str) # create empty dict for metadata
    
    # file name
    txt_file_path = os.path.split(xml_file_path)[1]
    txt_file_path = os.path.splitext(txt_file_path)[0] + '.txt'
    metadata['filename'] = txt_file_path
    
    # get article IDs
    article_ids = article_meta.iter('article-id')
    for i in article_ids:
        metadata[i.attrib['pub-id-type']+'-id'] = i.text

    # get title
    for i in article_meta.iter('article-title'):
        metadata['title'] += i.text

    # get authors
    authors = []
    for contrib in article_meta.iter('contrib'):
        for string_name in contrib.findall('string-name'):
            surname = string_name.findtext('surname')
            given_names = string_name.findtext('given-names')
            name = f'{surname}, {given_names}'
            authors.append(name)
    metadata['authors'] = '; '.join(authors)

    # get year, month, volume, issue, page numbers
    metadata['year'] = article_meta.find('pub-date').find('year').text
    metadata['month'] = article_meta.find('pub-date').find('month').text
    metadata['volume'] = article_meta.find('volume').text
    try:
        metadata['issue'] = article_meta.find('issue').text
    except:
        metadata['issue'] = ''
    try:
        metadata['pages'] = str(article_meta.find('fpage').text) + "-" + str(article_meta.find('lpage').text)
    except:
        metadata['pages'] = ''

    # get keywords
    try:
        metadata['keywords'] = '; '.join([kwd.text for kwd in article_meta.find('kwd-group').findall('kwd')])
    except AttributeError:
        metadata['keywords'] = ''

    # get abstract
    abstract = article_meta.iter('abstract')
    for a in abstract:
        for i in a:
            metadata['abstract'] += i.text + " "

    return pd.Series(metadata)


#########################################################################################

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