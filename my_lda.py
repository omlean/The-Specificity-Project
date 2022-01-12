import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

import pyLDAvis
import pyLDAvis.gensim as gensimvis
pyLDAvis.enable_notebook()

from my_files import get_text, load_model



class MyCorpus():

    def __init__(self,
                 doc_path_list,
                 clean_function,
                 dictionary=None):

        self.doc_path_list = doc_path_list
        self.dictionary = dictionary
        self.clean_function = clean_function
        if self.dictionary is not None:
            _ = self.dictionary[0]
            self.id2word = self.dictionary.id2token
    
    def __len__(self):
        return len(self.doc_path_list)
    
    def make_dictionary(self, 
                        save_directory=None,
                        file_name="dictionary"):

        print("Creating dictionary...")
        self.dictionary = Dictionary((self.clean_function(get_text(file)) for file in tqdm(self.doc_path_list)))
        _ = self.dictionary[0]
        print("...complete")
        self.id2word = self.dictionary.id2token
        print('id2word created')
        if save_directory is not None:
            self.dict_save_path = save_directory + file_name + '.dict'
            self.dictionary.save(self.dict_save_path)
            print(f"Saved dictionary to {self.dict_save_path}")
    
    def filter_extremes(self,
                        no_below=5,
                        no_above=0.5,
                        keep_n=100000,
                        keep_tokens=None,
                        save_directory=None,
                        file_name="dictionary"):

        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n, keep_tokens=keep_tokens)
        _ = self.dictionary[0]
        self.id2word = self.dictionary.id2token
        if save_directory is not None:
            self.dict_save_path = save_directory + file_name + '.dict'
            self.dictionary.save(self.dict_save_path)
            print(f"Saved trimmed dictionary to {self.dict_save_path}")
            
    def doc2bow(self, string):
        doc = self.clean_function(string)
        return self.dictionary.doc2bow(doc)
    
    def path2bow(self, path):
        with open(path, 'r') as file:
            text = self.clean_function(file.read())
        return self.dictionary.doc2bow(text)
    
    def __iter__(self):
        for doc_path in self.doc_path_list:
            yield self.path2bow(doc_path)
            
#########################################################################################

def make_doc_path_list(directory, extension='.txt'):
    path_list = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(extension)]
    return path_list

#########################################################################################

def doc_topics(corpus, model):
    """Returns DataFrame of topic predictions for all documents in the corpus"""
    num_topics = model.get_topics().shape[0]
    data = np.zeros((len(corpus), num_topics))
    counter = 0
    for doc in tqdm(corpus):
        preds = model[doc]
        for num, val in preds:
            data[counter, num] = val
        counter += 1
    df = pd.DataFrame(data=data, index=corpus.doc_path_list)
    return df

#########################################################################################

def make_pylda(path, corpus):
    """
    Generates and saves a PyLDAVis .html visualisation for the input LDA model.
    Arguments:
    path [str]: full path of .pkl gensim LDA model
    corpus [MyCorpus() object]: corpus object with a dictionary accessible with corpus.dictionary"""
    if not os.path.exists(path):
        print("Input path doesn't exist")
    else:
        print("Loading model")
        with open(path, 'rb') as file:
            model = pickle.load(file)
        print("Preparing gensimvis")
        vis = gensimvis.prepare(model, corpus, corpus.dictionary, sort_topics=False)
        print("Saving html")
        html_path = path.replace(".pkl", ".html")
        pyLDAvis.save_html(vis, html_path)
        print("Complete")
    
#########################################################################################

def label_topics(model_id: str,
                 dict_path: str = 'models/dictionary.dict',
                 dest_folder: str = 'models'):

    """Generates a csv file with topic ids and top words. Labels can be entered manually into the .csv file."""
    
    # check file doesn't already exist; raise an exception otherwise
    filename = f'{model_id}_labels.csv'
    dest_path = os.path.join(dest_folder, filename)
    if os.path.exists(dest_path):
        raise FileExistsError('File already exists at', dest_path)
    
    # load LDA model
    model_path = f'models/{model_id}.pkl'
    model = load_model(model_path)

    # load dictionary
    dictionary = Dictionary.load(dict_path)
    
    # initiate empty dataframe
    df = pd.DataFrame(columns=['topic_no', 'top_words', 'label'])

    num_topics = model.get_topics().shape[0]  # get number of topics
    
    for topic in range(num_topics):
        word_freq_pairs = model.show_topic(topic)  # get list of tuples of top words and frequencies
        
        # add the top words to a list
        top_words = []
        for word, freq in word_freq_pairs:
            top_words.append(word)
        
        # create a Series for addition to the DataFrame
        row = {}
        row['topic_no'] = topic
        row['top_words'] = ', '.join(top_words)
        row['label'] = ''
        df = df.append(pd.Series(row), ignore_index=True)
    
    # save the DataFrame as a csv
    df.to_csv(dest_path, index=False)
    
#########################################################################################