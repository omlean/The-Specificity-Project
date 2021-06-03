from tqdm import tqdm

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

from my_files import get_text



class MyCorpus():

    def __init__(self, doc_path_list, clean_function, dictionary=None):
        self.doc_path_list = doc_path_list
        self.dictionary = dictionary
        self.clean_function = clean_function
        if self.dictionary is not None:
            _ = self.dictionary[0]
            self.id2word = self.dictionary.id2token
    
    def __len__(self):
        return len(self.doc_path_list)
    
    def make_dictionary(self, save_directory=None, file_name="dictionary"):
        print("Creating dictionary...")
        self.dictionary = Dictionary((self.clean_function(get_text(file)) for file in tqdm(filelist)))
        _ = self.dictionary[0]
        print("...complete")
        self.id2word = self.dictionary.id2token
        print('id2word created')
        if save_directory is not None:
            self.dict_save_path = save_directory + file_name + '.dict'
            self.dictionary.save(self.dict_save_path)
            print(f"Saved dictionary to {self.dict_save_path}")
    
    def filter_extremes(self, no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None, save_directory=None, file_name="dictionary"):
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n, keep_tokens=keep_tokens)
        if save_directory is not None:
            self.dict_save_path = save_directory + file_name + '.dict'
            self.dictionary.save(self.dict_save_path)
            print(f"Saved trimmed dictionary to {self.dict_save_path}")
            
        
    def get_doc_bow(self, path):
        with open(path, 'r') as file:
            text = self.clean_function(file.read())
        return self.dictionary.doc2bow(text)
    
    def __iter__(self):
        for doc_path in self.doc_path_list:
            yield self.get_doc_bow(doc_path)