from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# topics per document
def plot_topics_per_doc(doc_topics):
    """Plots histogram of topics per document from doc_topics dataframe"""
    topic_count = doc_topics.apply(lambda x: (x > 0).sum(), axis=1).values
    plt.hist(topic_count, bins=max(topic_count))
    plt.show()
    
#########################################################################################    

def plot_topic_words(model):
    """Plots probabilities of words in each topic"""
    topic_words = model.get_topics()
    topic_words.sort(axis=1)
    topic_words = topic_words[:,::-1]
    
    plt.rcParams["figure.figsize"] = (20,15)
    fig, ax = plt.subplots()
    for i in range(topic_words.shape[0]):
        topic = np.log10(topic_words[i])
        ax.plot(range(100), topic[:100])
    plt.xlabel('words (ordered by probability)', fontsize=18)
    plt.ylabel('log10 probability', fontsize=18)
    fig.show()

#########################################################################################

def word_report(word, corpus, model, top_topics=5):
    """Prints summary values and plots for the input word relative to the input corpus and model."""
    word_id = corpus.dictionary.token2id[word]
    term_topics = model.get_term_topics(word_id, minimum_probability=0)
    term_topics.sort(key=lambda x: x[1], reverse=True)
    j = top_topics if top_topics <= len(term_topics) else len(term_topics)
    for i in range(j):
        topic, prob = term_topics[i]
        print(f'Topic {topic}: {prob}:')
        topic_words = model.print_topic(topic, topn=10)
        topic_words = re.findall(r'\*"([^"]+)"', topic_words)
        for word in topic_words:
            print(word)
        print('\n')
        
#########################################################################################

def word_compare(data, words, model, dictionary):
    """Input: 
    data [pd.DataFrame]: DataFrame
    words [list]: List of words
    model [Gensim LdaModel() object]
    dictionary [Gensim dictionary object]: Dictionary used to train the model (must have dictionary.token2id attribute)
    Output: Plot of p values per topic for each word in the list"""
    
    word_ids = []
    wordlist = []
    for word in words:
        try:
            word_ids.append(dictionary.token2id[word])
            wordlist.append(word)
        except KeyError:
            print(f"'{word}' not found in dictionary - skipped")
    word_topics = model.get_topics()[:,word_ids]
    topic_labels = [col for col in data.columns if ":" in col]
    data = pd.DataFrame(word_topics, columns=wordlist, index=topic_labels).sort_values(words[0], ascending=False)
    
    data.plot.bar(figsize=(20,10))
    plt.ylabel('p', size=20)
    plt.xticks(size=15)
    plt.legend(prop={'size': 18})
    plt.show()
    
#########################################################################################