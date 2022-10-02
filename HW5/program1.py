import pickle

from nltk import word_tokenize
from nltk.util import ngrams

def preprocess_text(raw_text):
    """returns tokens processed from raw_text"""

    tokens = word_tokenize(raw_text.lower().replace('\n', ''))
    return tokens

def get_val_counts(search_list):
    """returns dictionary of each distinct value in search_list
    and the number of times they occur"""

    val_counts = {}

    for u in search_list:
        val_counts[u] = val_counts.setdefault(u, 0) + 1
    
    return val_counts

def get_unigram_and_bigram_counts(filename):
    """returns tuple of dictionary of unigrams and their frequencies
    and dictionary of bigrams and their frequencies given a file of text"""

    text = ''
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    unigrams = preprocess_text(text)
    bigrams = list(ngrams(unigrams, 2))

    unigrams_counts = get_val_counts(unigrams)
    bigram_counts = get_val_counts(bigrams)

    return unigrams_counts, bigram_counts

def main():
    # making frequency dictionaries
    en_unigram_dict, en_bigram_dict = get_unigram_and_bigram_counts('LangId.train.English')
    fr_unigram_dict, fr_bigram_dict = get_unigram_and_bigram_counts('LangId.train.French')
    it_unigram_dict, it_bigram_dict = get_unigram_and_bigram_counts('LangId.train.Italian')

    # dumping dictionaries
    with open('dicts.pickle', 'wb') as f:
        pickle.dump(en_unigram_dict, f)
        pickle.dump(en_bigram_dict, f)
        pickle.dump(fr_unigram_dict, f)
        pickle.dump(fr_bigram_dict, f)
        pickle.dump(it_unigram_dict, f)
        pickle.dump(it_bigram_dict, f)

if __name__ == '__main__':
    main()