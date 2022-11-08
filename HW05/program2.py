import pickle

from nltk import word_tokenize
from nltk.util import ngrams

def bigram_prob(bigram, unigram_dict, bigram_dict, dict_size):
    """returns the LaPlace smoothed probability of a bigram
    given the unigram and bigram dictionaries"""

    return (
        (bigram_dict.setdefault(bigram, 0) + 1)
        / (unigram_dict.setdefault(bigram[0], 0) + dict_size)
    )

def test(
    en_unigram_dict,
    en_bigram_dict,
    fr_unigram_dict,
    fr_bigram_dict,
    it_unigram_dict,
    it_bigram_dict,
):
    # calculating vocabulary size
    dict_size = len(en_unigram_dict)    \
                + len(fr_unigram_dict)  \
                + len(it_unigram_dict)
    
    num_correct = 0
    total_lines = 0
    incorrect_lines = []

    with open('LangId.test', 'r', encoding='utf-8') as test, \
        open('LangId.sol', 'r', encoding='utf-8') as sol,    \
        open('LangId.guess', 'w', encoding='utf-8') as out:
        while (line := test.readline()):
            # getting bigrams in line of test file
            line = line.lower().replace('\n', '')
            tokens = word_tokenize(line)
            line_bigrams = list(ngrams(tokens, 2))

            en_prob = 1
            fr_prob = 1
            it_prob = 1

            # calculating probability of bigrams being from each language
            for b in line_bigrams:
                en_prob *= bigram_prob(b, en_unigram_dict, en_bigram_dict, dict_size)
                fr_prob *= bigram_prob(b, fr_unigram_dict, fr_bigram_dict, dict_size)
                it_prob *= bigram_prob(b, it_unigram_dict, it_bigram_dict, dict_size)

            guess = ''
            real = sol.readline().split()[1] # getting true answer

            # picking most likely language
            if en_prob > fr_prob and en_prob > it_prob:
                guess = 'English'
            elif fr_prob > en_prob and fr_prob > it_prob:
                guess = 'French'
            else:
                guess = 'Italian'

            # writing prediction
            out.write(f'{total_lines + 1} {guess}\n')

            if guess == real:
                num_correct += 1
            else:
                incorrect_lines.append(f'Line {total_lines + 1:{3}}: guessed {guess}, was {real}')

            total_lines += 1
    
    print(f'Accuracy: {num_correct / total_lines}\n')
    print('Incorrect guesses:')
    
    # printing lines numbers of incorrect guesses
    for line in incorrect_lines:
        print(line)

def main():
    en_unigram_dict = {}
    en_bigram_dict = {}
    fr_unigram_dict = {}
    fr_bigram_dict = {}
    it_unigram_dict = {}
    it_bigram_dict = {}
    
    # reading dictionaries
    with open('dicts.pickle', 'rb') as f:
        en_unigram_dict = pickle.load(f)
        en_bigram_dict = pickle.load(f)
        fr_unigram_dict = pickle.load(f)
        fr_bigram_dict = pickle.load(f)
        it_unigram_dict = pickle.load(f)
        it_bigram_dict = pickle.load(f)

    test(
        en_unigram_dict,
        en_bigram_dict,
        fr_unigram_dict,
        fr_bigram_dict,
        it_unigram_dict,
        it_bigram_dict
    )
        

if __name__ == '__main__':
    main()