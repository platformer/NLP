import sys
from random import randint

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(raw_text):
    '''Given a string, finds all alphabetical tokens that are not stopwords and are longer than 5 characters.
    Then, lemmatizes all such tokens, and finds all unique nouns in the text.

    Returns 2-tuple of (tokens, nouns).'''
    raw_text = raw_text.lower()
    tokens = nltk.word_tokenize(raw_text)

    tokens = [t for t in tokens if t.isalpha() and 
              t not in stopwords.words('english') and
              len(t) > 5]
    
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    unique_lemmas = set(lemmas)

    tags = nltk.pos_tag(unique_lemmas)
    noun_lemmas = [tag[0] for tag in tags if tag[1][0] == 'N']

    print(f'Number of tokens: {len(tokens)}')
    print(f'Number of nouns:  {len(noun_lemmas)}')

    return (tokens, noun_lemmas)

def guessing_game(word_list):
    '''Starts a guessing game with word_list as its dictionary'''

    def setup_round():
        '''Initializes round of game.

        Returns 3-tuple of (new word, blank display string, empty list of guesses).'''
        word = word_list[randint(0, len(word_list) - 1)]
        display_str = ['_ ' for _ in range(len(word))]
        guessed_chars = []
        return word, display_str, guessed_chars

    points = 5
    word, display_str, guessed_chars = setup_round()

    while points >= 0:
        print(''.join(display_str))

        guess = ''
        # keep asking for a guess until it's a single character
        while len(guess) != 1:
            guess = input('Guess a character: ').strip()
            if len(guess) > 1:
                print("Too long")
        
        # quit if guess is !
        if (guess == '!'):
            print('Goodbye!')
            return

        # ignore guess if is was guessed already
        if (guess in guessed_chars):
            print(f'You already guessed that. Score is {points}')
        # update display string and score if guess is correct
        elif (guess in word):
            for i in range(len(word)):
                if word[i] == guess:
                    display_str[i] = guess + ' '
            
            points += 1

            # start new round if word is complete
            if '_ ' not in display_str:
                print(''.join(display_str))
                print('You solved it!\n')
                print(f'Current score is {points}\n')
                print('Guess another word')
                word, display_str, guessed_chars = setup_round()
            else:
                print(f'Right! Score is {points}')
                guessed_chars += guess
        # update score if guess is incorrect
        else:
            points -= 1
            guessed_chars += guess
            print(f"Sorry, guess again. Score is {points}")
    
    print(f'\nYou lost. The word was {word}')

def main(argv):
    # checking if program arguments were passed
    if len(argv) < 2:
        print('ERROR: must specify relative path to data')
        return
    
    # read entire file into variable
    text = ''
    with open(argv[1], 'r') as f:
        text = f.read()

    # get all alphabetical tokens
    words = [t for t in nltk.word_tokenize(text.lower()) if t.isalpha()]
    # get all unique words
    unique_words = set(words)
    print(f"Lexical diversity: {len(unique_words) / len(words):.2f}")

    tokens, nouns = preprocess_text(text)
    # get pairs of nouns and their frequencies
    noun_counts = {n : 0 for n in nouns}

    for t in tokens:
        if t in noun_counts:
            noun_counts[t] += 1

    # sort nouns by frequency
    sorted_noun_counts = sorted(noun_counts.items(), key=lambda kv: kv[1], reverse=True)
    # get top 50 most common nouns
    common_noun_counts = sorted_noun_counts[:50]

    print("The 50 most common nouns and their frequencies:")
    print(common_noun_counts)

    guessing_game([word[0] for word in common_noun_counts])


if __name__ == '__main__':
    main(sys.argv)