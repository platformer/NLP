from math import log
import pickle
from urllib import request
from urllib.parse import urlparse
import re
import os
import tempfile
import shutil

from bs4 import BeautifulSoup

from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords

out_dir_name = 'out' # name of folder to store scraped text files
knowledge_base_filename = 'knowledge_base' # basename for knowledge base files
start_url = 'https://en.wikipedia.org/wiki/2022_Formula_One_World_Championship'
forbidden_url_terms =  [
    'twitter',
    'facebook',
    'instagram',
    'youtube',
    'mediawiki',
    'imgur',
    'giphy',
    'terms',
    'conditions',
    'privacy',
    'policy',
    'policies',
    'F1_22',
    'F1_Manager_2022'
]

def recursively_scrape_links(url, forbidden_url_terms):
    """Recursively accesses links in the starting url
    and returns a list of 20 urls that include the key term"""

    urls_to_search = [url]
    domain_occurrences = {}
    visited_urls = set()
    ret_urls = []

    while len(urls_to_search) > 0 and len(ret_urls) < 20:
        source = urls_to_search.pop(0) # take next url from queue

        # skip url if already visited
        if source in visited_urls:
            continue
        
        visited_urls.add(source)
        domain = urlparse(source).netloc # get website url

        # skip url if already have 5 urls from this domain
        if domain_occurrences.setdefault(domain, 0) >= 5:
            continue

        includes_forbidden_term = False

        for forbidden in forbidden_url_terms:
            if forbidden.lower() in source.lower():
                includes_forbidden_term = True
                break

        # skip url if it includes forbidden term
        if includes_forbidden_term:
            continue

        # skip url if any errors occur during request
        try:
            html = request.urlopen(source).read().decode('utf-8')
        except:
            continue

        print(f'searching {source}')
        soup = BeautifulSoup(html)

        # scrape urls from this page and add to queue
        for link in soup.find_all('a'):
            link_url = link.get('href')

            # skip this link if it's invalid
            try:
                urlparse(link_url)
            except:
                continue

            if not link_url or link_url[0] == '#':
                # skip link if null or leads nowhere
                continue
            elif link_url[0] == '/':
                # if link is root dir, replace it with domain url
                link_url = 'https://' + domain + link_url

            urls_to_search.append(link_url)
        
        ret_urls.append(source)
        # increment occurrences of this domain
        domain_occurrences[domain] = domain_occurrences[domain] + 1
    
    return ret_urls

def scrape_and_write_text(urls):
    """Given a list of urls, scrapes their text and outputs to files.
    Returns filenames of text files, which are named according to the source urls."""

    # create out directory
    # delete directory first if it exists already
    if (os.path.exists(out_dir_name)):
        tmp = tempfile.mkdtemp(dir=os.path.dirname(out_dir_name))
        shutil.move(out_dir_name, tmp)
        shutil.rmtree(tmp)
    os.makedirs(out_dir_name)

    filenames = []

    for url in urls:
        # skip url if request doesn't work
        try:
            html = request.urlopen(url).read().decode('utf-8')
        except:
            continue

        print(f'scraping {url}')
        soup = BeautifulSoup(html)
        
        # forming file name
        # remove protocol segment
        filename = re.sub(r'^.*//', '', url)
        # replace chars incompatible with Windows filenames
        filename = re.sub(r'<|>|:|"|/|\\|\||\?|\*', '~', filename)
        # add extension
        filename = out_dir_name + '/' + filename + '.txt'
        filenames.append(filename)

        with open(filename, 'w', encoding='utf-8') as f:
            # write text in p tags to file
            for p in soup.select('p'):
                f.write(p.get_text())

    return filenames

def clean_text(filenames):
    """Given names of text files, cleans and outputs text to new files.
    Returns names of new files."""

    cleaned_filenames = []

    for filename in filenames:
        print(f'cleaning {filename}')

        text = ''
        with open(filename, 'r', encoding='utf=8') as old:
            text = old.read()
            # removing newlines and tabs
            text = text.replace('\n', ' ')
            text = text.replace('\t', ' ')
            text = text.replace('.', '. ')
        
        # creating file name
        new_filename = re.sub('.txt$', '_cleaned.txt', filename)
        cleaned_filenames.append(new_filename)

        # writing each sentence on their own line
        with open(new_filename, 'w', encoding='utf-8') as new:
            sents = sent_tokenize(text)
            new.writelines(s + '\n' for s in sents)
    
    return cleaned_filenames

def get_important_terms(filenames):
    """Returns a dict of top 40 most common words for each file according to td-idf"""

    word_counts_per_file = {}
    doc_freqs = {}

    for filename in filenames:
        text = ''
        with open(filename, 'r', encoding='utf=8') as f:
            text = f.read()

        # ignore punctuation and stopwords
        words = [
            t.lower() for t in word_tokenize(text)
            if t.isalpha() and
            t not in stopwords.words('english')
        ]

        # get word freqs in this file
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.setdefault(w, 0) + 1

        word_counts_per_file[filename] = word_counts

        # increment how many docs include each word
        for w in set(words):
            doc_freqs[w] = doc_freqs.setdefault(w, 0) + 1

    top_terms_per_file = {}

    # get top 40 terms in each file according to tf-idf
    for filename in filenames:
        word_counts = word_counts_per_file[filename]
        tf_idf = {w:word_counts[w] / log(len(filenames) / (1 + doc_freqs[w])) for w in word_counts}
        sorted_word_counts = sorted(tf_idf.items(), key=lambda kv: kv[1], reverse=True)
        top_terms_per_file[filename] = [kv[0] for kv in sorted_word_counts[:40]]

    return top_terms_per_file

def build_knowledge_base(filenames, top_terms):
    """Returns a dict mapping each term to a list of sentences with that term"""

    knowledge_base = {}

    for filename in filenames:
        lines = []
        with open(filename, 'r', encoding='utf=8') as f:
            lines = f.readlines()
        
        for line in lines:
            for term in top_terms:
                if term.lower() in line.lower():
                    knowledge_base.setdefault(term, []).append(line)

    return knowledge_base

def main():
    urls = recursively_scrape_links(start_url, forbidden_url_terms)
    filenames = scrape_and_write_text(urls)
    cleaned_filenames = clean_text(filenames)
    top_terms_per_file = get_important_terms(filenames)

    for filename in top_terms_per_file:
        print(f'\nTop terms in {filename}')
        print(top_terms_per_file[filename])

    print()

    top_10_terms = [
        'verstappen',
        'leclerc',
        'fia',
        'ferrari',
        'hamilton',
        'red bull',
        'mercedes',
        'championship',
        'grand prix',
        'race'
    ]

    knowledge_base = build_knowledge_base(cleaned_filenames, top_10_terms)

    # writing knowledge base to human readable file
    print(f'Writing human-readable knowledge base to {out_dir_name}/{knowledge_base_filename}.txt')
    with open(f'{out_dir_name}/{knowledge_base_filename}.txt', 'w', encoding='utf-8') as f:
        for term in top_10_terms:
            f.write(f'Sentences with "{term}":\n')

            for sent in knowledge_base[term]:
                f.write(f'\t{sent}\n')

            f.write('\n')
    
    # pickling knowledge base
    print(f'Pickling knowledge base to {out_dir_name}/{knowledge_base_filename}.pickle')
    with open(f'{out_dir_name}/{knowledge_base_filename}.pickle', 'wb') as f:
        pickle.dump(knowledge_base, f)

if __name__ == '__main__':
    main()