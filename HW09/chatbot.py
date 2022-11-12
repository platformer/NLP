import re
import pickle
import os.path
from enum import Enum
from random import choice as randchoice
from random import random as randreal
from itertools import chain
from threading import Thread
from queue import Queue
from types import SimpleNamespace

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import spacy

import praw
from praw import Reddit
from prawcore import NotFound
from praw.models import Comment
from praw.models.comment_forest import CommentForest

from better_profanity import profanity

# info for accessing reddit API
reddit_app_client_id = 'CYoFVr0KS1ZrZM04yoBJAQ'
reddt_app_secret = 'DiGdLNw6P7fZonL7MS5rkP0MzZmbSA'
reddit_app_user_agent = 'python:nlp-chatbot-homework:v0.1 (by /u/nlp-throwaway)'

# directory and file names
data_dirname = 'chatbot_data'
user_data_dirname = 'user_data'
subreddit_data_dirname = 'subreddit_data'
users_filename = 'users'
user_subs_filename = 'subs'
user_likes_filename = 'likes'
user_dislikes_filename = 'dislikes'
user_am_stmts_filename = 'am'
user_my_stmts_filename = 'my'

prompt = "$> "
reset_phrase = 'reset'

# number of posts to scrape per sort per subreddit
num_posts_per_cat = 30

# max number of subs to allow
sub_limit = 15

# initial greetings
greetings = [
    "Hello",
    "Sup",
    "Heya",
    "Hey",
    "Hi",
]

# greetings once user has been confirmed
return_greetings = [
    'Oh hey! Nice to see you!',
    'I thought I recognized you!',
    "What's up!",
]

# responses to subreddit names
generic_responses = [
    'Cool',
    'Nice',
    "Oh I've been there before",
    'Got it',
    'Okay',
    'Noted',
]

# responses to user repeating previous message
repetition_responses = [
    'I heard you the first time',
    'I think you already said that',
    "I don't like repeating myself",
    "You're trying to trip me up, aren't ya?"
]

# generic responses
canned_respones_to_statement = [
    'Tell me more',
    'Interesting',
    'okay',
    'Not sure I get it, but sure',
    'Go on',
    'I see',
    "How's that?",
]

canned_respones_to_question = [
    "I'm not sure"
    'Not sure how to answer that',
    'what do you think?',
    'what are your thoughts on that?',
]

# responses to user exit
goodbyes = [
    'Goodbye',
    'See you later!',
    'cya',
    'bye!',
]


# phrases user may use to end subreddit naming
ending_phrases = [
    "that's it",
    "that's all",
    "that is it",
    "that is all",
    'done',
    'finish',
    'stop',
    'end',
    'not really',
]

# phrases user may use to say they are returning
affirmative_phrases = [
    'have',
    'sure',
    'think so',
    'yes',
    'yep',
    'yeah',
    'yea',
    'positive',
    'certainly',
    'of course',
    'certain',
    'probably',
]

# phrases user may use to say they are not returning
non_affirmative_phrases = [
    "haven't",
    'have not',
    'no',
    'nope',
    'na',
    'nah',
    'negative',
    'not positive',
    'not sure',
    "don't think",
    'do not think',
    'not sure',
    'probably not',
    'definitely not',
    'not really',
    'not at all',
    'not certain',
    'nada',
]

# words that indicate a message is a question
question_words = [
    'is',
    'does',
    'do',
    'what',
    'when',
    'where',
    'who',
    'why',
    'what',
    'how',
]

# commands for ending program
end_commands = [
    'quit',
    'exit',
]


class MemoryType(Enum):
    LIKE = 1
    DISLIKE = 2
    AM = 3
    MY = 4


def print_system_message(msg: str) -> None:
    """Prints `msg` in unique color for system messages."""
    print(f'\033[36m{msg}\033[0m')


def print_chatbot_message(msg: str) -> None:
    """Prints `msg` in unique color for chatbot responses."""
    print(f'\033[33m{msg}\033[0m')


def make_reddit() -> Reddit:
    """Makes handle to Reddit API using global credentials."""

    return praw.Reddit(
        client_id=reddit_app_client_id,
        client_secret=reddt_app_secret,
        user_agent=reddit_app_user_agent,
        check_for_async=False
    )


def scrape_post(id: str, q: 'Queue[Comment]') -> None:
    """Given post id and a Queue, scrapes comments and puts result on `q`."""

    lemmatizer = WordNetLemmatizer()
    sw = stopwords.words('english')
    reddit = make_reddit()
    post = reddit.submission(id)

    comments = post.comments
    comments.replace_more(limit=0)
    comments_flat = comments.list()

    for comment in comments_flat:
        # remove unnecessary properties
        del comment.author
        del comment.body_html
        del comment.created_utc
        del comment.distinguished
        del comment.edited
        del comment.id
        del comment.is_submitter
        del comment.link_id
        del comment.parent_id
        del comment.permalink
        del comment.saved
        del comment.score
        del comment.stickied
        del comment.subreddit
        del comment.subreddit_id

        comment.lemmatized = [
            lemmatizer.lemmatize(word)
            for word in word_tokenize(comment.body.lower())
            if word not in sw
        ]
    
    # use post as root of comment tree
    post_root = SimpleNamespace()
    post_root.body = post.title + ' ' + post.selftext
    post_root.replies = comments
    post_root.lemmatized = [
        lemmatizer.lemmatize(word)
        for word in word_tokenize(post_root.body.lower())
        if word not in sw
    ]

    q.put(post_root)


def scrape_subreddit(sub_name: str, q: 'Queue[str]') -> None:
    """
    Given subreddit name and a Queue, scrapes top posts, writes to file,
    and puts filename on `q`.
    """

    # skip scraping if file already exists
    if os.path.isfile(f'{data_dirname}/{subreddit_data_dirname}/{sub_name}'):
        q.put(sub_name)
        return

    reddit = make_reddit()
    sub = reddit.subreddit(sub_name)
    post_ids = set()

    threads = []
    q2 = Queue()

    # multithread post scraping
    for post in chain(
        sub.top(time_filter='all', limit=num_posts_per_cat),
        sub.top(time_filter='year', limit=num_posts_per_cat)
    ):
        if post.id in post_ids:
            continue

        post_ids.add(post.id)

        t = Thread(target=scrape_post, args=[post.id, q2])
        t.start()
        threads.append(t)

    post_comments = []

    for _ in threads:
        post_comments.append(q2.get())
    for t in threads:
        t.join()

    with open(f'{data_dirname}/{subreddit_data_dirname}/{sub_name}', 'wb') as f:
        pickle.dump(post_comments, f)

    q.put(sub_name)


def is_not_affirmative(stmt: str) -> bool:
    """Returns whether `stmt` appears non-affirmative."""

    stmt = stmt.lower()
    words = [
        phrase for phrase in non_affirmative_phrases
        if len(word_tokenize(phrase)) == 1
    ]
    phrases = [
        phrase for phrase in non_affirmative_phrases
        if len(word_tokenize(phrase)) > 1
    ]

    return (
        any(word in word_tokenize(stmt) for word in words)
        or any(phrase in stmt for phrase in phrases)
    )


def is_affirmative(stmt: str) -> bool:
    """Returns whether `stmt` appears affirmative."""

    stmt = stmt.lower()
    words = [
        phrase for phrase in affirmative_phrases
        if len(word_tokenize(phrase)) == 1
    ]
    phrases = [
        phrase for phrase in affirmative_phrases
        if len(word_tokenize(phrase)) > 1
    ]

    return (
        not is_not_affirmative(stmt)
        and (
            any(word in word_tokenize(stmt) for word in words)
            or any(phrase in stmt for phrase in phrases)
        )
    )


def is_question(msg: str) -> bool:
    "Returns whether `msg` appears to be a question."

    return (
        any(msg.split()[0].lower() == word.lower() for word in question_words)
        or word_tokenize(msg)[-1] == '?'
    )


def is_ending_phrase(msg: str) -> bool:
    """
    Returns if `msg` is command to conversational sequence.
    Used to stop listing subreddits during onboarding.
    """

    msg = msg.lower()

    return (
        is_not_affirmative(msg)
        or any(
            end_word.lower() in word_tokenize(msg)
            for end_word in [
                phrase for phrase in ending_phrases
                if len(word_tokenize(phrase)) == 1
            ]
        )
        or any(
            end_phrase.lower() in msg
            for end_phrase in [
                phrase for phrase in ending_phrases
                if len(word_tokenize(phrase)) > 1
            ]
        )
    )


def is_exit_command(msg: str) -> bool:
    """Returns if `msg` is a command to end the program"""

    return any(msg.lower() == end_cmd.lower() for end_cmd in end_commands)


def am_statments(msg: str) -> 'list[str]':
    """Returns objects of 'I am...' statments."""

    return re.findall(r'(?:i|I)\s+am\s+([^\.,?!]+)', msg)[:]


def my_statments(msg: str) -> 'list[tuple[str, str]]':
    """
    Finds objects of 'My... is...' statments.
    Returns list of tuples of the form: (object, descriptor).
    """

    return re.findall(r'(?:m|M)y\s+([^\.,?!]+)\s+is\s+([^\.,?!]+)', msg)[:]


def parse_likes(dep_parse) -> 'list[str]':
    """
    Given a `spacy` dependency parse of a string,
    returns list of direct objects of statements resembling 'I like...'.
    """

    dobjs = []
    
    for token in dep_parse:
        if token.dep_.lower() == 'root':
            objs = [child for child in token.children if child.dep_ == 'dobj']

            for obj in objs:
                dobj = ' '.join(
                    [token.text for token in obj.lefts] +
                    [obj.text] +
                    [token.text for token in obj.rights]
                )

                if any(
                    child.dep_.lower() == 'nsubj' and child.text.lower() == 'i'
                    for child in token.children
                ) and any(
                    ss.name() == 'like.v.02' or ss.name() == 'like.v.03'
                    for ss in wordnet.synsets(token.text)
                ) and not any(
                    child.dep_.lower() == 'neg'
                    for child in token.children
                ):
                    dobjs.append(dobj)
    
    return dobjs


def parse_dislikes(dep_parse) -> 'list[str]':
    """
    Given a `spacy` dependency parse of a string,
    returns list of direct objects of statements resembling 'I don't like...'.
    """

    dobjs = []
    
    for token in dep_parse:
        if token.dep_.lower() == 'root':
            objs = [child for child in token.children if child.dep_ == 'dobj']

            for obj in objs:
                dobj = ' '.join(
                    [token.text for token in obj.lefts] +
                    [obj.text] +
                    [token.text for token in obj.rights]
                )

                if any(
                    child.dep_.lower() == 'nsubj' and child.text.lower() == 'i'
                    for child in token.children
                ) and any(
                    ss.name() == 'dislike.v.01' or ss.name() == 'hate.v.01'
                    for ss in wordnet.synsets(token.text)
                ) or (
                    any(
                        ss.name() == 'like.v.02' or ss.name() == 'like.v.03'
                        for ss in wordnet.synsets(token.text)
                    )
                    and any(
                        child.dep_.lower() == 'neg'
                        for child in token.children
                    )
                ):
                    dobjs.append(dobj)
    
    return dobjs


def sub_exists(sub_name: str, reddit: Reddit) -> bool:
    """
    Given a subreddit name and a `praw` handle to the Reddit API,
    returns whether the subreddit exists.
    """

    exists = True
    try:
        reddit.subreddits.search_by_name(sub_name, exact=True)
    except NotFound:
        exists = False
    return exists


def strip_sub_name(sub_name: str) -> str:
    """Returns string with 'r/' removed from the beginning."""

    if sub_name.startswith('r/'):
        return sub_name[2:].strip()
    return sub_name


def process_likes(
    like_objs: 'list[str]',
    sub_names: 'set[str]',
    likes: 'set[str]',
    dislikes: 'set[str]',
    reddit: Reddit,
    threads: 'list[Thread]',
    q: 'Queue[str]',
) -> str:
    """
    Returns a `str` of chatbot's response.
    Also may add `Thread`s to `threads`, the results of which will be put on `q`.
    """

    like_res = ''

    for obj in like_objs:
        obj = strip_sub_name(obj)

        if sub_exists(obj, reddit) and obj not in sub_names:
                t = Thread(target=scrape_subreddit, args=[obj, q])
                t.start()
                threads.append(t)
                like_res += f"So you like {obj}, huh? Let me make a note... "
        else:
            if obj in likes or obj in sub_names:
                like_res += f'Oh, I already knew you liked {obj}. '
            else:
                like_res += f'Ooh, now I know you like {obj}! '
        
        likes.add(obj)

        if obj in dislikes:
            dislikes.remove(obj)
    
    return like_res


def process_dislikes(
    dislike_objs: 'list[str]',
    all_sub_comments: 'dict[str, list[CommentForest]]',
    likes: 'set[str]',
    dislikes: 'set[str]',
    reddit: Reddit
) -> str:
    """
    Returns a `str` of chatbot's response.
    Also may update list of user's subreddit list.
    """

    dislike_res = ''

    for obj in dislike_objs:
        obj = strip_sub_name(obj)

        if sub_exists(obj, reddit) and obj in all_sub_comments.keys():
            dislike_res += f"I'll make a note that you don't like {obj} anymore. "
            all_sub_comments.pop(obj)
        else:
            if obj in dislikes:
                dislike_res += f"I think you've told me before. You must really hate {obj}. "
            else:
                dislike_res += f"Huh, I didn't know you disliked {obj}. "
        
        dislikes.add(obj)

        if obj in likes:
            likes.remove(obj)
    
    return dislike_res


def cosine_similarity(x: 'list[str]', y: 'list[str]') -> float:
    """
    Given two lists of tokenized strings,
    returns the cosine similarity of the two strings.
    """

    x_set = {w for w in x}
    y_set = {w for w in y}
    words = x_set.union(y_set)

    x_sum = 0
    y_sum = 0
    dot_prod = 0

    for w in words:
        in_x = 1 if w in x_set else 0
        in_y = 1 if w in y_set else 0
        dot_prod += in_x * in_y
        x_sum += in_x
        y_sum += in_y

    return dot_prod / float((x_sum**0.5)*(y_sum**0.5)+1)


def best_response_from_post(
    post_root: Comment,
    lem_stmts: 'list[list[str]]',
    q: 'Queue[tuple[str, float]]'
) -> None:
    """
    Given a `Comment` representing a post plus its replies
    and a list of recent chat messages in tokenized/lemmatized form,
    finds the best response according to the respective cosine similarities to comments in `cf`.
    Tuple of response and its cosine similarity is put on `q`
    """

    cl = [post_root] + post_root.replies.list()
    cos = -1
    res = ''

    if len(lem_stmts) >= 3:
        # chat history is 3 or more messages long

        for c in cl:
            sim_gp = cosine_similarity(lem_stmts[2], c.lemmatized)
            for cr in c.replies:
                sim_p = cosine_similarity(lem_stmts[1], cr.lemmatized)
                for crr in cr.replies:
                    if len(crr.replies) < 1:
                        continue

                    sim_c = cosine_similarity(lem_stmts[0], crr.lemmatized)
                    temp = sim_gp + sim_p * 2 + sim_c * 4

                    if temp > cos:
                        cos = temp
                        res = randchoice(crr.replies).body
    if len(lem_stmts) >= 2 and cos == -1:
        # chat history is 2 messages long

        for c in cl:
            sim_p = cosine_similarity(lem_stmts[1], c.lemmatized)
            for cr in c.replies:
                if len(cr.replies) < 1:
                    continue

                sim_c = cosine_similarity(lem_stmts[0], cr.lemmatized)
                temp = sim_p + sim_c * 2

                if temp > cos:
                    cos = temp
                    res = randchoice(cr.replies).body
    if len(lem_stmts) >= 1 and cos == -1:
        # chat history is 1 message long

        for c in cl:
            if len(c.replies) < 1:
                continue

            sim_c = cosine_similarity(lem_stmts[0], c.lemmatized)
            if sim_c > cos:
                cos = sim_c
                res = randchoice(c.replies).body

    q.put((res, cos))


def get_best_res_to_msg(
    all_sub_comments: 'dict[str, list[CommentForest]]',
    msgs: 'list[str]',
    lemmatizer: WordNetLemmatizer,
    sw: 'list[str]'
):
    """
    Given chat history `msgs`, searches through `all_sub_comments` to find the best response.
    """

    lem_msgs = []

    for msg in msgs:
        lem_msg = [
            lemmatizer.lemmatize(word)
            for word in word_tokenize(msg.lower())
            if word not in sw
        ]
        lem_msgs.append(lem_msg)

    q = Queue()
    threads = []

    for s in all_sub_comments.values():
        for p in s:
            t = Thread(target=best_response_from_post, args=[p, lem_msgs, q])
            t.start()
            threads.append(t)

    responses = []

    for _ in threads:
        responses.append(q.get())
    for t in threads:
        t.join()

    responses.sort(key=lambda x: x[1], reverse=True)
    best_cos = responses[0][1]
    filtered = [res[0] for res in responses if res[1] == best_cos]
    return randchoice(filtered)


def search_user_data(
    msg_tokenized: 'list[str]',
    mem_type: MemoryType,
    data: 'set[str]|dict[str, str]',
    q: Queue
):
    """Finds the most similar memory to `msg_tokenized` from within `data` and puts it on `q`."""

    cos = -1
    res = None

    for d in data:
        tokenized = None

        if mem_type == MemoryType.MY:
            tokenized = word_tokenize(d + ' ' + data[d])
        else:
            tokenized = word_tokenize(d)

        temp_cos = cosine_similarity(msg_tokenized, tokenized)

        if temp_cos > cos:
            cos = temp_cos

            if mem_type == MemoryType.MY:
                res = (d, data[d])
            else:
                res = d
    
    q.put((mem_type, res, cos))


def get_relevant_memory(
    msg: str,
    likes: 'set[str]',
    dislikes: 'set[str]',
    am_stmts: 'set[str]',
    my_stmts: 'dict[str, str]'
) -> 'tuple[str, str|tuple[str, str]]':
    """Returns a relevant memory from stored user data."""

    mem_type_to_data = {
        MemoryType.LIKE : likes,
        MemoryType.DISLIKE : dislikes,
        MemoryType.AM : am_stmts,
        MemoryType.MY : my_stmts
    }

    msg_tokenized = word_tokenize(msg)
    threads = []
    q = Queue()
    
    for mem_type in mem_type_to_data.keys():
        t = Thread(target=search_user_data, args=[
            msg_tokenized,
            mem_type,
            mem_type_to_data[mem_type],
            q
        ])
        t.start()
        threads.append(t)

    cos = -1
    res = None
    mem_type = None

    for _ in threads:
        temp_mem_type, temp_res, temp_cos = q.get()

        if temp_cos > cos:
            cos = temp_cos
            res = temp_res
            mem_type = temp_mem_type
    for t in threads:
        t.join()

    if res != None:
        if mem_type == MemoryType.LIKE:
            res = f'I remember you said you liked {res}'
        elif mem_type == MemoryType.DISLIKE:
            res = f"I remember you said you didn't like {res}"
        elif mem_type == MemoryType.AM:
            res = f'I remember you said you were {res}'
        elif mem_type == MemoryType.MY:
            res = f'I remember you said your {res[0]} was {res[1]}'

    return res


def make_canned_response(
    msg: str,
    likes: 'set[str]',
    dislikes: 'set[str]',
    am_stmts: 'set[str]',
    my_stmts: 'dict[str, str]'
) -> str:
    """Returns a canned response for a given msg."""

    if is_question(msg):
        return randchoice(canned_respones_to_question)
    else:
        res = get_relevant_memory(msg, likes, dislikes, am_stmts, my_stmts)

        if res == None or randreal() < 0.5:
            return randchoice(canned_respones_to_statement)
        
        return res


def make_response(
    all_sub_comments: 'dict[str, list[Comment]]',
    msgs: 'list[str]',
    name: str,
    likes: 'set[str]',
    dislikes: 'set[str]',
    am_stmts: 'set[str]',
    my_stmts: 'dict[str, str]',
    lemmatizer: WordNetLemmatizer,
    sw: 'list[str]',
    spacy_nlp,
    reddit: Reddit
) -> 'tuple[str, list[Thread], Queue]':
    """
    Returns a tuple:
        `str` of chatbot's response,\n
        `list` of any `Thread`s that were spawned,\n
        `Queue` to receive thread results
    """

    ret = None

    if len(msgs) >= 3 and msgs[0].lower() == msgs[2].lower():
        # user typed same message twice in a row
        ret = (randchoice(repetition_responses), [], None)
    else:
        new_am_stmts = am_statments(msgs[0])

        for stmt in new_am_stmts:
            am_stmts.add(stmt)

        new_my_stmts = my_statments(msgs[0])

        for obj, des in new_my_stmts:
            my_stmts[obj] = des

        dep_parse = spacy_nlp(msgs[0])
        like_objs = parse_likes(dep_parse)
        dislike_objs = parse_dislikes(dep_parse)
        like_res = ''
        threads = []
        q = Queue()

        if len(like_objs) > 0:
            like_res += process_likes(
                like_objs,
                all_sub_comments.keys(),
                likes,
                dislikes,
                reddit,
                threads,
                q
            )
        if len(dislike_objs) > 0:
            like_res += process_dislikes(
                dislike_objs,
                all_sub_comments,
                likes,
                dislikes,
                reddit
            )

        if like_res != '':
            ret = (like_res, threads, q)
        else:
            best_res = get_best_res_to_msg(all_sub_comments, msgs, lemmatizer, sw)

            if (
                cosine_similarity(word_tokenize(best_res), word_tokenize(msgs[0])) == 0
                and randreal() < 0.5
            ):
                ret = (
                    make_canned_response(
                        msgs[0],
                        likes,
                        dislikes,
                        am_stmts,
                        my_stmts
                    ),
                    [],
                    None
                )
            else:
                ret = (best_res, [], None)
    
    return ret


def dump_user_data(
    name: str,
    subs: 'set[str]',
    likes: 'set[str]',
    dislikes: 'set[str]',
    am_stmts: 'set[str]',
    my_stmts: 'dict[str, str]'
):
    """Writes user data to file."""

    name = name.lower()

    if not os.path.isdir(f'{data_dirname}/{user_data_dirname}/{name}'):
        os.makedirs(f'{data_dirname}/{user_data_dirname}/{name}')

    with open(f'{data_dirname}/{user_data_dirname}/{name}/{user_subs_filename}', 'wb') as f:
        pickle.dump(subs, f)
    with open(f'{data_dirname}/{user_data_dirname}/{name}/{user_likes_filename}', 'wb') as f:
        pickle.dump(likes, f)
    with open(f'{data_dirname}/{user_data_dirname}/{name}/{user_dislikes_filename}', 'wb') as f:
        pickle.dump(dislikes, f)
    with open(f'{data_dirname}/{user_data_dirname}/{name}/{user_am_stmts_filename}', 'wb') as f:
        pickle.dump(am_stmts, f)
    with open(f'{data_dirname}/{user_data_dirname}/{name}/{user_my_stmts_filename}', 'wb') as f:
        pickle.dump(my_stmts, f)


def chat_loop(
    all_sub_comments: 'dict[str, list[Comment]]',
    name: str,
    likes: 'set[str]',
    dislikes: 'set[str]',
    am_stmts: 'set[str]',
    my_stmts: 'dict[str, str]',
    sw: 'list[str]',
    reddit: Reddit
):
    """Main loop of program."""

    spacy_nlp = spacy.load("en_core_web_sm")
    profanity.load_censor_words()
    lemmatizer = WordNetLemmatizer()
    msgs = []

    print_chatbot_message("Alright I'm all ears now. What's on your mind?")

    while True:
        msg = input(prompt)

        # end chat
        if is_exit_command(msg):
            print_system_message('(Saving data...)')
            dump_user_data(
                name,
                set(all_sub_comments.keys()),
                likes,
                dislikes,
                am_stmts,
                my_stmts
            )
            print_chatbot_message(randchoice(goodbyes))
            return

        # reset context
        if msg.lower() == reset_phrase.lower():
            msgs = []
            continue

        if any(msg.lower() == end_cmd.lower() for end_cmd in end_commands):
            return

        msgs.insert(0, msg)
        res, threads, q = make_response(
            all_sub_comments,
            msgs,
            name,
            likes,
            dislikes,
            am_stmts,
            my_stmts,
            lemmatizer,
            sw,
            spacy_nlp,
            reddit
        )

        print_chatbot_message(profanity.censor(res))
        msgs.insert(0, res)

        while len(msgs) > 3:
            msgs.pop()

        # load new subreddits if any mentioned by user
        if len(threads) > 0:
            for _ in threads:
                sub_name = q.get()
                all_sub_comments_keys = all_sub_comments.keys()

                if sub_name not in all_sub_comments_keys:
                    if len(all_sub_comments) == sub_limit:
                        sub_remove = randchoice(list(all_sub_comments_keys))
                        all_sub_comments.pop(sub_remove)

                    with open(f'{data_dirname}/{subreddit_data_dirname}/{sub_name}', 'rb') as f:
                        all_sub_comments[sub_name] = pickle.load(f)
            for t in threads:
                t.join()


def main():
    print_system_message('(Loading...)')

    # preload wordnet to prevent threading errors
    wordnet.ensure_loaded()
    sw = stopwords.words('english')

    # make data directories if they don't exist
    if not os.path.isdir(f'{data_dirname}/{user_data_dirname}'):
        os.makedirs(f'{data_dirname}/{user_data_dirname}')
    if not os.path.isdir(f'{data_dirname}/{subreddit_data_dirname}'):
        os.makedirs(f'{data_dirname}/{subreddit_data_dirname}')

    greeting = randchoice(greetings)
    print_chatbot_message(f"{greeting}, what's your name?")

    is_returning_user = None
    
    # loop until user is confirmed/identified
    while True:
        name = input(prompt).strip()

        if is_exit_command(name):
            return

        if os.path.isdir(f'{data_dirname}/{user_data_dirname}/{name.lower()}'):
            # user data is found

            print_chatbot_message('Sounds familiar. Have we spoken before?')
            res = input(prompt).lower()

            # ask if user is indeed a returning user
            while True:
                if is_exit_command(res):
                    return

                if is_not_affirmative(res):
                    is_returning_user = False
                    break
                elif is_affirmative(res) and 'not' not in res:
                    is_returning_user = True
                    break
                else:
                    print_chatbot_message("Sorry, I didn't understand. Have we spoken before?")
                    res = input(prompt)

            if not is_returning_user:
                # user is new, must make new name

                print_chatbot_message(
                    "So you're new after all?\n" +
                    "Sorry, but you'll have to go by a different name. What do you want me to call you?"
                )
            else:
                # user is indeed a returning user

                return_greeting = randchoice(return_greetings)
                print_chatbot_message(return_greeting)
                break
        else:
            # user is new

            print_chatbot_message("Nice to meet you!")
            break

    reddit = make_reddit()
    sub_names = set()
    likes = set()
    dislikes = set()
    am_stmts = set()
    my_stmts = {}
    threads = []
    q = Queue()

    if is_returning_user:
        # user is not new, open existing files

        with open(f'{data_dirname}/{user_data_dirname}/{name.lower()}/{user_subs_filename}', 'rb') as f:
            sub_names = pickle.load(f)
        with open(f'{data_dirname}/{user_data_dirname}/{name.lower()}/{user_likes_filename}', 'rb') as f:
            likes = pickle.load(f)
        with open(f'{data_dirname}/{user_data_dirname}/{name.lower()}/{user_dislikes_filename}', 'rb') as f:
            dislikes = pickle.load(f)
        with open(f'{data_dirname}/{user_data_dirname}/{name.lower()}/{user_am_stmts_filename}', 'rb') as f:
            am_stmts = pickle.load(f)
        with open(f'{data_dirname}/{user_data_dirname}/{name.lower()}/{user_my_stmts_filename}', 'rb') as f:
            my_stmts = pickle.load(f)

        for sub_name in sub_names:
            t = Thread(target=scrape_subreddit, args=[sub_name, q])
            t.start()
            threads.append(t)

        print_chatbot_message("I remember your favorite subs now! Is there a new one you want to mention?")
    else:
        # user is new
        print_chatbot_message("Name a subreddit you're interested in")

    # loop to initialize subreddit interests
    while len(sub_names) < sub_limit:
        sub_name = strip_sub_name(input(prompt).lower().strip())

        if is_exit_command(sub_name):
            dump_user_data(name, sub_names, likes, dislikes, am_stmts, my_stmts)
            return

        # check if user is ending sequence
        if is_ending_phrase(sub_name):
            if len(sub_names) == 0:
                print_chatbot_message('Sorry, but you gotta name at least one...')
            else:
                print_chatbot_message("Got it! Let's move on...")
                break
        # check if user input is a subreddit
        elif len(sub_name.split()) == 1 and sub_exists(sub_name, reddit):
            if sub_name not in sub_names:
                sub_names.add(sub_name)
                t = Thread(target=scrape_subreddit, args=[sub_name, q])
                t.start()
                threads.append(t)

            response = randchoice(generic_responses)
            print_chatbot_message(f"{response}, anything else?")
        else:
            print_chatbot_message("Sorry, I didn't understand that. Try another?")
    
    print_system_message('(Processing...)')

    all_sub_comments = {}

    # scrape subreddits
    for _ in threads:
        sub_name = q.get()
        sub_comments = None

        with open(f'{data_dirname}/{subreddit_data_dirname}/{sub_name}', 'rb') as f:
            sub_comments = pickle.load(f)

        all_sub_comments[sub_name] = sub_comments
    for t in threads:
        t.join()

    # main loop
    chat_loop(
        all_sub_comments,
        name,
        likes,
        dislikes,
        am_stmts,
        my_stmts,
        sw,
        reddit
    )


if __name__ == '__main__':
    main()
