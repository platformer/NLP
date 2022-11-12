## Installing

### Google Colab

Steps:
1.  Go to this [Google Colab notebook](https://colab.research.google.com/drive/1JoAQF8m72Gm78dE8HQVkZePeKSYPPXTy?usp=sharing).
2.  Click 'File'
3.  Click 'Save a copy in Drive`
4.  Run the copied notebook

Note: on Google Colab, you may see errors regarding `asyncio` while the program executes.
These errors do not show up when running the program locally.
The errors are a result of one of the chatbot's dependencies making multiple HTTP requests
and the program itself being heavily multithreaded. These two factors don't seem to play nicely
with Google Colab's environment. Sorry!

### Local Install

Your Python version must be 3.8 or higher and must be 64 bit.  This program also requires a few python libraries to run.

Steps to install:
1.  `pip install -U pip setuptools wheel`
2.  `pip install nltk`
3.  `pip install praw`
4.  `pip install better_profanity`
5.  `pip install -U spacy`
6.  `python -m spacy download en_core_web_sm`
7.  Open your python repl with `python` and type the following:
    1.  `import nltk`
    2.  `nltk.download('all')`
    3.  exit the repl with `quit()`

## Removal

### Google Colab

Delete the notebook from your Google Drive.

### Local Install

Use the following commands to remove the respective packages:

*   `pip uninstall nltk`
*   `pip uninstall praw`
*   `pip uninstall better_profanity`
*   `pip uninstall en_core_web_sm`
*   `pip uninstall spacy`

In both cases, if you simply wish to remove the chatbot's data, you can delete the folder called `chatbot_data`.

## Usage

After the chatbot asks for your name, it'll ask you to name a few subreddits you're interested in.
The chatbot will scrape the subreddits and use them as material for its responses.
The chatbot tends to make more relevant responses if the subreddits you choose are smaller and more focused.
If you don't have any subreddits you're interested in, here are a few generic options:

*   r/books
*   r/gaming
*   r/movies
*   r/music
*   r/science
*   r/technology
*   r/worldnews

The process of scraping the subreddits may take a couple minutes before the conversation can commence, especially
if the subreddits are large. The subreddit data is cached so that subsequent runs of the program won't take as long.

The chatbot can also recognize 'I like...' or 'I don't like...' statements.
These can be used to modify your subreddit preferences during the conversation.
Note that the program will only remember a maximum of 15 subreddits at a time,
so if you try to go over that limit, the chatbot will drop a random subreddit to compensate.

The chatbot can recognize 'I am...' and 'My... is...' statements. These will be added to the chatbot's memories about you.

The chatbot will remember the most recent messages in order to maintain context
and (hopefully) give responses that are relevant to the topic of discussion.
If you want to reset the chatbot's memory of the recent chat history, enter `reset`.

To quit the program at any point, enter `quit` or `exit`. This will also make the program save your user data to file.
