
# ~~~~~~~~~~~~~~~~~~~~~~ IMPORTING REQUIRED PACKAGES ~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import re   # for regular expression operations
import emoji 
# For plotting - 
import matplotlib.pyplot as plt
import seaborn as sns
# For removing stop words, word tokenization, lemmatization -
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# For sentimental analysis -
from nltk.sentiment import SentimentIntensityAnalyzer
# To save the file - 
import pickle

# Reading the data -
file_list = ['Feb.csv', 'Mar.csv', 'Apr.csv', 'May.csv', 'Jun.csv']
df_total = pd.read_csv('Jan.csv')
for file in file_list:
    temp = pd.read_csv(file)
    df_total = df_total.append(temp, ignore_index=True)
    print(df_total.shape)

del(temp, file, file_list)

# Getting count of teams -
team_count = pd.DataFrame(df_total['team'].value_counts().reset_index())
team_count.columns = ['team', 'tweets_count']

# Selecting top 5 teams - 
team_count = team_count.iloc[0:5]
df = df_total[df_total['team'].isin(team_count['team'])].copy()
del(df_total)

# ~~~~~~~~~~~~~~~~~~~~~~ INITIAL TEXT CLEANING ~~~~~~~~~~~~~~~~~~~~~~

# Removing any URLs -
df['text'] = df['text'].str.replace(r"http\S+", "")

# Saving all hashtags -
def extract_hash_tags(s):
    return list(set(part[1:] for part in s.split() if part.startswith('#')))

df['hashtags'] = df['text'].apply(extract_hash_tags)

# Removing the usernames from the tweets - 
def remove_usernames(text):
    return re.sub(r'@[^\s]+', '', text)
df['text'] = df['text'].apply(remove_usernames)

# Removing emojis -
def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u'', text)

# We get the general stop words for the english language.
stop_words = stopwords.words('english')

# Including the punctuations in the stop words -
stop_words += list(string.punctuation)

# Including numbers to the list of stop words -
stop_words += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Removing further words, which we don't want - 
stop_words += ['’', '...', '…', '‘', "\'s", "\'\'", '“', '”', 'rt', 'wt...', 'w...', '``',
               'm...', 'say', '-', '_', '__','hi','\n','\n\n', '&amp;', ' ', '.', '-', '—',
               'got', "it's", 'it’s', "i'm", 'i’m', 'im', 'want', 'like', '$', '@', "n't", '']

# Adding list of team names in the stop words -
team_name = list(df['team'].unique())
for team in team_name:
   tokens = nltk.word_tokenize(team)
   stop_words += list(token.lower() for token in tokens)

# Initializing the Lemmatizer object -
lemmatizer = WordNetLemmatizer()

# Function to clean the data - 
def clean_text_data(text):
    tokens = nltk.word_tokenize(text)
    lemmatized =[]
    for token in tokens:
        lemmatized.append(lemmatizer.lemmatize(token))
    clean_text = [token.lower() for token in lemmatized if token.lower() not in stop_words]
    return(clean_text)

# ~~~~~~~~~~ Cleaning the  sentiment data -

# For sentiment analysis we rephrase the emojis -
df['sentiment_text'] = df['text'].apply(emoji.demojize)
# Applying the cleaning function -
df['sentiment_text'] = df['sentiment_text'].apply(clean_text_data)

# ~~~~~~~~~~ Cleaning the topic modelling data -

# For topic modelling we remove the emojis - 
df['topic_modeling_text'] = df['text'].apply(remove_emoji)
# Applying the cleaning function -
df['topic_modeling_text'] = df['topic_modeling_text'].apply(clean_text_data)

# ~~~~~~~~~~~~~~~~~~~~~~ GETTING SENTIMENTS OF ALL TWEETS ~~~~~~~~~~~~~~~~~~~~~~

# Joining the token back into a sentence -
def join_sentiment_text(row):
    row = " ".join(row)
    return row
df['sentiment_text'] = df['sentiment_text'].apply(join_sentiment_text)

# Instatiating the sentiment intensity analyzer -
sid = SentimentIntensityAnalyzer()

# Finding sentiment of each tweet - 
df['sentiment_score'] = df['sentiment_text'].apply(lambda review: sid.polarity_scores(review))

# Getting the sentiment from dictionary - 
def get_sentiment(score_dict):
    if score_dict['compound'] > 0.2:
        return 'Positive'
    elif score_dict['compound'] < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

# Storing the sentiment in a separate column
df['tweet_sentiment'] = df['sentiment_score'].apply(get_sentiment)
df.drop(['sentiment_score'], axis = 1, inplace = True)

# Setting numeric values to the sentiment score
df['sentiment_numeric'] = df['tweet_sentiment'].map({"Positive" : 1,
                                                     "Neutral" : 0,
                                                     "Negative" : -1})

# Getting the creation date of the tweets -
df['created'] = pd.to_datetime(df['created'])
df['created'] = df['created'].dt.date

# Removing doc_id - 
df.drop(['doc_id'], axis = 1, inplace = True)

# Dividing the data into 2 parts to make reading faster - 
# Dataframe for sentiments - 
df_sentiment = df[['created', 'team', 'tweet_sentiment', 'sentiment_numeric']]
# Saving the df -
pickle.dump(df_sentiment, open('sentiment_data.p', 'wb'))

# Dataframe to get the topics and form wordclouds -
df_topic_modeling = df.drop(['text', 'sentiment_text', 'tweet_sentiment', 'sentiment_numeric'], axis = 1)

# Making the word frequency distribution graph to save and use to reduce inital computation time - 
# Getting the number of words in respective tweets -
all_words = [word for tokens in df_topic_modeling['topic_modeling_text'] for word in tokens]
tweet_lengths = [len(tokens) for tokens in df_topic_modeling['topic_modeling_text']]
vocab = sorted(list(set(all_words)))

# Printing the answer -
print('{} words total, with a vocabulary size of {}'.format(len(all_words), len(vocab)))
print('Max tweet length is {}'.format(max(tweet_lengths)))

# Taking only a part of it - 
tweet_lengths = [num for num in tweet_lengths if num < 25]

# Plotting the distribution of the word count of tweets - 
fig1 = plt.figure(figsize = (15,8))
sns.countplot(tweet_lengths)
plt.title('Tweet Length Distribution', fontsize = 18)
plt.xlabel('Words per Tweet', fontsize = 14)
plt.ylabel('Number of Tweets', fontsize = 14)
plt.savefig('tweets_distribution.png')
plt.show()

# Since we have the word count, we can break the tweets data into teams so that we only have to load a small part of data at a time.
# We can further use cache function of streamlit to ensure once a data for a particular team is loaded, we don't have to lead it again.
teams = list(df_sentiment['team'].unique())

for selected_team in teams:
    temp = df_topic_modeling.loc[df_topic_modeling['team'] == selected_team]
    pickle.dump(temp, open(f'{selected_team}.p', 'wb'))

del(temp, teams, selected_team)


