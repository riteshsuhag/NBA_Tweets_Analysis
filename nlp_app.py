
# Loading required packages - 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from PIL import Image
import string
import regex
import re
import emoji
import base64
# Forming word cloud - getting word frequencies -
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
# For topic modeling - 
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim
import gensim.corpora as corpora

# ~~~~~~~~~~~~~~~~~~~~~~~ REQUIRED FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~

# Function to form primary graphs for a selected team - 

def get_data(team):
    # return pickle.load(open(f'Data_Cleaning/{team}.p', 'rb'))
    return pd.read_pickle(f'Data_Cleaning/{team}.p')

def plot_graph(temp, col1, col2):
    
    df_team = temp
    
    # ~~~~~~~ Plotting word cloud
    
    col1.write("""
               ### Tweets word cloud - 
               """)
    col1.write(" ")
    col1.write(" ")
    # Getting each token from all the twets and finding the most common words -
    flat_words = [item for sublist in df_team['topic_modeling_text'] for item in sublist]
    word_freq = FreqDist(flat_words)
    
    # Creating a dictionary containing the word and respective count -
    #retrieve word and count from FreqDist tuples
    most_common_count = [x[1] for x in word_freq.most_common(50)]
    most_common_word = [x[0] for x in word_freq.most_common(50)]
    top_50_word = dict(zip(most_common_word, most_common_count))
    
    # Getting the first 30 words for word cloud -
    # plasma, magma, inferno, viridis, cividis
    # best inferno :P
    wordcloud = WordCloud(colormap = 'inferno', background_color = 'white')\
    .generate_from_frequencies(top_50_word)
    # Plotting the word cloud of 50 words using matplotlib -
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    col1.pyplot()


def plot_sentiment_graph(team, col):
    df_team = df_sentiment.loc[df_sentiment['team'] == team]

    df_team = df_team.groupby(['created']).agg({'sentiment_numeric' : 'sum'}).reset_index()
    
    col.write("""
               ### Sentiment trend of the team - 
               """)
    
    plt.plot(df_team['created'], df_team['sentiment_numeric'], color = 'navy')
    plt.title('Daily Tweets Sentiment Score', fontsize = 22)
    plt.xlabel('Date', fontsize = 20)
    plt.ylabel('Sentiment Score', fontsize = 20)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.axhspan(0, plt.ylim()[1], facecolor='lightgreen')
    plt.axhspan(plt.ylim()[0],0, facecolor='salmon')
    #plt.xticks('rotation=20')
    plt.tight_layout()
    col.pyplot()

def plot_hashtag_graphs(temp, col1, col2):
    col1.write("""
               ### Most Commonly used hashtags - 
               """)
    col1.write(" ")
    col1.write(" ")
    
    hash_temp = temp.loc[temp["hashtags"].apply( lambda hashtag: hashtag !=[]),['hashtags']]

    single_hash = pd.DataFrame(
        [hashtag for hashtags_list in hash_temp.hashtags
        for hashtag in hashtags_list],
        columns=['hashtag'])
    
    hash_count = single_hash.groupby('hashtag').size().reset_index(name='counts').sort_values('counts', ascending=False).reset_index(drop=True)
    hash_count = hash_count[0:25]
    
    top_50_hash = pd.Series(hash_count["counts"].values,index=hash_count["hashtag"]).to_dict()
    
    hashcloud = WordCloud(colormap = 'plasma', background_color = 'white')\
    .generate_from_frequencies(top_50_hash)
    # Plotting the word cloud of 50 words using matplotlib -
    plt.figure(figsize=(12, 8))
    plt.imshow(hashcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    col1.pyplot()
    
    # find popular hashtags - make into python set for efficiency
    hash_count_temp = set(hash_count['hashtag'])
    
    # make a new column with only the popular hashtags
    hash_temp['popular_hashtags'] = hash_temp["hashtags"].apply(
                lambda hashtag_list: [hashtag for hashtag in hashtag_list
                                      if hashtag in hash_count_temp])
    
    # drop rows without popular hashtag
    hash_temp = hash_temp.loc[hash_temp["popular_hashtags"].apply(lambda hashtag_list: hashtag_list !=[])]
    
    # make new dataframe
    hash_temp = hash_temp.loc[:, ['popular_hashtags']]
    
    for hashtag in hash_count_temp:
        # make columns to encode presence of hashtags
        hash_temp['{}'.format(hashtag)] = hash_temp["popular_hashtags"].apply(
            lambda hashtag_list: int(hashtag in hashtag_list))
    
    hash_temp = hash_temp.drop('popular_hashtags', axis=1)
    
    # calculate the correlation matrix
    correlations = hash_temp.corr()
    
    # plot the correlation matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(correlations,
        cmap='RdBu',
        vmin=-1,
        vmax=1,
        square = True,
        cbar_kws={'label':'correlation'})
    plt.title("Correlation between Top 25 Hashtags \n", fontsize = 25)
    col2.pyplot()

def get_topics(df, num_topics):
    
    df_temp = df.sample(frac = 0.2)
    
    text_dict = corpora.Dictionary(df_temp['topic_modeling_text'])
    
    tweets_bow = [text_dict.doc2bow(tweet) for tweet in df_temp['topic_modeling_text']]
    
    tweets_lda = LdaModel(tweets_bow,
                      num_topics = num_topics,
                      id2word = text_dict,
                      random_state = 1,
                      passes=5)
    
    words = [re.findall(r'"([^"]*)"',t[1]) for t in tweets_lda.print_topics()]
    topics = [' '.join(t[0:10]) for t in words]
    
    # Getting the coherence score - 
    st.write(' ')
    coherence_model = CoherenceModel(model=tweets_lda, texts=df_temp['topic_modeling_text'], 
                                   dictionary=text_dict, coherence='c_v')
    coherence_lda = coherence_model.get_coherence()
    
    return topics, coherence_lda

# ~~~~~~~~~~~~~~~~~~~~~~~ STREAMLIT PAGE FRONT END ~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~ NBA Tweets Analysis page

def nba_analysis_page():
    st.write("""
             # NBA Tweets Analysis!
             
             The purpose of this page is to give insights into tweets of different NBA teams. Data used is tweets from Jan 2020 to June 2020. First, a general overview of the teams is shown. The user can then select any particular team to get a detailed analysis.  
             The tweets can also be divided into different topics using LDA. PyLDAvis graph is recommended to explore different topics.
             
             """)
             
    st.write(' ')
    
    about_expander = st.expander("About")
    about_expander.write("""
                         This page is designed to give an overview of the tweets of NBA teams over a period of 6 months (Jan 2020 to June 2020).  
                         
                         We first see the distribution of tweets by teams over the period to understand which team had a more online presence. We also see the word distribution of the tweets.  
                         
                         Once we have the overall picture of the tweets, we can select a particular to understand the tweets of the team. In this section, we see the most used words in the tweets and most used hash-tags. This helps us in understanding which hashtags resonate most with the fans and how we can tweak those to get more traction. We also analyze the sentiment of the tweets over the period.  
                         
                         Lastly, we can also try to derive topics from the tweets. We can select the number of topics we want to divide the tweets into and graph the PyLDAvis graph to explore the topics.
                         """)
    st.write(" ")
    st.write("""
             ## Initial Overview of all the tweets
             
             Below we can see the number of tweets by each team and the word frequency distribution of all the tweets.
             """)
    
    # Dividing the screen into 2 columns -
    col1, col2 = st.columns((1,1.4))
    
    # displaying the distribution of tweets - 
    team_count = pd.DataFrame(df_sentiment['team'].value_counts().reset_index())
    team_count.columns = ['team', 'tweets_count']
    
    # Plotting the distribution - 
    col1.write("""
               #### Distribution of tweets -
               """)
    col1.write(' ')
    plt.bar(team_count['team'], team_count['tweets_count'])
    plt.xlabel('Team', fontsize = 12)
    plt.ylabel('Number of Tweets', fontsize = 12)
    plt.title('Distribution of Tweets', fontsize = 16)
    plt.xticks(rotation=20)
    plt.tight_layout()
    col1.pyplot()
    
    # Loading the image - 
    image = Image.open('Data_Cleaning/tweets_distribution.png')
    
    col2.write("""
               #### Word frequency distribution -
               """)
    col2.image(image, use_column_width=True)
    
    st.write(' ')
    st.write("""
             #### Now that we have a little bit of information about the tweets, let's explore how it changes according to different teams -
             """)
    st.write(' ')
    
    # Dividing the screen into 2 columns -
    col1, col2, col3 = st.columns((1,1,1))
    
    selected_team = col2.selectbox('Select a team -', ['~ Select ~', 'Chicago Bulls', 'Miami Heat', 'Boston Celtics', 'Toronto Raptors', 
                                                       'Houston Rockets'])
    
    st.write(' ')
   
    if selected_team != '~ Select ~':
        
        st.write("""
                 The 2 plots help in understanding the overall polarity of the tweets. The word cloud of most frequent words helps in getting an overview of the tweets. The sentiment trend of the tweets helps in understanding whether a team is majorly associated with positive public sentiment or not.
                 """)
        st.write(' ')
        # Dividing the screen into 2 parts for team specific graphs -
        word_col1, word_col2, word_col3 = st.columns((2,0.2,2))
        
        # Forming word cloud, sentiment score and most hashtags used cloud - 
        temp = get_data(selected_team)
        
        plot_graph(temp, word_col1, word_col3)
        
        plot_sentiment_graph(selected_team, word_col3)
        
        st.write(' ')
        st.write("""
                The next two plots help in analyzing the hashtags associated with the tweets. The word cloud gives an overview of the most used tweets. The correlation matrix can be used to understand which hashtags occur together. This can be used to come up with another hashtag and place it strategically to get the most popularity.
                """)
        st.write(' ')
        # Dividing the screen for the sentiments - 
        senti_col1, senti_col2, senti_col3 = st.columns((2,0.2,2))
        
        plot_hashtag_graphs(temp, senti_col1, senti_col3)
        
        # Getting to the topics - 
        st.write(' ')
        st.write("""
                 The marketing team can use this information to make an informed decision. If needed, we also have the option to investigate the topics of the tweets.  
                 
                 To get the topics, we will make use of topic modeling which refers to the task of identifying topics that best describes a set of documents. One popular topic modeling technique is known as Latent Dirichlet Allocation (LDA). LDA imagines a fixed set of topics. Each topic represents a set of words. And the goal of LDA is to map all the documents to the topics in a way, such that the words in each document are mostly captured by those imaginary topics.  

                 The number of topics we want to divide the tweets into can be selected using the select box below.  
                 
                 * **Note:** For visualization purposes, we only find topics on a subset of the data (20%) to ensure the program finishes in a timely manner. The same logic can be applied to the entire data.
                 """)
        
        topic_col1, topic_col2, topic_col3 = st.columns((1,3,1))
        
        num_of_topics = topic_col2.selectbox('Select the number of topics for Topic Modeling', ['~ Select Num of Topics ~', 2, 3, 4, 5 ,6 ,7, 8, 9, 10])
        
        if num_of_topics != '~ Select Num of Topics ~':
            
            topic_col2.info('Please wait while topic are being calculated. It may take 2-5 minutes.')
            topics, coherence_lda = get_topics(temp, num_of_topics)
            
            st.info('We see the words most associated with their respective topics below.')
            # Printing the results -
            st.write(f'We divide the tweets into {num_of_topics} topics - ')
            st.write('#### The topics are - \n')
            
            for id, t in enumerate(topics): 
                st.write(f"------ Topic {id} ------")
                st.write(t, end="\n")
            
            st.write('\n **Coherence Score: **', round(coherence_lda, 3))
            st.write("(Topic Coherence measures score a single topic by measuring the degree of semantic similarity between high scoring words in the topic. These measurements help distinguish between topics that are semantically interpretable topics and topics that are artifacts of statistical inference.)")
            # Add image and give download functionality!
            
            st.write("""
                     ### PyLDAvis
                     
                     Trying to understand the topics above is a bit difficult. This is made easy with the help of 'pyLDAvis' plots. 
                     
                     The 'pyLDAvis' is an interactive LDA visualization python package. The area of circle represents the importance of each topic over the entire corpus, the distance between the center of circles indicate the similarity between topics. For each topic, the histogram on the right side listed the top 30 most relevant terms. The bars represent the terms that are most useful in interpreting the topic currently selected.  
                     
                     A snapshot of the same can be seen below. Unfortunately streamlit doesn't currently support the display of such interactive plots, but the user can refer to the code below to plot the same on their local machine.
                     
                     """)
            
            st.write(' ')
                        
            with st.echo():
                try:
                    # Code for pyLDAvis -  
                    vis = pyLDAvis.gensim.prepare() # LDA_Model,  # Bag_of_Words, # dictionary
                    
                    # To save the file in the working directory -  
                    pyLDAvis.save_html(vis, 'LDA_Visualization.html')
                except:
                    pass
            
            st.write(' ')
            st.write('#### Sample Image - ')
            st.write(' ')
            # Setting the image - 
            image = Image.open('pyldavis.png')
            if image.mode != "RGB":
                image = image.convert('RGB')
            # Setting the image width -
            st.image(image, use_column_width=True)

# ~~~~~~~~~~~~~~~~~~~~~~~ User Area Page

def remove_urls(df, url_command):
    if url_command == 'Remove URLs':
        df['cleaned_text'] = df['cleaned_text'].str.replace(r"http\S+", "")
    return df

def hashtag_handler(df, hashtag_command):
    if hashtag_command == 'Extract Hashtags':
        def extract_hash_tags(s):
            return list(set(part[1:] for part in s.split() if part.startswith('#')))
        df['hashtags'] = df['cleaned_text'].apply(extract_hash_tags)
    return df

def username_remover(df, username_command):
    if username_command == 'Remove User Names':
        def remove_usernames(text):
            return re.sub(r'@[^\s]+', '', text)
        df['cleaned_text'] = df['cleaned_text'].apply(remove_usernames)
    return df

def handle_emoji(df, emoji_command):
    if emoji_command == 'Remove Emojis':
        def remove_emoji(text):
            return emoji.get_emoji_regexp().sub(u'', text)
        df['cleaned_text'] = df['cleaned_text'].apply(remove_emoji)
    elif emoji_command == 'Rephrase emojis':
        df['cleaned_text'] = df['cleaned_text'].apply(emoji.demojize)
    return df

def tokenize_func(df):
    def converting_tokens(text):
        tokens = nltk.word_tokenize(text)
        return tokens
    df['cleaned_text'] = df['cleaned_text'].apply(converting_tokens)
    return df

def lemmatize_func(df, lemmatizing_command):
    if lemmatizing_command == 'Yes':
        df = tokenize_func(df)
        # Initializing the Lemmatizer object -
        lemmatizer = WordNetLemmatizer()
        def lemmatize_row(tokens):
            lemmatized =[]
            for token in tokens:
                lemmatized.append(lemmatizer.lemmatize(token))
            return lemmatized
        df['cleaned_text'] = df['cleaned_text'].apply(lemmatize_row)
    return df

def stemming_func(df, stemming_command, lemmatizing_command):
    if stemming_command == 'Yes':
        if lemmatizing_command != 'Yes':
            df = tokenize_func(df)
        # Initializing the Lemmatizer object -
        ps = nltk.stem.PorterStemmer()
        def stemming_row(tokens):
            stemmed =[]
            for token in tokens:
                stemmed.append(ps.stem(token))
            return stemmed
        df['cleaned_text'] = df['cleaned_text'].apply(stemming_row)
    return df   


def get_stop_words(custom_words):
    stop_words = stopwords.words('english')
    stop_words += list(string.punctuation)
    stop_words += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    stop_words += ['’', '...', '…', '‘', "\'s", "\'\'", '“', '”', 'rt', 'wt...', 'w...', '``',
               'm...', 'say', '-', '_', '__','hi','\n','\n\n', '&amp;', ' ', '.', '-', '—',
               'got', "it's", 'it’s', "i'm", 'i’m', 'im', 'want', 'like', '$', '@', "n't", '']
    if custom_words is not None:
        stop_words += custom_words
    return stop_words


def remove_stop_words(df, stop_words, stop_word_command, lemmatizing_command, stemming_command):
    if stop_word_command == 'Remove Stop words':
        if lemmatizing_command != 'Yes' and stemming_command != 'Yes':
            df = tokenize_func(df)
        def row_stop_word(tokens):
            clean_text = [token.lower() for token in tokens if token.lower() not in stop_words]
            return clean_text
        df['cleaned_text'] = df['cleaned_text'].apply(row_stop_word)
    return df

def load_example_file():
    input_df = pd.read_csv('Data_Cleaning/Apr.csv')
    input_df = input_df.sample(1000, random_state = 42)
    return input_df

# Download csv file -
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() # Strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_text.csv">Download CSV File</a>'
    return href

# Joining the token back into a sentence -
def get_sentiment_score(df, sentiment_command):    
    if sentiment_command == "Yes":    
        def join_sentiment_text(row):
            row = " ".join(row)
            return row
        df['sentiment_text'] = df['cleaned_text'].apply(join_sentiment_text)
                    
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
        df['sentiment'] = df['sentiment_score'].apply(get_sentiment)
        df.drop(['sentiment_text', 'sentiment_score'], axis = 1, inplace = True)
    return df

def get_topics_users(df, num_topics):
    text_dict = corpora.Dictionary(df['cleaned_text'])
    bow = [text_dict.doc2bow(tweet) for tweet in df['cleaned_text']]
    lda_model = LdaModel(bow,
                      num_topics = num_topics,
                      id2word = text_dict,
                      random_state = 1,
                      passes=5)
    words = [re.findall(r'"([^"]*)"',t[1]) for t in lda_model.print_topics()]
    topics = [' '.join(t[0:10]) for t in words]
    # Getting the coherence score - 
    st.write(' ')
    coherence_model = CoherenceModel(model=lda_model, texts=df['cleaned_text'], 
                                   dictionary=text_dict, coherence='c_v')
    coherence_lda = coherence_model.get_coherence()
    return topics, coherence_lda

def clean_func(input_df, url_command, hashtag_command, username_command, emoji_command, lemmatizing_command, stemming_command, user_stop_words, stop_word_command, sentiment_command):
    input_df = remove_urls(input_df, url_command)
    input_df = hashtag_handler(input_df, hashtag_command)
    input_df = username_remover(input_df, username_command)
    input_df = handle_emoji(input_df, emoji_command)
    input_df = lemmatize_func(input_df, lemmatizing_command)
    input_df = stemming_func(input_df, stemming_command, lemmatizing_command)
    total_stop_words = get_stop_words(user_stop_words)
    input_df = remove_stop_words(input_df, total_stop_words, stop_word_command, lemmatizing_command, stemming_command)
    input_df = get_sentiment_score(input_df, sentiment_command)
    return input_df

def basic_nlp():
    
    # Intro to the section -
    st.write("""
             # Basic NLP Analytics!
             
             Welcome to the Basic NLP Analytics page, the main aim of this page is to make the process of NLP and text cleaning relatively easier for beginners.
             
             Here the user can input their own data for NLP pre-processing. The users will be able to view the original data and cleaned data to analyze the difference. The users can then form a word cloud and gain a better understanding of the data they are dealing with.
             
             """)
             
    st.write(' ')     
    st.write("** The user can also use the reference code for text-preprocessing available at the end for personal use/projects.**")
    st.write(' ')
    
    # About the section - 
    about_expander = st.expander('About')
    about_expander.write("""
                         Please read through the points to gain a better understanding of the functionality of the application. Here the user can upload their own data to play around with. Once the data has been uploaded -   
                         
                         1. The user will be able to see the uploaded file. Please note - the code will automatically drop any NA values from the data to ensure smooth operation.  
                         2. Once the data has been uploaded, the user will be asked to select the column on which they wish to perform NLP.   
                         3. The user can perform the following text cleaning tasks -   
                             * Remove stops words (Can also add custom stop-words to remove based on their data!)   
                             * Remove/rephrase emojis  
                             * Transform the text into tokens  
                             * Perform lemmatization  
                             * Perform stemming  
                         4. Once the data is ready, the user will be able to view a word cloud of the data using top 'N' words.   
                         5. The user can then perform sentimental analysis which is calculated using the Sentiment Intensity Analyzer.
                         6. The user can download the cleaned CSV for further analysis. 
                         """)
    about_expander.write(" ")
    about_expander.write(" ")
        
    st.write(' ')
    st.write(' ')
    
    st.sidebar.write('---')
    # Select Box to ask for input - 
    input_preference = st.sidebar.selectbox("Input file", ["~ Select ~", "Input from Computer", "Use Example file"])
    
    user_data = 2
    if input_preference == "Input from Computer":
        uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type = ['csv'])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            user_data = 1
    elif input_preference == "Use Example file":
        input_df = load_example_file()
        user_data = 0
        
    
    if input_preference != "~ Select ~" and user_data != 2:
        if input_df is not None:
            col1, col2, col3 = st.columns((1,4,1))
            col2.write('Below we can see 100 random rows from the data ')
            
            col2.dataframe(input_df.sample(100, random_state = 42))
            
            if user_data == 1:
                col_options = ['~ Select a column ~']
                df_columns = list(input_df.columns)
                col_options = col_options + df_columns
                text_column = col2.selectbox('Choose the column for text pre-processing', col_options, index = 0)
            elif user_data == 0:
                text_column = 'text'
                col2. write("""
                            ** We will be carrying out the text cleaning on the 'text' column.**
                            """)
        if text_column != "~ Select a column ~":
            
            input_df = input_df.dropna(subset = [text_column])
            if user_data == 1:
                input_df = pd.DataFrame({text_column : input_df[text_column]})
            input_df['cleaned_text'] = input_df[text_column]
            
            st.write(' ')
            st.write("""
                     
                     ### Text pre-processing:
                         
                    Now that the text column has been selected, the user can choose from various operations below.  
                    
                    **Note:**
                    * The code directly drops any NA values (only in the text column) in the data to avoid any unforeseen errors and the best user experience.
                    * Removing stop words will automatically remove any custom stop words added.  
                    * The program will tokenize the texts before removing stop words and performing lemmatizing or stemming.
                    * The user can download the text, cleaned text, hashtags (if any), and sentiment score columns as CSV (currently data up to 250MB can be downloaded. This helps to reduce the size as much as possible.)
                    
                    Once the data is cleaned we can see the new data frame at the end along with a word cloud containing the top 50 words and the sentiment distribution.  
                    
                    
                     """)
            
            op_col0, op_col1, op_col2 = st.columns((1,4,1))
            
            
            url_command = op_col1.selectbox('Remove any URLs present', ['~ Select ~', 'Remove URLs'], index = 0)
            hashtag_command = op_col1.selectbox("If it's twitter data, would you like to extract the hashtags?", ['~ Select ~', 'Extract Hashtags', "Not Twitter data"], index = 0)
            username_command = op_col1.selectbox("If it's twitter data, would you like to take out the user names?", ['~ Select ~', 'Remove User Names', "Not Twitter data"], index = 0)
            emoji_command = op_col1.selectbox('How would you like to handle emojis -', ['~ Select ~', 'Remove Emojis', "Rephrase emojis"], index = 0)
            
            lemmatizing_command = op_col1.selectbox('Would you like to lemmatize -', ['~ Select ~', 'Yes', "No"], index = 0)
            stemming_command = op_col1.selectbox('Would you like to do stemming -', ['~ Select ~', 'Yes', "No"], index = 0)
            
            stop_word_command = op_col1.selectbox('Remove stop-words', ['~ Select ~', 'Remove Stop words', "Keep stop words"], index = 0)
            
            if stop_word_command == "Remove Stop words":
                user_stop_words = op_col1.text_area("Enter custom words you want to exclude (separated using a comma) - ", height = 100)
                user_stop_words = user_stop_words.replace(" ", "").split(",")
            else:
                user_stop_words = []
             
            sentiment_command = op_col1.selectbox('Get Sentiment Score', ['~ Select ~', 'Yes', 'No'])
            
            # input_df = remove_urls(input_df, url_command)
            # input_df = hashtag_handler(input_df, hashtag_command)
            # input_df = username_remover(input_df, username_command)
            # input_df = handle_emoji(input_df, emoji_command)
            # input_df = lemmatize_func(input_df, lemmatizing_command)
            # input_df = stemming_func(input_df, stemming_command, lemmatizing_command)
            # total_stop_words = get_stop_words(user_stop_words)
            # input_df = remove_stop_words(input_df, total_stop_words, stop_word_command, lemmatizing_command, stemming_command)
            # input_df = get_sentiment_score(input_df, sentiment_command)
            
            cleaned_df = clean_func(input_df, url_command, hashtag_command, username_command,
                                  emoji_command, lemmatizing_command, stemming_command, user_stop_words, 
                                  stop_word_command, sentiment_command)
            st.write(' ')
            st.write("""
                     ** Below the user can analyze the changes in the 'cleaned_text' column of the data frame - **
                     """)
            st.write(' ')
            
            df_col1, df_col2, df_col3 = st.columns((1,4,1))
            df_col2.dataframe(cleaned_df.sample(100, random_state = 42))
            
            input_word_cloud1, input_word_cloud2, input_word_cloud3 = st.columns((1,3,1))
            try:
                input_word_cloud2.markdown(f"{filedownload(cleaned_df)}", unsafe_allow_html = True)
            except RuntimeError:
                input_word_cloud2.info("Unfortunately, streamlit only allows file size upto 50 MB to be downloaded. I request you to please use a smaller version of your file.")
            
            
            if (stop_word_command == "Remove Stop words" or lemmatizing_command == 'Yes' or stemming_command == 'Yes'):
                
                flat_words = [item for sublist in cleaned_df['cleaned_text'] for item in sublist]
                word_freq = FreqDist(flat_words)
                
                # Creating a dictionary containing the word and respective count -
                #retrieve word and count from FreqDist tuples
                most_common_count = [x[1] for x in word_freq.most_common(50)]
                most_common_word = [x[0] for x in word_freq.most_common(50)]
                top_50_word = dict(zip(most_common_word, most_common_count))
                
                input_word_cloud2. write(' ')
                
                input_word_cloud2. write(' ')
                input_word_cloud2.write("** Wordcloud of the text column - **")
                # Getting the first 30 words for word cloud -
                # plasma, magma, inferno, viridis, cividis
                # best inferno :P
                wordcloud = WordCloud(colormap = 'inferno', background_color = 'white')\
                .generate_from_frequencies(top_50_word)
                # Plotting the word cloud of 50 words using matplotlib -
                plt.figure(figsize=(12, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.tight_layout(pad=0)
                input_word_cloud2.pyplot()
                
                try:
                    if sentiment_command == "Yes":
                        sentiment_df = cleaned_df.groupby(["sentiment"]).agg({text_column : "count"}).reset_index()
                        plt.bar(sentiment_df["sentiment"], sentiment_df[text_column], color = ["red", "grey", "green"])
                        plt.xlabel("Sentiment", fontsize = 25)
                        plt.xticks(fontsize=25)
                        plt.yticks(fontsize=25)
                        plt.ylabel("Number of occurences", fontsize = 25)
                        plt.title("Sentiment Distribution", fontsize = 35)
                        input_word_cloud2.pyplot()
                except:
                    pass
                
                st.write(" ")
                st.write("""
                         ### Get Topics -
                         
                         Now that we have cleaned the data, it is ready for some topic modelling. The user can choose the number of topics, he/she wants to divide the data into. 
                         
                         The code uses Latent Dirichlet Allocation (LDA) to find the topics. It is an unsupervised learning technique to find abstract topics in the document. Further I also provide a coherence score which can be used to measure the division of topics.  
                         
                         The user can select the number of topics using the slider below.  
                         """)
                
                st.write(" ")
                
                topic_col1, topic_col2, topic_col3 = st.columns((1,3,1))
                
                user_num_topics = topic_col2.slider("Select the number of topics - ", 2, 10)
                
                topic_col2. write(" ")
                
                topicb_col1, topicb_col2, topicb_col3 = st.columns((2,1,2))
                
                if topicb_col2.button("Get Topics"):
                    ftopic_col1, ftopic_col2, ftopic_col3 = st.columns((1,3,1))
                    topics, coherence_lda = get_topics_users(cleaned_df, user_num_topics)
                    # Printing the results -
                    ftopic_col2.write(f'We divide the text into {user_num_topics} topics - ')
                    ftopic_col2.write('#### The topics are -')
                    
                    for id, t in enumerate(topics): 
                        ftopic_col2.write(f"------ Topic {id} ------")
                        ftopic_col2.write(t, end="\n")
                    
                    ftopic_col2.write(f'\n **Coherence Score: ** {round(coherence_lda, 3)}')
                    
                    st.write("""
                     ### PyLDAvis
                     
                     Trying to understand the topics above is a bit difficult. This is made easy with the help of 'PyLDAvis' plots. 
                     
                     Unfortunately streamlit doesn't currently support the display of such interactive plots, but the user can refer to the code below to plot the same on their local machine.
                     
                     """)
            
                    st.write(' ')
                                
                    with st.echo():
                        try:
                            # Code for pyLDAvis -  
                            vis = pyLDAvis.gensim.prepare() # LDA_Model,  # Bag_of_Words, # dictionary
                            
                            # To save the file in the working directory -  
                            pyLDAvis.save_html(vis, 'LDA_Visualization.html')
                        except:
                            pass
                        
            else:
                input_word_cloud2. write(' ')
                input_word_cloud2. write(' ')
                input_word_cloud2.info("The cleaned text isn't tokenized yet, so can't create the word cloud or get topics.")
            
            st.write("---")
            st.write("### Reference Code")
            # Giving the code for reference.
            with st.echo():
                
                # Reference Code for text pre-processing - 
                
                try:
                    
                    # ~~~~~~~~~~ Importing required packages -
                    import re
                    import emoji
                    import nltk
                    import string
                    
                    # df represents the data frame that contains the text data -
                    df = pd.Dataframe()
                    
                    # Replace text with the name of the column that contains the text data -
                    text_column = "text"
                    df["cleaned_text"] = df[text_column]
                    
                    # ~~~~~~~~~~ To remove URLs -
                    df['cleaned_text'] = df['cleaned_text'].str.replace(r"http\S+", "")
                    
                    # ~~~~~~~~~~ To extract Hashtags -
                    def extract_hash_tags(s):
                        return list(set(part[1:] for part in s.split() if part.startswith('#')))
                    df['hashtags'] = df['cleaned_text'].apply(extract_hash_tags)
                    
                    # ~~~~~~~~~~ To remove User-names -
                    def remove_usernames(text):
                        return re.sub(r'@[^\s]+', '', text)
                    df['cleaned_text'] = df['cleaned_text'].apply(remove_usernames)
                    
                    # ~~~~~~~~~~ To remove Emojis -
                    def remove_emoji(text):
                        return emoji.get_emoji_regexp().sub(u'', text)
                    df['cleaned_text'] = df['cleaned_text'].apply(remove_emoji)
                    
                    # ~~~~~~~~~~ To rephrase Emojis -
                    df['cleaned_text'] = df['cleaned_text'].apply(emoji.demojize)
                    
                    # ~~~~~~~~~~ To tokenize the texts - 
                    def converting_tokens(text):
                        tokens = nltk.word_tokenize(text)
                        return tokens
                    df['cleaned_text'] = df['cleaned_text'].apply(converting_tokens)
                    
                    # ~~~~~~~~~~ To Lemmatize text (remember to tokenize the text before lemmatizing)
                    lemmatizer = nltk.stem.WordNetLemmatizer()
                    def lemmatize_row(tokens):
                        lemmatized =[]
                        for token in tokens:
                            lemmatized.append(lemmatizer.lemmatize(token))
                        return lemmatized
                    df['cleaned_text'] = df['cleaned_text'].apply(lemmatize_row)
                    
                    # ~~~~~~~~~~ For performing stemming (remember to tokenize the text before stemming)
                    ps = nltk.stem.PorterStemmer()
                    def stemming_row(tokens):
                        stemmed =[]
                        for token in tokens:
                            stemmed.append(ps.stem(token))
                        return stemmed
                    df['cleaned_text'] = df['cleaned_text'].apply(stemming_row)
                    
                    # ~~~~~~~~~~ Removing Stop-words -
                    
                    # Add the custom words in the list below - 
                    custom_words = [""]
                    
                    def getting_stop_words(custom_words):
                        stop_words = nltk.corpus.stopwords.words('english')
                        stop_words += list(string.punctuation)
                        stop_words += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                        if custom_words is not None:
                            stop_words += custom_words
                        return stop_words
                    stop_words = getting_stop_words(custom_words)
                    def remove_stop_word(tokens):
                        clean_text = [token.lower() for token in tokens if token.lower() not in stop_words]
                        return clean_text
                    df['cleaned_text'] = df['cleaned_text'].apply(remove_stop_word)
                    
                except:
                    pass
            
        else:
            st.info("Choose a column for text Analysis")
    else:
        st.info('Awaiting input file')
    
# ~~~~~~~~~~~~~~~~~~~~~~~ Home Page



# ~~~~~~~~~~~~~~~~~~~~~~~ Common front end

# Setting the page layout -
st.set_page_config(layout = 'wide', page_title = "NLP App")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Setting the image - 
image = Image.open('nlp_image_1.png')
if image.mode != "RGB":
    image = image.convert('RGB')
# Setting the image width -
st.image(image, use_column_width=True)

# Loading the sentiment data because we require it immediately - 
# df_sentiment = pickle.load(open('Data_Cleaning/sentiment_data.p', 'rb'))
df_sentiment = pd.read_pickle('Data_Cleaning/sentiment_data.p')

# Sidebar navigation for users -
st.sidebar.header('Navigation tab -')
navigation_tab = st.sidebar.selectbox('Choose a tab', ('Home-Page', 'NBA Tweet Analysis', 'Basic NLP'))

# Displaying pages according to the selection -

# Default page -
if navigation_tab == 'Home-Page':
    # Introduction about the project -
    st.write("""
         # Natural Language Processing App!
         
         This app serves two purposes - 
         * **NBA Tweet Analysis:** This page shows the analysis done by the author on tweets of NBA teams, to assist the marketing team. They can examine the sentiment of the tweets and the overall trends/polarity of the tweets. This can assist them in making an informed decision about where to allocate their spend.
         * **User Area:** Users can use this area to upload their own data or a set of data and do simple text processing with the help of a simple UI.
         
         """)
    st.write(' ')
    st.info('Please scroll through different sections using the navigation tab on the left')
    
    
    st.write(' ')
    
# Analytics Page -
elif navigation_tab == 'NBA Tweet Analysis':
    nba_analysis_page()

# Customer loyalty page -
elif navigation_tab == 'Basic NLP':
    basic_nlp()
