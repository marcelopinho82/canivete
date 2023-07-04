
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

import seaborn as sns
from collections import Counter
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ------------------------------------------------------------------------------

# Code Snippet for Top Stopwords Barchart
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
def plot_top_stopwords_barchart(text):
    stop=set(stopwords.words('english'))

    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]
    from collections import defaultdict
    dic=defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word]+=1

    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]
    x,y=zip(*top)
    fig = plt.figure(figsize=(20, 10))
    plt.bar(x,y)
    st.pyplot(fig)

# Code Snippet for Top Non-Stopwords Barchart
def plot_top_non_stopwords_barchart(text):
    stop=set(stopwords.words('english'))

    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    counter=Counter(corpus)
    most=counter.most_common()
    x, y=[], []
    for word,count in most[:40]:
        if (word not in stop):
            x.append(word)
            y.append(count)

    fig = plt.figure(figsize=(20, 10))
    sns.barplot(x=y,y=x)
    st.pyplot(fig)

# Code Snippet for Word Number Histogram
def plot_word_number_histogram(text):
    fig = plt.figure(figsize=(20, 10))
    text.str.split().\
        map(lambda x: len(x)).\
        hist()
    st.pyplot(fig)

# Code Snippet for Word Length Histogram
def plot_word_length_histogram(text):
    fig = plt.figure(figsize=(20, 10))
    text.str.split().\
        apply(lambda x : [len(i) for i in x]). \
        map(lambda x: np.mean(x)).\
        hist()
    st.pyplot(fig)

# Code Snippet for Top Stopwords Barchart
def plot_top_stopwords_barchart(text):
    stop=set(stopwords.words('english'))

    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]
    from collections import defaultdict
    dic=defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word]+=1

    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]
    x,y=zip(*top)
    fig = plt.figure(figsize=(20, 10))
    plt.bar(x,y)
    st.pyplot(fig)

# Code Snippet for Top N-grams Barchart
def plot_top_ngrams_barchart(text, n=2):
    stop=set(stopwords.words('english'))

    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:10]

    top_n_bigrams=_get_top_ngram(text,n)[:10]
    x,y=map(list,zip(*top_n_bigrams))
    fig = plt.figure(figsize=(20, 10))
    sns.barplot(x=y,y=x)
    st.pyplot(fig)

# Code Snippet for Polarity Histogram
from textblob import TextBlob
def plot_polarity_histogram(text):

    def _polarity(text):
        return TextBlob(text).sentiment.polarity

    polarity_score = text.apply(lambda x : _polarity(x))
    fig = plt.figure(figsize=(20, 10))
    polarity_score.hist()
    st.pyplot(fig)


# Code Snippet for Sentiment Barchart
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

def sentiment_vader(text, sid):
    ss = sid.polarity_scores(text)
    ss.pop('compound')
    return max(ss, key=ss.get)

def sentiment_textblob(text):
        x = TextBlob(text).sentiment.polarity

        if x<0:
            return 'neg'
        elif x==0:
            return 'neu'
        else:
            return 'pos'

def plot_sentiment_barchart(text, method='TextBlob'):
    if method == 'TextBlob':
        sentiment = text.map(lambda x: sentiment_textblob(x))
    elif method == 'Vader':
        nltk.download('vader_lexicon')
        sid = SentimentIntensityAnalyzer()
        sentiment = text.map(lambda x: sentiment_vader(x, sid=sid))
    else:
        raise ValueError('Textblob or Vader')

    fig = plt.figure(figsize=(20, 10))
    plt.bar(sentiment.value_counts().index,
            sentiment.value_counts())
    st.pyplot(fig)

def polarity(text):
    return TextBlob(text).sentiment.polarity

def sentiment(x):
    if x<0:
        return 'neg'
    elif x==0:
        return 'neu'
    else:
        return 'pos'

# Code Snippet for Named Entity Barchart
import spacy
from collections import Counter
import seaborn as sns

def plot_named_entity_barchart(text):
    nlp = spacy.load("en_core_web_sm")

    def _get_ner(text):
        doc=nlp(text)
        return [X.label_ for X in doc.ents]

    ent=text.apply(lambda x : _get_ner(x))
    ent=[x for sub in ent for x in sub]
    counter=Counter(ent)
    count=counter.most_common()

    x,y=map(list,zip(*count))
    fig = plt.figure(figsize=(20, 10))
    sns.barplot(x=y,y=x)
    st.pyplot(fig)


# Code Snippet for Most Common Named Entity Barchart
import spacy
from collections import  Counter
import seaborn as sns

def plot_most_common_named_entity_barchart(text, entity="PERSON"):
    nlp = spacy.load("en_core_web_sm")

    def _get_ner(text,ent):
        doc=nlp(text)
        return [X.text for X in doc.ents if X.label_ == ent]

    entity_filtered=text.apply(lambda x: _get_ner(x,entity))
    entity_filtered=[i for x in entity_filtered for i in x]

    counter=Counter(entity_filtered)
    x,y=map(list,zip(*counter.most_common(10)))

    fig = plt.figure(figsize=(20, 10))
    sns.barplot(y,x).set_title(entity)
    st.pyplot(fig)


# Code Snippet for Most Common Part of Speach Barchart
import nltk
from nltk.tokenize import word_tokenize
import seaborn as sns
from collections import Counter

def plot_most_common_part_of_speach_barchart(text, part_of_speach='NN'):
    nltk.download('averaged_perceptron_tagger')

    def _filter_pos(text):
        pos_type=[]
        pos=nltk.pos_tag(word_tokenize(text))
        for word,tag in pos:
            if tag==part_of_speach:
                pos_type.append(word)
        return pos_type


    words=text.apply(lambda x : _filter_pos(x))
    words=[x for l in words for x in l]
    counter=Counter(words)
    x,y=list(map(list,zip(*counter.most_common(7))))

    fig = plt.figure(figsize=(20, 10))
    sns.barplot(x=y,y=x).set_title(part_of_speach)
    st.pyplot(fig)


# Code Snippet for Parts of Speach Barchart
import nltk
from nltk.tokenize import word_tokenize
import seaborn as sns
from collections import Counter

def plot_parts_of_speach_barchart(text):
    nltk.download('averaged_perceptron_tagger')

    def _get_pos(text):
        pos=nltk.pos_tag(word_tokenize(text))
        pos=list(map(list,zip(*pos)))[1]
        return pos

    tags=text.apply(lambda x : _get_pos(x))
    tags=[x for l in tags for x in l]
    counter=Counter(tags)
    x,y=list(map(list,zip(*counter.most_common(7))))

    fig = plt.figure(figsize=(20, 10))
    sns.barplot(x=y,y=x)
    st.pyplot(fig)

# ------------------------------------------------------------------------------

st.markdown("# PLN")

# ------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------

import os
DIR = "./"
entries = [entry for entry in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, entry))]
option = st.selectbox('Qual o conjunto de dados gostaria de analisar?', mp.Filter(entries, ['csv']))
df_csv = pd.read_csv(option)
df_csv.drop_duplicates(inplace=True)
df = mp.filter_dataframe(df_csv)
st.dataframe(df.head())

# ------------------------------------------------------------------------------
# Gráficos
# ------------------------------------------------------------------------------

# https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools

# ------------------------------------------------------------------------------

categorical = df.select_dtypes(include=['object']).columns.tolist()

if categorical:

  categorical = [var for var in df.columns if df[var].dtype=='O']
  for x in categorical:
    df[x] = df[x].str.lower()

  # ----------------------------------------------------------------------------

  x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(include=['object']).columns, key = "x_pnl")

  st.write("Histograma do número de palavras")
  plot_word_number_histogram(df[x_column])

  st.write("Histograma do tamanho da palavra")
  plot_word_length_histogram(df[x_column])

  st.write("Gráfico de barras de stopwords")
  plot_top_stopwords_barchart(df[x_column])

  st.write("Gráfico de barras das principais palavras sem interrupção")
  plot_top_non_stopwords_barchart(df[x_column])

  st.write("Gráfico de barras de N-grams - 2")
  plot_top_ngrams_barchart(df[x_column],2)

  st.write("Gráfico de barras de N-grams - 3")
  plot_top_ngrams_barchart(df[x_column],3)

  st.write("Análise de sentimentos")
  plot_polarity_histogram(df[x_column])

  st.write("Análise de sentimento TextBlob")
  plot_sentiment_barchart(df[x_column], method='TextBlob')

  st.write("Análise de sentimentos de Vader")
  plot_sentiment_barchart(df[x_column], method='Vader')

  df['polarity_score']=df[x_column].\
    apply(lambda x : polarity(x))

  df['polarity']=df['polarity_score'].\
    map(lambda x: sentiment(x))

  st.write("Vamos dar uma olhada em algumas das manchetes positivas.")
  st.dataframe(df[df['polarity']=='pos'][x_column].head())

  st.write("Vamos dar uma olhada em algumas das manchetes negativas.")
  st.dataframe(df[df['polarity']=='neg'][x_column].head())

# ------------------------------------------------------------------------------
