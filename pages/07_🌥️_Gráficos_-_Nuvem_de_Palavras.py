
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

# https://gist.github.com/sebleier/554280
import requests
stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
lista_stopwords = set(stopwords_list.decode().splitlines())

# https://www.datacamp.com/tutorial/wordcloud-python

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
stopwords.update(lista_stopwords)

# ------------------------------------------------------------------------------

def show_wordcloud(data):
    # https://www.datacamp.com/tutorial/wordcloud-python
    # https://stackoverflow.com/questions/28786534/increase-resolution-with-word-cloud-and-remove-empty-border
    # https://towardsdatascience.com/standing-out-from-the-cloud-how-to-shape-and-format-a-word-cloud-bf54beab3389
    wordcloud = WordCloud(width=800, height=400, stopwords=stopwords, background_color="white").generate(data)
    fig = plt.figure(figsize=[20,10])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot(fig)

def show_wordcloud2(data, imagem):
    mask = np.array(Image.open(imagem))
    wordcloud = WordCloud(width=800, height=400, stopwords=stopwords, background_color="white", mode="RGBA", max_words=1000, mask=mask).generate(data)
    # create coloring from image
    image_colors = ImageColorGenerator(mask)
    fig = plt.figure(figsize=[20,10])
    plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    plt.show()
    st.pyplot(fig)

# ------------------------------------------------------------------------------

st.markdown("# Gráficos - Nuvem de Palavras")

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

# ------------------------------------------------------------------------------
# Nuvem de palavras
# ------------------------------------------------------------------------------

categorical = df.select_dtypes(include=['object']).columns.tolist()

if categorical:

  if st.checkbox('NUVEM DE PALAVRAS SEM IMAGEM'):

    column = st.selectbox('Selecione a coluna', categorical, key = "selecione_coluna_nuvem_de_palavras_sem_imagem")
    x = column.strip()
    try:
      text = " ".join(var for var in df[x])
      st.write("Existem {} palavras na combinação de todos os valores do atributo {}.".format(len(text), x))
      show_wordcloud(text)
    except:
      st.write("Não foi possível gerar o gráfico. Tente outro atributo.")
      pass

  if st.checkbox('NUVEM DE PALAVRAS COM IMAGEM'):

    import os
    DIR = "./"
    entries = [entry for entry in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, entry))]
    column = st.selectbox('Selecione a coluna', categorical, key = "selecione_coluna_nuvem_de_palavras_com_imagem")
    imagem = st.selectbox('Selecione a imagem', mp.Filter(entries, ['png']), key = "selecione_imagem_nuvem_de_palavras_com_imagem")
    x = column.strip()
    try:
      text = " ".join(var for var in df[x])
      st.write("Existem {} palavras na combinação de todos os valores do atributo {}.".format(len(text), x))
      show_wordcloud2(text, imagem)
    except:
      st.write("Não foi possível gerar o gráfico. Tente outro atributo ou outra imagem.")
      pass

# ------------------------------------------------------------------------------
