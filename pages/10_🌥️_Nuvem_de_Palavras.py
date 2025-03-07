
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from collections import Counter

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

import nltk
nltk.download('stopwords')  # Baixa o conjunto de stopwords do nltk

from nltk.corpus import stopwords

# Obter as stopwords do nltk
stopwords = set(stopwords.words('portuguese'))  # Usando stopwords em português, altere conforme necessário

# ------------------------------------------------------------------------------

# Função para remover as stopwords
def remove_stopwords(text):
    # Divide o texto em palavras e remove as stopwords
    words = text.split()
    words_filtered = [word for word in words if word.lower() not in stopwords]
    return " ".join(words_filtered)

# ------------------------------------------------------------------------------

def show_wordcloud(data):
    # Remover stopwords antes de gerar a nuvem de palavras
    cleaned_data = remove_stopwords(data)
    
    wordcloud = WordCloud(width=800, height=400, stopwords=stopwords, background_color="white").generate(cleaned_data)
    fig = plt.figure(figsize=[20,10])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot(fig)
    
# https://www.kaggle.com/datasets/aashita/masks?resource=download&select=user.png
def show_wordcloud2(data, imagem):
    # Remover stopwords antes de gerar a nuvem de palavras
    cleaned_data = remove_stopwords(data)

    mask = np.array(Image.open(imagem))
    wordcloud = WordCloud(width=800, height=400, stopwords=stopwords, background_color="white", mode="RGBA", max_words=1000, mask=mask).generate(cleaned_data)
    # create coloring from image
    image_colors = ImageColorGenerator(mask)
    fig = plt.figure(figsize=[20,10])
    plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    plt.show()
    st.pyplot(fig)

def get_top_non_stopwords(text, stop_words):
    # Converta as stopwords para minúsculas para garantir correspondência case-insensitive
    stop_words = set(word.lower() for word in stop_words)
    
    # Dividindo o texto em palavras
    new = text.str.split()
    new = new.values.tolist()
    corpus = [word for sublist in new for word in sublist]
    
    # Contando as ocorrências de cada palavra
    counter = Counter(corpus)
    most_common = counter.most_common()
    
    # Filtrando as palavras que não são stopwords, ignorando case
    filtered_words = [
        (word, count) for word, count in most_common if word.lower() not in stop_words
    ][:40]
    
    # Criando um DataFrame para as palavras e suas frequências
    df_top_words = pd.DataFrame(filtered_words, columns=['word', 'count'])
    
    return df_top_words

def get_top_ngrams(text, n=2):

    corpus = text.astype(str).tolist()
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:40]

# ------------------------------------------------------------------------------

st.markdown("# Nuvem de Palavras")

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

    # ------------------------------------------------------------------------------
    # Filtragem e geração de Word Cloud com base em filtros
    # ------------------------------------------------------------------------------
    st.markdown("# Controle de Filtragem para Nuvem de Palavras")

    # Seleção da coluna alvo
    target_column = st.selectbox('Selecione a coluna para filtrar:', df.columns.tolist())

    # Verificar se a coluna foi selecionada
    if target_column:
        unique_values = df[target_column].dropna().unique()
        
        # Seleção de valor único da coluna
        selected_value = st.selectbox(f'Selecione um valor único de "{target_column}":', unique_values)
        
        # Aplicação do filtro
        if selected_value:
            filtered_df = df[df[target_column] == selected_value]
            st.write(f"Dados filtrados para {target_column} = {selected_value}:")
            st.dataframe(filtered_df)
            
            # Combinação dos textos para a nuvem de palavras
            column = st.selectbox('Selecione a coluna para gerar a nuvem de palavras:', categorical)
            if column:

                df_top_original = get_top_non_stopwords(filtered_df[column], stopwords)   

                # Criando o gráfico com Altair
                bar_chart_original = alt.Chart(df_top_original).mark_bar().encode(
                    x=alt.X('count:Q', title='Frequência'),
                    y=alt.Y('word:N', sort='-x', title='Palavras'),
                    color=alt.Color('count:Q', scale=alt.Scale(scheme='blues'), title='Frequência')
                ).properties(
                    title="Top 40 Palavras Não Stopwords",
                )

                st.write("### Top 40 Palavras Não Stopwords")
                st.altair_chart(bar_chart_original, use_container_width=True)  

                n_value = st.selectbox(
                    "Selecione o valor para N-grams:",
                    options=range(2, 11),  # Selecionando valores de 2 a 10 para os n-grams
                    index=0  # Definindo o valor padrão para 2
                )

                # ------------------------------------------------------------------------------

                st.markdown(f"### Top {n_value}-grams")

                # ------------------------------------------------------------------------------                

                # Processando o texto original
                top_n_bigrams_original = get_top_ngrams(filtered_df[column], n_value)
                df_original = pd.DataFrame(top_n_bigrams_original, columns=['N-gram', 'Frequency'])

                # Encontrar o intervalo de frequência máximo
                max_freq = df_original['Frequency'].max()
                min_freq = df_original['Frequency'].min()

                if min_freq < max_freq:
                                     
                    # Slider para definir o intervalo de frequência
                    min_freq, max_freq = st.slider(
                        f"Defina o intervalo de frequência para as N-grams (para n={n_value}):",
                        min_value=min_freq, max_value=max_freq, value=(min_freq, max_freq)
                    )        

                    # Filtrando as N-grams com base na frequência
                    df_filtered = df_original[(df_original['Frequency'] >= min_freq) & (df_original['Frequency'] <= max_freq)]

                    # Gráfico para texto original
                    chart_original = alt.Chart(df_filtered).mark_bar().encode(
                        x=alt.X('Frequency:Q', title='Frequência'),
                        y=alt.Y('N-gram:N', sort='-x', title='N-gramas'),
                        tooltip=['N-gram', 'Frequency']
                        ).properties(
                            title=f"Top {n_value}-grams",
                        )

                    st.write(f"### Top {n_value}-grams")
                    st.altair_chart(chart_original)


                try:
                    filtered_text = " ".join(var for var in filtered_df[column])
                    st.write(f"Existem {len(filtered_text)} palavras na combinação de todos os valores do atributo {column}.")

                    # Gerar a nuvem de palavras
                    if st.checkbox('Gerar nuvem de palavras'):
                        show_wordcloud(filtered_text)

                    if st.checkbox('Gerar nuvem de palavras com imagem'):                    
                        import os
                        DIR = "./"
                        entries = [entry for entry in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, entry))]                        
                        imagem = st.selectbox('Selecione a imagem', mp.Filter(entries, ['png']), key = "selecione_imagem_nuvem_de_palavras_com_imagem")        
                        show_wordcloud2(filtered_text, imagem)

                except:
                    st.write("Não foi possível gerar o gráfico. Tente outro atributo.")
                    pass
            
else:
    st.error("O conjunto de dados selecionado não possui colunas categóricas.")
            
# ------------------------------------------------------------------------------
