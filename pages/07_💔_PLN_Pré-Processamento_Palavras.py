
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

import re
import altair as alt
from collections import Counter
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# ------------------------------------------------------------------------------

def downloadCSV(df, file_name):

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Download CSV Processado',
        data=csv,
        mime='text/csv',
        file_name=file_name
    )

# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------

def remove_words_from_column(df, column, words_list):
    """
    Remove specified words from the selected column in the dataframe.
    """
    for words in words_list:
        # Create a regex pattern to match the N-gram in the text
        pattern = r'\b' + r'\s+'.join(re.escape(word) for word in words.split()) + r'\b'
        # Remove the word from the column
        df[column] = df[column].str.replace(pattern, '', regex=True)
    return df

# ------------------------------------------------------------------------------

st.markdown("# Processamento de Linguagem Natural (PLN) - Pré-Processamento - Análise de Palavras Mais Frequentes em Textos")

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
# Seleção da coluna do dataset
# ------------------------------------------------------------------------------

st.markdown("### Seleção da Coluna")

column = st.selectbox("Selecione a coluna para analisar as palavras:", options=df.select_dtypes(include=['object']).columns)

# Mensagem de erro caso nenhuma coluna seja selecionada
if not column:
    st.error("Por favor, selecione uma coluna para realizar a análise de palavras mais frequentes.")
    st.stop()  # Interrompe a execução se nenhuma coluna for selecionada

if df[column].isnull().all():
    st.error("A coluna selecionada está vazia. Por favor, escolha outra.")
    st.stop()

df[column] = df[column].astype(str)

# ------------------------------------------------------------------------------

st.markdown(f"### Exibindo palavras mais frequentes")

# ------------------------------------------------------------------------------

selected_words = []  # Inicializando a lista de palavras selecionadas

try:

    # Processando o texto original
    
    stop_words = set(stopwords.words("portuguese")).union(set(stopwords.words("english")))
    df_top_original = get_top_non_stopwords(df[column], stop_words)
    st.dataframe(df_top_original)

    # Encontrar o intervalo de frequência máximo
    max_freq = df_top_original['count'].max()
    min_freq = df_top_original['count'].min()

    # Slider para definir o intervalo de frequência
    min_freq, max_freq = st.slider(
        f"Defina o intervalo de frequência desejado para as palavras mostradas no gráfico:",
        min_value=min_freq, max_value=max_freq, value=(min_freq, max_freq)
    )        

    # Filtrando as palavras com base na frequência
    df_filtered = df_top_original[(df_top_original['count'] >= min_freq) & (df_top_original['count'] <= max_freq)]
    
    # Criando o gráfico com Altair
    bar_chart_original = alt.Chart(df_filtered).mark_bar().encode(
        x=alt.X('count:Q', title='Frequência'),
        y=alt.Y('word:N', sort='-x', title='Palavras'),
        color=alt.Color('count:Q', scale=alt.Scale(scheme='blues'), title='Frequência')
        ).properties(
            title=f"Top 40 Palavras Não Stopwords (Frequência entre {min_freq} e {max_freq})",
        )
    
    st.write(f"### Palavras mais frequentes no intervalo de {min_freq} a {max_freq}:")
    st.altair_chart(bar_chart_original, use_container_width=True)

    # ----------------------------------------------------------------------
    # Checkbox para remoção das palavras
    
    st.markdown("#### Selecione as palavras que deseja remover:")

    for palavra, _ in df_filtered.values:  # Agora usando df_filtered, não mais top_n_bigrams_original
        checkbox = st.checkbox(f"Remover: '{palavra}'", key=palavra, value=True)
        if checkbox:
            selected_words.append(palavra)

except Exception as e:
    st.warning(f"Não foi possível processar as palavras: {str(e)}")

# ------------------------------------------------------------------------------ 

if st.button("Processar"):
    if selected_words:
        # Remover as palavras selecionadas do dataframe
        df_to_download = df.copy()
        df_to_download = remove_words_from_column(df_to_download, column, selected_words)        
        st.write(f"As seguintes palavras foram removidas: {', '.join(selected_words)}.")        

        # Exibindo o DataFrame após a remoção das palavras
        st.markdown("### Resultado após remoção das palavras selecionadas no DataFrame:")
        st.dataframe(df_to_download)

        # Exibindo o gráfico com as palavras restantes
        st.markdown("### Gráfico com as palavras restantes após o processamento")

        # Re-calculando as palavras restantes após a remoção       
        top_words_post = get_top_non_stopwords(df_to_download[column], stop_words)
        
        # Criando o gráfico com Altair
        chart_processed = alt.Chart(top_words_post).mark_bar().encode(
            x=alt.X('count:Q', title='Frequência'),
            y=alt.Y('word:N', sort='-x', title='Palavras'),
            color=alt.Color('count:Q', scale=alt.Scale(scheme='blues'), title='Frequência')
            ).properties(
                title=f"Top 40 Palavras Restantes Após Remoção)",
            )

        st.write(f"### Top 40 Palavras Restantes Após Remoção")
        st.altair_chart(chart_processed)

        downloadCSV(df_to_download, file_name=option.split('.')[0] + '_Sem_Palavras_Mais_Frequentes.csv')
