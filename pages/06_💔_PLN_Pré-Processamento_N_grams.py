
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

import nltk
from sklearn.feature_extraction.text import CountVectorizer
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

def remove_ngrams_from_column(df, column, ngram_list):
    """
    Remove specified N-grams from the selected column in the dataframe.
    """
    for ngram in ngram_list:
        # Create a regex pattern to match the N-gram in the text
        pattern = r'\b' + r'\s+'.join(re.escape(word) for word in ngram.split()) + r'\b'
        # Remove the N-gram from the column
        df[column] = df[column].str.replace(pattern, '', regex=True)
    return df

# ------------------------------------------------------------------------------

st.markdown("# Processamento de Linguagem Natural (PLN) - Pré-Processamento - Análise de N-grams")

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

column = st.selectbox("Selecione a coluna para analisar as N-grams:", options=df.select_dtypes(include=['object']).columns)

# Mensagem de erro caso nenhuma coluna seja selecionada
if not column:
    st.error("Por favor, selecione ao menos uma coluna para analisar as N-grams.")
    st.stop()  # Interrompe a execução se nenhuma coluna for selecionada

if df[column].isnull().all():
    st.error("A coluna selecionada está vazia. Por favor, escolha outra.")
    st.stop()

n_start, n_end = st.slider(
    "Defina o intervalo para os N-grams:",
    min_value=2, max_value=10, value=(2, 5)
)
n_range = (n_start, n_end)

# ------------------------------------------------------------------------------

st.markdown(f"### Exibindo N-grams para o intervalo de: {n_start} a {n_end}")

# ------------------------------------------------------------------------------
# Loop para gerar gráficos para cada valor de n
selected_ngrams = []  # Inicializando a lista de N-grams selecionadas

for n in range(n_range[1], n_range[0] - 1, -1):

    try:

        # Processando o texto original
        top_n_bigrams_original = get_top_ngrams(df[column], n)
        df_original = pd.DataFrame(top_n_bigrams_original, columns=['N-gram', 'Frequency'])

        # Encontrar o intervalo de frequência máximo
        max_freq = df_original['Frequency'].max()
        min_freq = df_original['Frequency'].min()

        # Slider para definir o intervalo de frequência
        min_freq, max_freq = st.slider(
            f"Defina o intervalo de frequência para as N-grams (para n={n}):",
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
                title=f"Top {n}-grams (Frequência entre {min_freq} e {max_freq})",
            )
        
        st.write(f"### Top {n}-grams (Frequência entre {min_freq} e {max_freq})")
        st.altair_chart(chart_original)

        # ----------------------------------------------------------------------
        # Checkbox para remoção das N-grams
        
        st.markdown("#### Selecione as N-grams que deseja remover:")

        for ngram, _ in df_filtered.values:  # Agora usando df_filtered, não mais top_n_bigrams_original
            checkbox = st.checkbox(f"Remover: '{ngram}'", key=ngram, value=True)
            if checkbox:
                selected_ngrams.append(ngram)

    except Exception as e:
        st.warning(f"Não foi possível processar N-grams para n={n}: {str(e)}")

# ------------------------------------------------------------------------------ 

if st.button("Processar"):
    if selected_ngrams:
        # Remover as N-grams selecionadas do dataframe
        df_to_download = df.copy()
        df_to_download = remove_ngrams_from_column(df_to_download, column, selected_ngrams)        
        st.write(f"N-grams removidas: {', '.join(selected_ngrams)}")        

        # Exibindo o DataFrame após a remoção das N-grams
        st.markdown("### DataFrame após remoção das N-grams selecionadas")
        st.dataframe(df_to_download)

        # Exibindo o gráfico com as N-grams restantes
        st.markdown("### Gráfico com as N-grams restantes após o processamento")

        # Re-calculando as N-grams restantes após a remoção
        for n in range(n_range[1], n_range[0] - 1, -1):
            top_n_bigrams_post = get_top_ngrams(df_to_download[column], n)
            df_post = pd.DataFrame(top_n_bigrams_post, columns=['N-gram', 'Frequency'])
            
            # Gráfico para texto após o processamento
            chart_processed = alt.Chart(df_post).mark_bar().encode(
                x=alt.X('Frequency:Q', title='Frequência'),
                y=alt.Y('N-gram:N', sort='-x', title='N-gramas'),
                tooltip=['N-gram', 'Frequency']
            ).properties(
                title=f"Top {n}-grams Restantes Após Remoção)",
            )

            st.write(f"### Top {n}-grams Restantes Após Remoção")
            st.altair_chart(chart_processed)

        downloadCSV(df_to_download, file_name=option.split('.')[0] + '_Sem_Ngrams.csv')
