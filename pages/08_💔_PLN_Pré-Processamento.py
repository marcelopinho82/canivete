
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
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import html
from string import punctuation
import difflib

# ------------------------------------------------------------------------------

def int_to_roman(num):
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syms = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
    ]
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num

roman_numerals = [int_to_roman(i) for i in range(1, 49)]

# ------------------------------------------------------------------------------

# Função para identificar diferenças e gerar texto de alteração
def get_differences(original, processed):
    differ = difflib.Differ()
    diff = list(differ.compare(original.split(), processed.split()))
    # Destacar apenas as diferenças
    #highlighted_diff = " ".join([word for word in diff if word.startswith("- ") or word.startswith("+ ")])
    highlighted_diff = " ".join([word for word in diff if word.startswith("- ")])
    return highlighted_diff

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

# Função para extrair top 10 stopwords
def get_top_stopwords(text_series, stop_words, text_type):
    new = text_series.str.split()
    new = new.values.tolist()
    corpus = [word for i in new for word in i]
    
    dic = defaultdict(int)
    for word in corpus:
        if word in stop_words:
            dic[word] += 1
    
    top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:10]
    df_top = pd.DataFrame(top, columns=['stopword', 'count'])
    df_top['type'] = text_type
    return df_top

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

# Função para identificar palavras do top 10 tamanhos
def get_top_10_word_lengths(df, column):
    # Quebrar os textos em listas de palavras
    df['Palavras'] = df[column].str.split()
    
    # Achando o tamanho de todas as palavras
    df['Tamanhos'] = df['Palavras'].apply(lambda x: [len(p) for p in x])
    
    # Unindo todos os tamanhos em uma única lista
    all_lengths = [length for sublist in df['Tamanhos'] for length in sublist]
                 
    # Top 10 tamanhos únicos
    top_10_lengths = sorted(set(all_lengths), reverse=True)[:10]    
    st.write("Top 10 - Maiores tamanhos de palavras:", set(top_10_lengths))    
    
    # Filtrar as linhas que contêm palavras do top 10 tamanhos
    df['Top_10'] = df['Palavras'].apply(
        lambda words: any(len(word) in top_10_lengths for word in words)
    )
    
    # Filtrando palavras que têm tamanhos no top 10
    df['Palavras_Top_10'] = df['Palavras'].apply(
        lambda words: [word for word in words if len(word) in top_10_lengths]
    )
    
    # Filtrando tamanhos que estão no top 10
    df['Tamanhos_Top_10'] = df['Tamanhos'].apply(
        lambda sizes: [size for size in sizes if size in top_10_lengths]
    )
    
    # Resultado: apenas as linhas relevantes
    result = df[df['Top_10']]    
    return result[['Palavras_Top_10', 'Tamanhos_Top_10']]
        
# ------------------------------------------------------------------------------

def plot_percentage_large_words_altair(df, column, threshold):
    """
    Plota o percentual de palavras maiores que um certo tamanho em relação ao total usando Altair.
    
    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os textos.
        column (str): Nome da coluna com textos.
        threshold (int): Tamanho mínimo das palavras a serem consideradas.
    """
    
    # Contar o total de palavras
    total_words = sum(df['Tamanhos'].apply(len))
    
    # Contar as palavras maiores que o tamanho definido
    large_words = sum(
        size > threshold for sizes in df['Tamanhos'] for size in sizes
    )
    
    # Calcular o percentual
    percentage_large_words = (large_words / total_words) * 100

    # Exibir os dados calculados
    st.write(f"Total de palavras: {total_words}")
    st.write(f"Palavras maiores que {threshold}: {large_words}")
    st.write(f"Percentual: {percentage_large_words:.2f}%")
    
    # Criar DataFrame para visualização
    data = pd.DataFrame({
        'Categoria': [f'Palavras > {threshold}', 'Outras palavras'],
        'Percentual': [percentage_large_words, 100 - percentage_large_words]
    })
    
    # Criar gráfico de pizza com Altair
    chart = alt.Chart(data).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field='Percentual', type='quantitative'),
        color=alt.Color(field='Categoria', type='nominal'),
        tooltip=['Categoria', 'Percentual']
    ).properties(        
        width=400,
        height=400

    )

    st.altair_chart(chart, use_container_width=True)    

# ------------------------------------------------------------------------------

def concatenate_columns(df, columns_to_concat, new_column_name, separator=' ', options=None, stop_words=None):

    """
    Concatena colunas especificadas em uma nova coluna e aplica tratamentos configuráveis.

    Parâmetros:
    - df: DataFrame para realizar a operação.
    - columns_to_concat: lista de colunas a serem concatenadas.
    - new_column_name: nome da nova coluna concatenada.
    - separator: string usada para separar os valores concatenados (padrão é um espaço).
    - options: lista de opções para tratamentos (ex.: ['remover_urls', 'minusculo']).
    - stop_words: lista de stopwords a ser usada (necessária se remover_stopwords=True).

    Retorna:
    - DataFrame atualizado com a nova coluna.
    """

    if options is None:
        options = []

    stemmer = PorterStemmer()

    def process_text(row):

        # Remove espaços antes e depois de cada valor
        trimmed_values = row.apply(lambda x: str(x).strip())
        
        # Concatena os valores das colunas especificadas - OK
        concatenated_text = separator.join(trimmed_values)

        # Decodifica entidades HTML (ex.: &amp; -> &) - OK
        if 'decodificar_entidades_html' in options: 
            concatenated_text = html.unescape(concatenated_text)
            concatenated_text = re.sub(r'<[^>]+>', ' ', concatenated_text)

        # Substitui 'nan' por espaços - OK
        if 'substituir_nan' in options:
            concatenated_text = re.sub(r'\bnan\b', ' ', concatenated_text, flags=re.IGNORECASE)

        # Remove URLs do texto - OK
        if 'remover_urls' in options:
            # https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python 
            concatenated_text = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', concatenated_text, flags=re.MULTILINE)
            concatenated_text = re.sub('http://\S+|https://\S+', '', concatenated_text, flags=re.IGNORECASE)

        # Remove e-mails - OK
        if 'remover_emails' in options:
            # Regex para identificar e-mails
            pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            concatenated_text = re.sub(pattern, ' ', concatenated_text, flags=re.IGNORECASE)

        # Substitui quebras de linha por espaços - OK
        if 'remover_quebras_de_linha' in options:            
            concatenated_text = re.sub(r'[\r\n]+', ' ', concatenated_text)

        # Remove porcentagens - OK
        if 'remover_porcentagens' in options:
            concatenated_text = re.sub(r'\b\d+%\b|\d{1,3}%', ' ', concatenated_text)

        # Remove datas - OK
        if 'remover_datas' in options:
            # Regex para identificar datas
            pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
            concatenated_text = re.sub(pattern, ' ', concatenated_text)
            pattern = r'\b\d{2}/\d{4}\b'
            concatenated_text = re.sub(pattern, ' ', concatenated_text)
            pattern = r'\b\d{1,2}\s+de\s+(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+\d{4}\b'
            concatenated_text = re.sub(pattern, ' ', concatenated_text, flags=re.IGNORECASE)
            pattern = r'\b\d{1,2}\s+de\s+(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\b'
            concatenated_text = re.sub(pattern, ' ', concatenated_text, flags=re.IGNORECASE)
            pattern = r'\b\d{1,2}\s+DE\s+(JANEIRO|FEVEREIRO|MARÇO|ABRIL|MAIO|JUNHO|JULHO|AGOSTO|SETEMBRO|OUTUBRO|NOVEMBRO|DEZEMBRO)\s+DE\s+\d{4}\b'
            concatenated_text = re.sub(pattern, ' ', concatenated_text, flags=re.IGNORECASE)
            pattern = r'\b\d{1,2}\s+DE\s+(JANEIRO|FEVEREIRO|MARÇO|ABRIL|MAIO|JUNHO|JULHO|AGOSTO|SETEMBRO|OUTUBRO|NOVEMBRO|DEZEMBRO)\b'
            concatenated_text = re.sub(pattern, ' ', concatenated_text, flags=re.IGNORECASE)
        
        # Remove algarismos romanos - OK
        if 'remover_algarismos_romanos' in options:
            concatenated_text = re.sub(r'\s+(?:' + '|'.join(roman_numerals) + r')\.\s+', ' ', concatenated_text, flags=re.IGNORECASE)
            concatenated_text = re.sub(r'\s+(?:' + '|'.join(roman_numerals) + r')\s+', ' ', concatenated_text, flags=re.IGNORECASE)

        # Remove dias da semana - OK
        if 'remover_dias_da_semana' in options:
            dias_da_semana = [
            'segunda-feira', 'terça-feira', 'quarta-feira', 'quinta-feira', 'sexta-feira', 'sábado', 'domingo',
            'SEGUNDA-FEIRA', 'TERÇA-FEIRA', 'QUARTA-FEIRA', 'QUINTA-FEIRA', 'SEXTA-FEIRA', 'SÁBADO', 'DOMINGO',
            ]
            pattern = r'\b(?:' + '|'.join(dias_da_semana) + r')\b'
            concatenated_text = re.sub(pattern, ' ', concatenated_text, flags=re.IGNORECASE)

        # Remove horas - OK
        if 'remover_horas' in options:                        
            pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(AM|PM|am|pm)?\b'
            concatenated_text = re.sub(pattern, ' ', concatenated_text)
            pattern = r'\d{1,2}[hH]\d{2}'
            concatenated_text = re.sub(pattern, ' ', concatenated_text)            
            pattern = r'\b\d{1,2}\s?horas\b|\b\d{1,2}\s?HORAS\b'
            concatenated_text = re.sub(pattern, ' ', concatenated_text)
            pattern = r'\b\d{1,2}:\d{2}\s?horas\b|\b\d{1,2}:\d{2}\s?HORAS\b'
            concatenated_text = re.sub(pattern, ' ', concatenated_text)
            pattern = r'\b\d{1,2}\s?minutos\b|\b\d{1,2}\s?MINUTOS\b'
            concatenated_text = re.sub(pattern, ' ', concatenated_text)

        # Remove números de telefone - OK
        if 'remover_telefones' in options:            
            pattern = r'\+\d{2,3}\s?\(\d{2,3}\)\s?\d{4,5}-\d{4,5}'
            concatenated_text = re.sub(pattern, ' ', concatenated_text)
            pattern = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
            concatenated_text = re.sub(pattern, ' ', concatenated_text)       

        # Remove ordinais - OK
        if 'remover_ordinais' in options:
            concatenated_text = re.sub(r'\b\d+(st|nd|rd|th|ST|ND|RD|TH|º|°|ª)\b', ' ', concatenated_text)
            concatenated_text = re.sub(r'\s\d+(a|o|A|O)\s', ' ', concatenated_text)
            concatenated_text = re.sub(r'\b(aula|parte|AULA|PARTE)[\s]?\d+\b', ' ', concatenated_text)

        # Remove caracteres especiais - OK
        if 'remover_caracteres_especiais' in options:               
            concatenated_text = re.sub('º', ' ', concatenated_text)
            concatenated_text = re.sub('°', ' ', concatenated_text)
            concatenated_text = re.sub('ª', ' ', concatenated_text)

        # Remove números soltos - OK
        if 'remover_numeros' in options:            
            concatenated_text = re.sub(r'\b[0-9]+\b\s*', ' ', concatenated_text)

        # Adiciona espaço antes de letras maiúsculas que seguem letras minúsculas - OK
        if 'adicionar_espaco_antes_de_caps' in options:
            concatenated_text = re.sub(r'\s([a-z])([A-Z])\s', r' \1 \2 ', concatenated_text)                      

        # Adiciona espaço ao redor de cada pontuação solta - OK
        if 'espaço_ao_redor_de_pontuacao' in options:            
            concatenated_text = re.sub(r'\:', ': ', concatenated_text)
            concatenated_text = re.sub(r'\,', ', ', concatenated_text)
            concatenated_text = re.sub(r'\;', '; ', concatenated_text)
            concatenated_text = re.sub(r'\.', '. ', concatenated_text)
            concatenated_text = re.sub(r'\!', '! ', concatenated_text)
            concatenated_text = re.sub(r'\?', '? ', concatenated_text)
            concatenated_text = re.sub(r'\'', "' ", concatenated_text)
            concatenated_text = re.sub(r'\"', '" ', concatenated_text)
            concatenated_text = re.sub(r'\(', ' (', concatenated_text)
            concatenated_text = re.sub(r'\)', ') ', concatenated_text)
            concatenated_text = re.sub(r'\[', ' [', concatenated_text)
            concatenated_text = re.sub(r'\]', '] ', concatenated_text)
            concatenated_text = re.sub(r'\{', ' {', concatenated_text)
            concatenated_text = re.sub(r'\}', '} ', concatenated_text)
            concatenated_text = re.sub(r'\—', '— ', concatenated_text)
            concatenated_text = re.sub(r'\.\.\.', '... ', concatenated_text)
            concatenated_text = re.sub(r'\=', ' = ', concatenated_text)
            concatenated_text = re.sub(r'\"', ' " ', concatenated_text)
            concatenated_text = re.sub(r'\”', ' " ', concatenated_text)
            concatenated_text = re.sub(r'\“', ' “ ', concatenated_text)

        # Remove todas as pontuações - OK
        if 'remover_pontuacao' in options:
            punctuations = punctuation.replace('-', '')  # Remove o hífen
            regex_pattern = f"[{re.escape(punctuations)}]"
            concatenated_text = re.sub(regex_pattern, ' ', concatenated_text)
            concatenated_text = re.sub(r'\“', ' ', concatenated_text)
            concatenated_text = re.sub(r'\“', ' ', concatenated_text)            
            concatenated_text = re.sub(' – ', ' ', concatenated_text)            
            concatenated_text = re.sub(' - ', ' ', concatenated_text)
            concatenated_text = re.sub('•', ' ', concatenated_text)
            concatenated_text = re.sub('●', ' ', concatenated_text)
            concatenated_text = re.sub('▶', ' ', concatenated_text)
            concatenated_text = re.sub('', ' ', concatenated_text)
            concatenated_text = re.sub('·', ' ', concatenated_text)
            concatenated_text = re.sub('™', ' ', concatenated_text)    
            concatenated_text = re.sub('﻿', ' ', concatenated_text)    
            concatenated_text = re.sub('¦', ' ', concatenated_text)    
            concatenated_text = re.sub('§', ' ', concatenated_text)    
            concatenated_text = re.sub('©', ' ', concatenated_text)    
            concatenated_text = re.sub('®', ' ', concatenated_text)    
            concatenated_text = re.sub('»', ' ', concatenated_text)    
            concatenated_text = re.sub('¿', ' ', concatenated_text)    
            concatenated_text = re.sub('˚', ' ', concatenated_text)    
            concatenated_text = re.sub('…', ' ', concatenated_text)    
            concatenated_text = re.sub('→', ' ', concatenated_text)    
            concatenated_text = re.sub('■', ' ', concatenated_text)    
            concatenated_text = re.sub('▪', ' ', concatenated_text)    
            concatenated_text = re.sub('▫', ' ', concatenated_text)    
            concatenated_text = re.sub('▬', ' ', concatenated_text)    
            concatenated_text = re.sub('►', ' ', concatenated_text)    
            concatenated_text = re.sub('○', ' ', concatenated_text)    
            concatenated_text = re.sub('◦', ' ', concatenated_text)    
            concatenated_text = re.sub('✓', ' ', concatenated_text)    
            concatenated_text = re.sub('✔', ' ', concatenated_text)    
            concatenated_text = re.sub('✦', ' ', concatenated_text)    
            concatenated_text = re.sub('➢', ' ', concatenated_text)    
            concatenated_text = re.sub('➤', ' ', concatenated_text)    
            concatenated_text = re.sub('￼', ' ', concatenated_text)    
            concatenated_text = re.sub('🎪', ' ', concatenated_text)    
            concatenated_text = re.sub('🤝', ' ', concatenated_text)    

        # Converte o texto para minúsculas - OK
        if 'minusculo' in options:
            concatenated_text = concatenated_text.lower()

        # Remove espaços duplicados - OK
        if 'remover_espacos_duplicados' in options:            
            concatenated_text = re.sub(r'\s+', ' ', concatenated_text).strip()

        # Filtra palavras que não são stopwords - OK       
        if 'remover_stopwords' in options and stop_words:
            concatenated_text = ' '.join([word for word in concatenated_text.split() if word not in stop_words])

        # Remove palavras duplicadas mantendo a ordem - OK
        if 'remover_palavras_duplicadas' in options:
            concatenated_text = ' '.join(dict.fromkeys(concatenated_text.split()))

        # Remove palavras grandes - OK
        if 'remover_palavras_grandes' in options:
            concatenated_text = ' '.join([word for word in concatenated_text.split() if len(word) <= max_word_length])

        # Aplica stemização nas palavras - OK
        if 'stemizar' in options:
            words = word_tokenize(concatenated_text)
            concatenated_text = ' '.join([stemmer.stem(word) for word in words])

        return concatenated_text

    # Aplica a função process_text para concatenar as colunas
    df[new_column_name] = df[columns_to_concat].apply(lambda row: process_text(row), axis=1)
    
    return df

def drop_columns(df, columns_to_drop):
    """
    Remove as colunas especificadas do DataFrame.

    Parâmetros:
    - df: DataFrame em que a operação será realizada.
    - columns_to_drop: lista das colunas a serem removidas.

    Retorna:
    - DataFrame atualizado sem as colunas removidas.
    """
    return df.drop(columns=columns_to_drop, errors='ignore')

def preprocess_dataframe(df, columns_to_concat, columns_to_drop, new_column_name,  separator=' ', options=None, stop_words=None):
    """
    Função de pré-processamento que concatena colunas e remove as colunas originais.

    Parâmetros:
    - df: DataFrame a ser processado.
    - columns_to_concat: lista das colunas a serem concatenadas.
    - new_column_name: nome da nova coluna concatenada.
    - columns_to_drop: lista das colunas a serem removidas.

    Retorna:
    - DataFrame atualizado com a nova coluna concatenada e as colunas removidas.
    """
    # Concatena as colunas especificadas em uma nova coluna
    if columns_to_concat:
        df = concatenate_columns(df, columns_to_concat, new_column_name, separator, options=options, stop_words=stop_words)

    # Remove as colunas especificadas
    if columns_to_drop:
        df = drop_columns(df, columns_to_drop)
    
    return df

# ------------------------------------------------------------------------------

st.markdown("# Processamento de Linguagem Natural (PLN) - Pré-Processamento")

# ------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------

import os
DIR = "./"
entries = [entry for entry in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, entry))]
option = st.selectbox('Qual o conjunto de dados gostaria de analisar?', mp.Filter(entries, ['csv']))
df_csv = pd.read_csv(option, encoding='utf-8')
df_csv.drop_duplicates(inplace=True)
df = mp.filter_dataframe(df_csv)
st.dataframe(df.head())

# ------------------------------------------------------------------------------

# Seleção de colunas para concatenar
columns_to_concat = st.multiselect("Selecione as colunas para concatenar:", options=df.select_dtypes(include=['object']).columns, default=df.select_dtypes(include=['object']).columns)
columns_to_drop = st.multiselect("Selecione as colunas para apagar:", options=df.select_dtypes(include=['object']).columns, default=[])

options = st.multiselect(
    "Selecione os tratamentos a serem aplicados:",
    options=[
                'adicionar_espaco_antes_de_caps',
                'decodificar_entidades_html',
                'espaço_ao_redor_de_pontuacao',
                'minusculo',
                'remover_algarismos_romanos',
                'remover_caracteres_especiais',
                'remover_datas',
                'remover_dias_da_semana',
                'remover_emails',
                'remover_espacos_duplicados',
                'remover_horas',
                'remover_numeros',
                'remover_ordinais',
                'remover_palavras_duplicadas',
                'remover_palavras_grandes',
                'remover_pontuacao',
                'remover_porcentagens',
                'remover_quebras_de_linha',
                'remover_stopwords',
                'remover_telefones',
                'remover_urls',
                'stemizar',
                'substituir_nan',
            ],
    default=[
                'adicionar_espaco_antes_de_caps',
                'decodificar_entidades_html',
                'espaço_ao_redor_de_pontuacao',
                'minusculo',
                'remover_algarismos_romanos',
                'remover_caracteres_especiais',
                'remover_datas',
                'remover_dias_da_semana',
                'remover_emails',
                'remover_espacos_duplicados',
                'remover_horas',
                'remover_numeros',
                'remover_ordinais',
                'remover_palavras_duplicadas',
                'remover_pontuacao',
                'remover_porcentagens',
                'remover_quebras_de_linha',
                'remover_stopwords',
                'remover_telefones',
                'remover_urls',
                'substituir_nan',
             ]
)

# Configuração de stopwords
if 'remover_stopwords' in options:
    stop_words_language = st.selectbox("Idioma das stopwords:", ["ambas", "portuguese", "english", "spanish"])
    
    if stop_words_language == "ambas":
        stop_words = set(stopwords.words("portuguese")).union(set(stopwords.words("english"))).union(set(stopwords.words("spanish")))
    else:
        stop_words = set(stopwords.words(stop_words_language))
else:
    stop_words = set(stopwords.words("portuguese")).union(set(stopwords.words("english"))).union(set(stopwords.words("spanish")))

if "remover_palavras_grandes" in options:
    max_word_length = st.slider("Defina o tamanho máximo da palavra:", min_value=10, max_value=50, value=20)

# ------------------------------------------------------------------------------

# Botão para executar o processamento
if st.button("Processar"):

    if columns_to_concat or columns_to_drop:

        # Processando o texto
        new_df = df.copy()
        new_df = preprocess_dataframe(new_df, columns_to_concat=columns_to_concat, columns_to_drop=columns_to_drop, new_column_name="Texto", separator=" ", options=[], stop_words=stop_words)
        new_df = preprocess_dataframe(new_df, columns_to_concat=columns_to_concat, columns_to_drop=columns_to_drop, new_column_name="Texto Processado", separator=" ", options=options, stop_words=stop_words)
        st.success("Texto processado com sucesso!")

        if options:
            new_df["Alterado"] = new_df["Texto"] != new_df["Texto Processado"]

            altered_df = new_df[new_df["Alterado"]]
            if len(options) == 1 and 'minusculas' not in options:
                altered_df["Diferenças"] = altered_df.apply(lambda row: get_differences(row["Texto"], row["Texto Processado"]), axis=1)

            if "Diferenças" in altered_df.columns:
                st.dataframe(altered_df[["Texto","Texto Processado","Diferenças"]].sort_values(by="Diferenças"), use_container_width=True)        
            else:
                st.dataframe(altered_df[["Texto","Texto Processado"]].sort_values(by="Texto Processado"), use_container_width=True)
        else:
            st.dataframe(new_df[["Texto","Texto Processado"]].sort_values(by="Texto Processado"), use_container_width=True)

        # ------------------------------------------------------------------------------            
        
        # Calcular o número de palavras antes e depois do tratamento usando .loc
        new_df.loc[:, 'word_count_original'] = new_df["Texto"].apply(lambda x: len(str(x).split()))
        new_df.loc[:, 'word_count_cleaned'] = new_df["Texto Processado"].apply(lambda x: len(str(x).split()))

        # ------------------------------------------------------------------------------
        # Gráficos
        # ------------------------------------------------------------------------------

        # Gráfico para textos originais
        hist_original = alt.Chart(new_df).mark_bar().encode(
            alt.X('word_count_original:Q', bin=alt.Bin(maxbins=30), title='Número de Palavras (Original)'),
            alt.Y('count()', title='Contagem'),
            color=alt.Column('word_count_original:N', title='Tamanho')
        ).properties(
            title="Distribuição do Número de Palavras (Original)",
        )

        # Gráfico para textos tratados
        hist_cleaned = alt.Chart(new_df).mark_bar().encode(
            alt.X('word_count_cleaned:Q', bin=alt.Bin(maxbins=30), title='Número de Palavras (Processado)'),
            alt.Y('count()', title='Contagem'),
            color=alt.Column('word_count_cleaned:N', title='Tamanho')
        ).properties(
            title="Distribuição do Número de Palavras (Processado)",
        )

        # Layout lado a lado com Streamlit
        st.write("### Comparação do Número de Palavras")
        st.altair_chart(hist_original.interactive(), use_container_width=True)
        st.altair_chart(hist_cleaned.interactive(), use_container_width=True)

        # ------------------------------------------------------------------------------

        # Dividindo o texto em palavras e calculando o tamanho de cada palavra (original)
        df_exploded_original = new_df['Texto'].str.split(expand=True).stack().reset_index(level=1, drop=True)
        df_word_lengths_original = df_exploded_original.str.len().reset_index(drop=True).to_frame(name='word_length')

        # Histograma para textos originais
        hist_word_length_original = alt.Chart(df_word_lengths_original).mark_bar().encode(
            alt.X('word_length:N',title='Tamanho das Palavras'),
            alt.Y('count()', title='Frequência'),
            color=alt.Column('word_length:N', title='Tamanho')
        ).properties(
            title="Distribuição do Tamanho das Palavras (Original)",
        )

        # Dividindo o texto em palavras e calculando o tamanho de cada palavra (processado)
        df_exploded_cleaned = new_df['Texto Processado'].str.split(expand=True).stack().reset_index(level=1, drop=True)
        df_word_lengths_cleaned = df_exploded_cleaned.str.len().reset_index(drop=True).to_frame(name='word_length')        

        # Histograma para textos processados
        hist_word_length_cleaned = alt.Chart(df_word_lengths_cleaned).mark_bar().encode(
            alt.X('word_length:N', title='Tamanho das Palavras'),
            alt.Y('count()', title='Frequência'),
            color=alt.Column('word_length:N', title='Tamanho')
        ).properties(
            title="Distribuição do Tamanho das Palavras (Processado)",
        )

        # Layout lado a lado com Streamlit
        st.write("### Comparação do Tamanho das Palavras")
        st.altair_chart(hist_word_length_original.interactive(), use_container_width=True)
        resultado = get_top_10_word_lengths(new_df, 'Texto')
        st.dataframe(resultado, use_container_width=True)

        st.altair_chart(hist_word_length_cleaned.interactive(), use_container_width=True)     
        resultado = get_top_10_word_lengths(new_df, 'Texto Processado')
        #st.dataframe(resultado, use_container_width=True)
        st.dataframe(new_df[new_df['Top_10']])
        
        # ------------------------------------------------------------------------------

        min_length = 20
        st.write(f"### Palavras com Tamanho Maior que {min_length}")
        new_df2 = new_df.copy()
        # Filtrando as palavras com tamanho maior que min_length
        new_df2["Filtered_Tamanhos"] = new_df2["Tamanhos"].apply(lambda sizes: [size for size in sizes if size > min_length])
        st.dataframe(new_df2[new_df2['Filtered_Tamanhos'].apply(lambda x: len(x) > 0)])

        # Contar o total de instâncias no DataFrame
        total_instancias = len(new_df2)

        # Contar o número de instâncias afetadas (onde a coluna 'Filtered_Tamanhos' está preenchida)
        instancias_afetadas = new_df2['Filtered_Tamanhos'].apply(lambda x: len(x) > 0).sum()
        instancias_nao_afetadas = total_instancias - instancias_afetadas

        # Calcular o percentual
        percentual_afetadas = (instancias_afetadas / total_instancias) * 100

        # Exibir o resultado
        st.write(f"Total de instâncias: {total_instancias}")
        st.write(f"Instâncias afetadas: {instancias_afetadas} ({percentual_afetadas:.2f}%)")        
        st.write(f"Instâncias não afetadas: {instancias_nao_afetadas}")    

        # Criar um DataFrame para o gráfico de pizza
        data_pizza = pd.DataFrame({
            'Categoria': ['Afetadas', 'Não Afetadas'],
            'Valor': [instancias_afetadas, instancias_nao_afetadas]
        })

        # Criar o gráfico de pizza usando Altair
        chart = alt.Chart(data_pizza).mark_arc().encode(
            theta='Valor:Q',
            color='Categoria:N',
            tooltip=['Categoria:N', 'Valor:Q']
        ).properties(
            width=400,
            height=400
        )

        # Exibir o gráfico no Streamlit
        st.altair_chart(chart, use_container_width=True)    

        plot_percentage_large_words_altair(new_df2, 'Texto Processado', threshold=min_length)

        # ------------------------------------------------------------------------------

        if "remover_stopwords" not in options:

            # Processando os textos originais e processados
            df_top_original = get_top_stopwords(new_df['Texto'], stop_words, 'Original')
            df_top_cleaned = get_top_stopwords(new_df['Texto Processado'], stop_words, 'Processado')

            # Gráfico de barras com Altair
            bar_chart_original = alt.Chart(df_top_original).mark_bar().encode(
                x=alt.X('count:Q', title='Frequência'),            
                y=alt.Y('stopword:N', sort='-x', title='Stopwords'),        
                color=alt.Color('type:N', title='Tipo de Texto'),
                column=alt.Column('type:N', title='Texto')
            ).properties(
                title="Top 10 Stopwords por Tipo de Texto (Original)",
            )

            # Gráfico de barras com Altair
            bar_chart_cleaned = alt.Chart(df_top_cleaned).mark_bar().encode(
                x=alt.X('count:Q', title='Frequência'),            
                y=alt.Y('stopword:N', sort='-x', title='Stopwords'),            
                color=alt.Color('type:N', title='Tipo de Texto'),
                column=alt.Column('type:N', title='Texto')
            ).properties(
                title="Top 10 Stopwords por Tipo de Texto (Processado)",
            )

            # Layout lado a lado com Streamlit
            st.write("### Comparação de Stopwords")
            st.altair_chart(bar_chart_original | bar_chart_cleaned, use_container_width=True)

        # ------------------------------------------------------------------------------

        df_top_original = get_top_non_stopwords(new_df['Texto'], stop_words)   
        df_top_cleaned = get_top_non_stopwords(new_df['Texto Processado'], stop_words)

        # Criando o gráfico com Altair
        bar_chart_original = alt.Chart(df_top_original).mark_bar().encode(
            x=alt.X('count:Q', title='Frequência'),
            y=alt.Y('word:N', sort='-x', title='Palavras'),
            color=alt.Color('count:Q', scale=alt.Scale(scheme='blues'), title='Frequência')
        ).properties(
            title="Top 40 Palavras Não Stopwords (Original)",
        )

        # Criando o gráfico com Altair
        bar_chart_cleaned = alt.Chart(df_top_cleaned).mark_bar().encode(
            x=alt.X('count:Q', title='Frequência'),
            y=alt.Y('word:N', sort='-x', title='Palavras'),
            color=alt.Color('count:Q', scale=alt.Scale(scheme='blues'), title='Frequência')
        ).properties(
            title="Top 40 Palavras Não Stopwords (Processado)",
        )
        
        # Layout lado a lado com Streamlit
        st.write("### Comparação de Top 40 Palavras Não Stopwords")
        st.altair_chart(bar_chart_original | bar_chart_cleaned, use_container_width=True)        

        # ------------------------------------------------------------------------------

        n_range=(2, 5)

        # Loop para gerar gráficos para cada valor de n
        for n in range(n_range[0], n_range[1] + 1):

            # Processando o texto original
            top_n_bigrams_original = get_top_ngrams(new_df['Texto'], n)
            df_original = pd.DataFrame(top_n_bigrams_original, columns=['N-gram', 'Frequency'])

            # Processando o texto processado
            top_n_bigrams_cleaned = get_top_ngrams(new_df['Texto Processado'], n)
            df_cleaned = pd.DataFrame(top_n_bigrams_cleaned, columns=['N-gram', 'Frequency'])    

            # Gráfico para texto original
            chart_original = alt.Chart(df_original).mark_bar().encode(
                x=alt.X('Frequency:Q', title='Frequência'),
                y=alt.Y('N-gram:N', sort='-x', title='N-gramas'),
                tooltip=['N-gram', 'Frequency']
            ).properties(
                title=f"Top {n}-grams (Original)",
            )
            
            # Gráfico para texto processado
            chart_cleaned = alt.Chart(df_cleaned).mark_bar().encode(
                x=alt.X('Frequency:Q', title='Frequência'),
                y=alt.Y('N-gram:N', sort='-x', title='N-gramas'),
                tooltip=['N-gram', 'Frequency']
            ).properties(
                title=f"Top {n}-grams (Processado)",
            )
            
            # Exibindo gráficos no Streamlit lado a lado
            st.write(f"### Comparação dos Top {n}-grams")
            st.altair_chart(chart_original | chart_cleaned, use_container_width=True)

        # ------------------------------------------------------------------------------
        
        df["Texto"] = new_df["Texto Processado"]
        downloadCSV(df, file_name=option.split('.')[0] + '_Processado.csv')

    else:
        st.warning("Selecione pelo menos uma coluna para concatenar ou excluir.")
