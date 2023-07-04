
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd

# ------------------------------------------------------------------------------

def verResultado(df):
  st.write('Resultado')
  st.dataframe(df)
  st.metric("Valores Ausentes", df.isna().sum().sum(), delta=None, delta_color="normal", help=None)

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

st.markdown("# Pré-Processamento")

# ------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------

import os
DIR = "./"
entries = [entry for entry in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, entry))]
option = st.selectbox('Qual o conjunto de dados gostaria de analisar?', mp.Filter(entries, ['csv']))
df_csv = pd.read_csv(option)
df_csv.drop_duplicates(inplace=True)

# ------------------------------------------------------------------------------

df = mp.filter_dataframe(df_csv)
st.dataframe(df)

# ------------------------------------------------------------------------------

st.metric("Valores Ausentes", df.isna().sum().sum(), delta=None, delta_color="normal", help=None)

# ------------------------------------------------------------------------------

categorical = df.select_dtypes(include=['object']).columns.tolist()

if categorical:

  if st.checkbox('VALORES ÚNICOS'):

    cat_cols = df.select_dtypes(include=object).columns.tolist()
    df_pre_proc = pd.DataFrame(df[cat_cols].melt(var_name='column', value_name='value').value_counts()).rename(columns={0: 'counts'}).sort_values(by=['column', 'counts'])
    df_pre_proc.reset_index(inplace=True)
    df_pre_proc.to_csv(option.split('.')[0] + '_Unique.csv', encoding='utf-8')
    df_pre_proc = pd.read_csv(option.split('.')[0] + '_Unique.csv')
    os.remove(option.split('.')[0] + '_Unique.csv')
    df_pre_proc.columns = ['N','Atributo','Valor','Contagem']
    df_pre_proc = df_pre_proc.drop('N', axis=1)
    st.dataframe(df_pre_proc)

    # https://docs.streamlit.io/knowledge-base/using-streamlit/how-download-file-streamlit

    csv = df_pre_proc.to_csv(index=False).encode('utf-8')
    st.download_button(label='Download CSV', data=csv, mime='text/csv', file_name=option.split('.')[0] + '_Unique.csv')

# ------------------------------------------------------------------------------

# https://towardsdatascience.com/data-preprocessing-with-python-pandas-part-1-missing-data-45e76b781993
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
# https://stackoverflow.com/questions/42789324/how-to-pandas-fillna-with-mode-of-column
# https://www.plus2net.com/python/pandas-dataframe-dropna-thresh.php

numerical = df.select_dtypes(exclude=['object']).columns.tolist()
categorical = df.select_dtypes(include=['object']).columns.tolist()

# ------------------------------------------------------------------------------

if ((df.isna().sum().sum()) > 0):

  st.write("O QUE GOSTARIA DE FAZER?")

  if st.checkbox('DESCARTAR AS LINHAS QUE CONTÊM VALORES AUSENTES'):
    df.dropna(axis=0, inplace=True)
    verResultado(df)

  if st.checkbox('DESCARTAR AS COLUNAS QUE CONTÊM VALORES AUSENTES'):
    df.dropna(axis=1, inplace=True)
    verResultado(df)

  if st.checkbox('MANTER APENAS AS LINHAS COM 2 OU MAIS DADOS VÁLIDOS'):
    df.dropna(how='any', axis=0, thresh=2, inplace=True)
    verResultado(df)

  if st.checkbox('MANTER APENAS AS LINHAS COM 3 OU MAIS DADOS VÁLIDOS'):
    df.dropna(how='any', axis=0, thresh=3, inplace=True)
    verResultado(df)

  if st.checkbox('MANTER APENAS AS LINHAS EM QUE 70% OU MAIS DADOS VÁLIDOS ESTÃO DISPONÍVEIS'):
    df.dropna(how='any', axis=0, thresh=df.shape[1]*0.7, inplace=True)
    verResultado(df)

  if st.checkbox('MANTER APENAS COLUNAS ONDE 80% OU MAIS DADOS VÁLIDOS ESTÃO DISPONÍVEIS'):
    df.dropna(how='any', axis=1, thresh=df.shape[0]*0.8, inplace=True)
    verResultado(df)

  if st.checkbox('MANTER APENAS AS COLUNAS ONDE HOUVER PELO MENOS 80% DE VALORES NÃO NULOS'):
    df.dropna(axis=1, thresh=0.8*len(df), inplace=True)
    verResultado(df)

  if st.checkbox('MANTENHA APENAS COLUNAS ONDE 11 OU MAIS DADOS VÁLIDOS ESTÃO DISPONÍVEIS'):
    df.dropna(how='any', axis=1, thresh=11, inplace=True)
    verResultado(df)

  if st.checkbox('PREENCHER OS VALORES NAN DAS COLUNAS NUMÉRICAS COM O VALOR MÉDIO'):
    for col in df.select_dtypes(exclude=['object']).columns.tolist():
      num = df[col].mean()
      df[col].fillna(num, inplace=True)
    verResultado(df)

  if st.checkbox('PREENCHER OS VALORES NAN DAS COLUNAS NUMÉRICAS COM A MEDIANA'):
    for col in df.select_dtypes(exclude=['object']).columns.tolist():
      num = df[col].median()
      df[col].fillna(num, inplace=True)
    verResultado(df)

  if st.checkbox('SUBSTITUIR TODOS OS VALORES AUSENTES PELO VALOR MAIS FREQUENTE'):
    for col in df.select_dtypes(include=['object']).columns.tolist():
      modes = df[col].mode()
      if not modes.empty:
        mode = modes[0]
        df[col].fillna(mode, inplace=True)
    verResultado(df)

  if st.checkbox('SUBSTITUIR UM VALOR AUSENTE EM UMA COLUNA, COM A INTERPOLAÇÃO ENTRE O ANTERIOR E O SEGUINTE'):
    df[numerical] = df[numerical].interpolate(method = 'linear', limit_direction = 'forward')
    verResultado(df)

# ------------------------------------------------------------------------------

  # https://docs.streamlit.io/library/api-reference/widgets/st.download_button

  csv_processado = df.to_csv(index=False).encode('utf-8')
  st.download_button(label='Download CSV Processado', data=csv_processado, mime='text/csv', file_name=option.split('.')[0] + '_Processado.csv')

# ------------------------------------------------------------------------------

csv_original = df_csv.to_csv(index=False).encode('utf-8')
st.download_button(label='Download CSV Original', data=csv_original, mime='text/csv', file_name=option.split('.')[0] + '_Original.csv')

# ------------------------------------------------------------------------------
