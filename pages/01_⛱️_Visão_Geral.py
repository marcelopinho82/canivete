
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

st.markdown("# Visão geral")

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

# https://imran-malik01.medium.com/data-science-and-machine-learning-web-application-medium-620cc6562d11
# https://stackoverflow.com/questions/32589829/how-to-get-value-counts-for-multiple-columns-at-once-in-pandas-dataframe
# https://stackoverflow.com/questions/22257527/how-do-i-get-a-summary-count-of-missing-nan-data-by-column-in-pandas

data_container = st.container()
with data_container:
  metrica1, metrica2 = st.columns(2)
  with metrica1:
    st.metric("Observações", df.shape[0], delta=None, delta_color="normal", help=None)
    st.metric("Categóricos", len(df.select_dtypes(include=['object']).columns.tolist()), delta=None, delta_color="normal", help=None)
  with metrica2:
    st.metric("Atributos", df.shape[1], delta=None, delta_color="normal", help=None)
    st.metric("Numéricos", len(df.select_dtypes(exclude=['object']).columns.tolist()), delta=None, delta_color="normal", help=None)

# ------------------------------------------------------------------------------

df_types = pd.DataFrame(df.dtypes, columns=['Tipo'])
numerical_cols = df_types[~df_types['Tipo'].isin(['object','bool'])].index.values
df_types['Count'] = df.count()
df_types['Valores Únicos'] = df.nunique()
df_types['Min'] = df[numerical_cols].min()
df_types['Max'] = df[numerical_cols].max()
df_types['Média'] = df[numerical_cols].mean()
df_types['Mediana'] = df[numerical_cols].median()
df_types['Desvio Padrão'] = df[numerical_cols].std()
st.write('Resumo:')
st.write(df_types)

# ------------------------------------------------------------------------------

st.write(f"Número de instâncias: Há {df.shape[0]} observações e {df.shape[1]} atributos neste conjunto de dados. Destes, {len(df.select_dtypes(include=['object']).columns.tolist())} categóricos e {len(df.select_dtypes(exclude=['object']).columns.tolist())} numéricos.")

st.write(f"Os atributos categóricos são: {', '.join(df.select_dtypes(include=['object']).columns.tolist())}")

st.write(f"Os atributos numéricos são: {', '.join(df.select_dtypes(exclude=['object']).columns.tolist())}")

i=1
for atributo in df.columns:
  st.write(f"{i}. {atributo.upper()}")
  st.write(f"{atributo.replace('_', ' ')} - ")
  st.write(f"\n")
  i=i+1
