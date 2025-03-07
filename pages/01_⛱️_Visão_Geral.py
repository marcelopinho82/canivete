
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

df_types = pd.DataFrame({
    'Tipo': ['Categóricos', 'Numéricos'],
    'Quantidade': [len(df.select_dtypes(include=['object', 'bool', 'category']).columns), len(df.select_dtypes(include=['number']).columns)]
})
st.bar_chart(data=df_types, x='Tipo', y='Quantidade', width=0, height=0, use_container_width=True)

# ------------------------------------------------------------------------------

# Criar DataFrame com os tipos de dados
dtype_counts = df.dtypes.value_counts()
df_types = pd.DataFrame({'Tipo': dtype_counts.index.astype(str), 'Quantidade': dtype_counts.values})

# Criar as colunas no layout
columns = st.columns(len(df_types))

# Exibir métricas por tipo em colunas
for i, (idx, row) in enumerate(df_types.iterrows()):
    with columns[i]:
        st.metric(f"Tipo de Dado: {row['Tipo']}", f"{row['Quantidade']}")
        
st.bar_chart(data=df_types, x='Tipo', y='Quantidade', width=0, height=0, use_container_width=True)

# ------------------------------------------------------------------------------

st.write(f"Número de instâncias: Há {df.shape[0]} observações e {df.shape[1]} atributos neste conjunto de dados. Destes, {len(df.select_dtypes(include=['object']).columns.tolist())} categóricos e {len(df.select_dtypes(exclude=['object']).columns.tolist())} numéricos.")

if df.select_dtypes(include=['object']).columns.tolist():
    st.subheader("Atributos categóricos")
    st.write(f"Os atributos categóricos são: {', '.join(df.select_dtypes(include=['object']).columns.tolist())}")

if df.select_dtypes(exclude=['object']).columns.tolist():
    st.subheader("Atributos numéricos")
    st.write(f"Os atributos numéricos são: {', '.join(df.select_dtypes(exclude=['object']).columns.tolist())}")
    
# ------------------------------------------------------------------------------
    
# Exibir as colunas por tipo de dado
st.subheader("Atributos por tipo de dado")

# Iterar sobre os tipos de dados e exibir as colunas correspondentes
for data_type in df_types['Tipo'].unique():
    cols_of_type = df.select_dtypes(include=[data_type]).columns.tolist()
    if cols_of_type:  # Verifica se existem colunas do tipo
        st.write(f"As colunas do tipo {data_type} são: {', '.join(cols_of_type)}")

# ------------------------------------------------------------------------------