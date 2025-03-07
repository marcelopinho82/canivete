
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd

# ------------------------------------------------------------------------------
  
def verResultadoTemporario(df_temporario, file_name):
    st.write("Resultado Pré-Processamento")
    st.dataframe(df_temporario)

    csv_temp = df_temporario.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Download CSV Processado',
        data=csv_temp,
        mime='text/csv',
        file_name=file_name
    )

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

st.markdown("# Dados Ausentes - Pré-Processamento")

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

st.metric("Valores Ausentes", df.isna().sum().sum(), delta=None, delta_color="normal", help=None)

# ------------------------------------------------------------------------------

# Identificar colunas categóricas e numéricas com valores ausentes
colunas_numericas_ausentes = df.select_dtypes(exclude=['object']).columns[df.select_dtypes(exclude=['object']).isna().any()].tolist()
colunas_categoricas_ausentes = df.select_dtypes(include=['object']).columns[df.select_dtypes(include=['object']).isna().any()].tolist()

# Mostrar opções se houver colunas numéricas com valores ausentes
if colunas_numericas_ausentes:
    st.subheader("Tratamentos para colunas numéricas com valores ausentes")
    
    if st.checkbox('Preencher os valores NaN das colunas numéricas com o valor médio'):
        df_temp = df.copy()
        for col in colunas_numericas_ausentes:
            df_temp[col].fillna(df_temp[col].mean(), inplace=True)
        verResultadoTemporario(df_temp, file_name=option.split('.')[0] + '_Preenchido_Com_Media.csv')

    if st.checkbox('Preencher os valores NaN das colunas numéricas com a mediana'):
        df_temp = df.copy()
        for col in colunas_numericas_ausentes:
            df_temp[col].fillna(df_temp[col].median(), inplace=True)
        verResultadoTemporario(df_temp, file_name=option.split('.')[0] + '_Preenchido_Com_Mediana.csv')

    if st.checkbox('Interpolar valores ausentes com interpolação linear'):
        df_temp = df.copy()
        df_temp[colunas_numericas_ausentes] = df_temp[colunas_numericas_ausentes].interpolate(method='linear', inplace=False)
        verResultadoTemporario(df_temp, file_name=option.split('.')[0] + '_Interpolado_Linearmente.csv')

# Mostrar opções se houver colunas categóricas com valores ausentes
if colunas_categoricas_ausentes:
    st.subheader("Tratamentos para colunas categóricas com valores ausentes")

    if st.checkbox('Substituir valores ausentes pelo valor mais frequente (Moda)'):
        df_temp = df.copy()
        for col in colunas_categoricas_ausentes:
            mode = df_temp[col].mode()[0]
            df_temp[col].fillna(mode, inplace=True)
        verResultadoTemporario(df_temp, file_name=option.split('.')[0] + '_Preenchido_Com_Moda.csv')

if colunas_numericas_ausentes or colunas_categoricas_ausentes:

    # Tratamentos gerais para ambas as categorias
    if st.checkbox('Descartar linhas com valores ausentes'):
        df_temp = df.copy()
        df_temp.dropna(axis=0, inplace=True)
        verResultadoTemporario(df_temp, file_name=option.split('.')[0] + '_Sem_Linhas_Com_Valores_Ausentes.csv')

    if st.checkbox('Descartar colunas com valores ausentes'):
        df_temp = df.copy()
        df_temp.dropna(axis=1, inplace=True)
        verResultadoTemporario(df_temp, file_name=option.split('.')[0] + '_Sem_Colunas_Com_Valores_Ausentes.csv')
        
else:
    st.success("Nenhum dado ausente encontrado.")

# ------------------------------------------------------------------------------

csv_original = df_csv.to_csv(index=False).encode('utf-8')
st.download_button(label='Download CSV Original', data=csv_original, mime='text/csv', file_name=option)
    
# ------------------------------------------------------------------------------
