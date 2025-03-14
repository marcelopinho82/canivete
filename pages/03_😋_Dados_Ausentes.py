
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

st.markdown("# Dados Ausentes")

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

if ((df.isna().sum().sum()) > 0):

    st.subheader("Visão geral")

    # https://stackoverflow.com/questions/54221673/is-there-a-way-to-export-pandas-dataframe-info-df-info-into-an-excel-file
    df_nulos = pd.DataFrame({
      "Atributo": df.columns,
      "Não Nulos": len(df)-df.isnull().sum().values,
      "Nulos": df.isnull().sum().values,
      "Tipo": df.dtypes.values
      })
    st.dataframe(df_nulos)

    st.subheader("Percentual de dados ausentes")

    df_medias_nulo = pd.DataFrame(df.isnull().mean().round(decimals=4))
    df_somas_nulo = pd.DataFrame(df.isnull().sum().round(decimals=4))
    df_medias_nao_nulo = pd.DataFrame(df.notnull().mean().round(decimals=4))
    df_somas_nao_nulo = pd.DataFrame(df.notnull().sum().round(decimals=4))
    new_df = pd.concat([df_somas_nulo, df_medias_nulo, df_somas_nao_nulo, df_medias_nao_nulo], axis=1)
    new_df.columns = ['Total Nulos', '% Nulos', 'Total Não Nulos', '% Não Nulos']
    e_nulo = new_df['Total Nulos']!=0
    new_df = new_df[e_nulo]
    st.dataframe(new_df)

    # ------------------------------------------------------------------------------

    st.subheader("Linhas com valores ausentes")

    st.dataframe(df[df.isna().any(axis=1)])

    # ------------------------------------------------------------------------------

    #https://subscription.packtpub.com/book/data/9781789806311/1/ch01lvl1sec04/quantifying-missing-data

    st.subheader("Quantificando dados ausentes")

    st.subheader("Pandas")

    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(1,1,1)
    df.isnull().mean().plot.bar(figsize=(10,5), ax=ax, log=True)
    plt.ylabel('Porcentagem de valores ausentes')
    plt.xlabel('Atributos')
    plt.title('Quantificando dados ausentes')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Missingno")

    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(1,1,1)
    msno.matrix(df, sparkline=True, figsize=(10,5), fontsize=12, color=(0.27, 0.52, 1.0), ax=ax);
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Pandas & Missingno")

    fig = plt.figure(figsize=(15,7))
    ax1 = fig.add_subplot(1,2,1)
    df.isnull().mean().plot.bar(figsize=(10,5), ax=ax1, log=True)
    plt.ylabel('Porcentagem de valores ausentes')
    plt.xlabel('Atributos')
    plt.title('Quantificando dados ausentes')

    ax2 = fig.add_subplot(1,2,2)
    msno.matrix(df, sparkline=True, figsize=(10,5), fontsize=12, color=(0.27, 0.52, 1.0), ax=ax2);
    plt.tight_layout()
    st.pyplot(fig)

    # ------------------------------------------------------------------------------

    # https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    # https://www.schemecolor.com/2022-fifa-world-cup-logo-colors-qatar.php

    st.subheader("Missingno bar normal")

    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(1,1,1)
    msno.bar(df, color="#FEC310", sort="ascending", fontsize=12, ax=ax);
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Missingno bar log")

    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(1,1,1)
    msno.bar(df, log=True, color="#56042C", sort="ascending", fontsize=12, ax=ax);
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Missingno bar normal & log")

    fig = plt.figure(figsize=(15,7))
    ax1 = fig.add_subplot(1,2,1)
    msno.bar(df, color="#FEC310", sort="ascending", fontsize=12, ax=ax1);
    ax2 = fig.add_subplot(1,2,2)
    msno.bar(df, log=True, color="#56042C", sort="ascending", fontsize=12, ax=ax2);
    plt.tight_layout()
    st.pyplot(fig)

    # ------------------------------------------------------------------------------

    # Gráfico

    # https://stackoverflow.com/questions/3584805/what-does-the-argument-mean-in-fig-add-subplot111

    st.subheader("Missingno matrix")

    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(1,1,1)
    msno_matrix = msno.matrix(df, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Missingno heatmap")

    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(1,1,1)
    msno_heatmap = msno.heatmap(df, fontsize=12, cmap="coolwarm", ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Correlações nulos")

    # Tabela
    df_correlacoes = mp.top_entries(df.isnull().corr())
    df_correlacoes = df_correlacoes[df_correlacoes['Correlação'].notna()]
    st.dataframe(df_correlacoes)

    # ------------------------------------------------------------------------------

    st.subheader("Missingno dendogram centroid")

    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(1,1,1)
    msno.dendrogram(df, orientation="right", method="centroid", fontsize=12, ax=ax);
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Missingno dendogram ward")

    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(1,1,1)
    msno.dendrogram(df, orientation="top", method="ward", fontsize=12, ax=ax);
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Missingno dendogram centroid & ward")

    fig = plt.figure(figsize=(15,7))
    ax1 = fig.add_subplot(1,2,1)
    msno.dendrogram(df, orientation="right", method="centroid", fontsize=12, ax=ax1);
    ax2 = fig.add_subplot(1,2,2)
    msno.dendrogram(df, orientation="top", method="ward", fontsize=12, ax=ax2);
    plt.tight_layout()
    st.pyplot(fig)
    
else:
    st.success("Nenhum dado ausente encontrado.")

# ------------------------------------------------------------------------------
