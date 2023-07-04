
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

st.markdown("# Correlações")

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

if st.checkbox("TABELA DE CORRELAÇÕES"):
  st.write(df.corr())

# ------------------------------------------------------------------------------

if st.checkbox("TABELA DE CORRELAÇÕES IDENTIFICADAS"):
  dfCorr = mp.top_entries(df)
  dfCorr.dropna(axis=0, inplace=True)
  st.dataframe(dfCorr)

if st.checkbox("TABELA DE CORRELAÇÕES FILTRADA"):
  dfCorr = mp.top_entries(df)
  dfCorr.dropna(axis=0, inplace=True)
  dfCorr = dfCorr.loc[((dfCorr['Correlação'] >= .5) | (dfCorr['Correlação'] <= -.5)) & (dfCorr['Correlação'] !=1.000)]
  st.dataframe(dfCorr)

# ------------------------------------------------------------------------------

# https://stackoverflow.com/questions/43335973/how-to-generate-high-resolution-heatmap-using-seaborn
# https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
# https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e

if st.checkbox("GRÁFICO DE CORRELAÇÕES"):

  dfCorr = df.corr()
  filteredDf = dfCorr
  fig = plt.figure(figsize=(15,15), dpi = 600)
  heatmap = sns.heatmap(filteredDf, vmin=-1, vmax=1, cbar=1, square=1, annot=True, cmap="PuOr", annot_kws={"size": 15})
  heatmap.set_title('Mapa de Calor de Correlações', fontdict={'fontsize':18}, pad=16);
  plt.tick_params(axis = 'x', labelsize = 12) # x font label size
  plt.tick_params(axis = 'y', labelsize = 12) # y font label size
  st.pyplot(fig)

  dfCorr = df.corr()
  filteredDf = dfCorr[(dfCorr !=1.000)]
  fig = plt.figure(figsize=(15,15), dpi = 600)
  heatmap = sns.heatmap(filteredDf, vmin=-1, vmax=1, cbar=1, square=1, annot=True, cmap="PuOr", annot_kws={"size": 15})
  heatmap.set_title('Mapa de Calor de Correlações', fontdict={'fontsize':18}, pad=16);
  plt.tick_params(axis = 'x', labelsize = 12) # x font label size
  plt.tick_params(axis = 'y', labelsize = 12) # y font label size
  st.pyplot(fig)

# ------------------------------------------------------------------------------

if st.checkbox("GRÁFICO DE CORRELAÇÕES TRIANGULAR"):

  # https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e

  fig = plt.figure(figsize=(15,15), dpi = 600)
  mask = np.triu(np.ones_like(df.corr(), dtype=np.bool)) # Define the mask to set the values in the upper triangle to True
  heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, cbar=1, square=1, annot=True, cmap="PuOr", annot_kws={"size": 15})
  heatmap.set_title('Mapa de Calor de Correlações', fontdict={'fontsize':18}, pad=16);
  plt.tick_params(axis = 'x', labelsize = 12) # x font label size
  plt.tick_params(axis = 'y', labelsize = 12) # y font label size
  st.pyplot(fig)

# ------------------------------------------------------------------------------

if st.checkbox("GRÁFICO DE CORRELAÇÕES FILTRADO"):

  dfCorr = df.corr()
  filteredDf = dfCorr[((dfCorr >= .5) | (dfCorr <= -.5)) & (dfCorr !=1.000)]
  fig = plt.figure(figsize=(15,15), dpi = 600)
  heatmap = sns.heatmap(filteredDf, vmin=-1, vmax=1, cbar=1, square=1, annot=True, cmap="PuOr", annot_kws={"size": 15})
  heatmap.set_title('Mapa de Calor de Correlações - Filtrado', fontdict={'fontsize':18}, pad=16);
  plt.tick_params(axis = 'x', labelsize = 12) # x font label size
  plt.tick_params(axis = 'y', labelsize = 12) # y font label size
  st.pyplot(fig)

  dfCorr = df.corr()
  filteredDf = dfCorr[((dfCorr >= .5) | (dfCorr <= -.5)) & (dfCorr !=1.000)]
  fig = plt.figure(figsize=(15,15), dpi = 600)
  mask = np.triu(np.ones_like(filteredDf, dtype=np.bool)) # Define the mask to set the values in the upper triangle to True
  heatmap = sns.heatmap(filteredDf, mask=mask, vmin=-1, vmax=1, cbar=1, square=1, annot=True, cmap="PuOr", annot_kws={"size": 15})
  heatmap.set_title('Mapa de Calor de Correlações - Filtrado', fontdict={'fontsize':18}, pad=16);
  plt.tick_params(axis = 'x', labelsize = 12) # x font label size
  plt.tick_params(axis = 'y', labelsize = 12) # y font label size
  st.pyplot(fig)

# ------------------------------------------------------------------------------

if st.checkbox("CORRELAÇÃO DE VARIÁVEIS ​​INDEPENDENTES COM A VARIÁVEL DEPENDENTE"):

  # https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e

  target = st.selectbox('Selecione a coluna alvo (target)', df.columns.tolist()[::-1], key = "corr_target")

  dfCorr = df.corr()
  filteredDf = dfCorr[((dfCorr >= .5) | (dfCorr <= -.5)) & (dfCorr !=1.000)]
  st.dataframe(filteredDf[[target]].dropna(axis=0).sort_values(by=target, ascending=False))
  fig = plt.figure(figsize=(8, 12))
  heatmap = sns.heatmap(filteredDf[[target]].dropna(axis=0).sort_values(by=target, ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
  heatmap.set_title(f'Atributos correlacionados com {target}', fontdict={'fontsize':18}, pad=16);
  st.pyplot(fig)

# ------------------------------------------------------------------------------

if st.checkbox('GRÁFICO DE LINHA'):

  target = st.selectbox('Selecione a coluna alvo (target)', df.columns.tolist()[::-1], key = "grafico_de_linha_target")

  # ------------------------------------------------------------------------------

  numerical = df.select_dtypes(exclude=['object']).columns.tolist()
  if target in numerical:
    numerical.remove(target)

  # ------------------------------------------------------------------------------

  y_column = st.selectbox('Selecione a coluna do eixo y', numerical, key = "y_line_plot_atributo_numerico")

  fig = plt.figure(figsize=(10,5))
  sns.lineplot(data=df, x=target, y=y_column, color='r')
  st.pyplot(fig)
