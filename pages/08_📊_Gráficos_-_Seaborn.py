
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

# Gerar um gráfico para cada variável categórica com a distribuição de
# frequência entre as classes
def count_plot(df, columns, label):
    for indx, var in enumerate(columns):
      fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
      sns.color_palette("pastel")
      g = sns.countplot(x=var, data=df, hue=label)
      st.pyplot(fig)

# Gerar um gráfico para cada variável numérica com a distribuição de
# frequência entre as classes
def dist_plot(df, columns, label):
    for indx, var in enumerate(columns):
      fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
      sns.color_palette("pastel")
      g = sns.histplot(x=var, data=df, hue=label, binwidth=3, palette='muted')
      st.pyplot(fig)

# ------------------------------------------------------------------------------

# https://colab.research.google.com/drive/12bricBWt7LMlFyudUxBQF9n6XKHw1wjT?usp=sharing#scrollTo=7QEWJ4FdEtxa

# Distribuição geral
def dist_plot2(df, columns):
    for indx, var in enumerate(columns):
      fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
      sns.color_palette("pastel")
      g = sns.histplot(x=var, data=df, palette='muted')
      st.pyplot(fig)

# Distribuição por classe
def dist_plot_perclass(df, columns, label):
    for indx, var in enumerate(columns):
      fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
      sns.color_palette("pastel")
      g = sns.histplot(x=var, data=df, hue=label, palette='muted')
      st.pyplot(fig)

# ------------------------------------------------------------------------------

st.markdown("# Gráficos - Seaborn")

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

target = st.selectbox('Selecione a coluna alvo (target)', df.columns.tolist()[::-1], key = "target")

# ------------------------------------------------------------------------------

st.write(df[target].value_counts())

# ------------------------------------------------------------------------------

df_X = df.loc[:, df.columns.tolist()]
df_X = df_X.drop(target, axis=1)
df_y = df.loc[:,[target]]

X = df_X.values
y = df_y.values

# ------------------------------------------------------------------------------

features_names = df.columns.tolist()
if target in features_names:
  features_names.remove(target)

categorical = df.select_dtypes(include=['object']).columns.tolist()
if target in categorical:
  categorical.remove(target)

numerical = df.select_dtypes(exclude=['object']).columns.tolist()
if target in numerical:
  numerical.remove(target)

# ------------------------------------------------------------------------------
# Gráficos
# ------------------------------------------------------------------------------

if st.checkbox('DISTRIBUIÇÃO DO ATRIBUTO ALVO'):
  fig = plt.figure(figsize=(16, 10))
  plt.hist(df[target])
  plt.title("Distribuição do atributo alvo")
  plt.show()
  st.pyplot(fig)

if st.checkbox('DISTRIBUIÇÃO DOS ATRIBUTOS'):
  dist_plot2(df, features_names)

if st.checkbox('DISTRIBUIÇÃO DOS ATRIBUTOS POR CLASSE'):
  dist_plot_perclass(df, features_names, target)

# ------------------------------------------------------------------------------

if categorical:

  if st.checkbox('GERAR UM GRÁFICO PARA CADA VARIÁVEL CATEGÓRICA COM A DISTRIBUIÇÃO DE FREQUÊNCIA ENTRE AS CLASSES'):

    count_plot(df, categorical, target)

  if st.checkbox('GERAR UM GRÁFICO PARA APENAS UMA VARIÁVEL CATEGÓRICA COM A DISTRIBUIÇÃO DE FREQUÊNCIA ENTRE AS CLASSES'):

    x_column = st.selectbox('Selecione a coluna do eixo x', categorical, key = "x_count_plot_atributo_categorico")

    fig = plt.figure(figsize=(16, 10))
    sns.color_palette("pastel")
    sns.countplot(x=x_column, data=df, hue=target)
    st.pyplot(fig)

# ------------------------------------------------------------------------------

if numerical:

  if st.checkbox('GERAR UM GRÁFICO PARA CADA VARIÁVEL NUMÉRICA COM A DISTRIBUIÇÃO DE FREQUÊNCIA ENTRE AS CLASSES'):

    dist_plot(df, numerical, target)

  if st.checkbox('GERAR UM GRÁFICO PARA APENAS UMA VARIÁVEL NUMÉRICA COM A DISTRIBUIÇÃO DE FREQUÊNCIA ENTRE AS CLASSES'):

    x_column = st.selectbox('Selecione a coluna do eixo x', numerical, key = "x_hist_plot_atributo_numerico")

    fig = plt.figure(figsize=(16, 10))
    sns.color_palette("pastel")
    g = sns.histplot(x=x_column, data=df, hue=target, binwidth=3)
    st.pyplot(fig)

# ------------------------------------------------------------------------------

if st.checkbox('DISTRIBUIÇÃO DE TODOS OS ATRIBUTOS POR CLASSE'):

  ave_values = df.groupby(target).median()
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
  sns.color_palette("pastel")
  ave_values.plot(kind="bar", figsize=(15,7), ax=ax)
  st.pyplot(fig)

# ------------------------------------------------------------------------------

if numerical:

  if st.checkbox('DISTRIBUIÇÃO DE TODOS OS ATRIBUTOS POR CLASSE - NORMALIZADOS'):

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[numerical] = scaler.fit_transform(df_norm[numerical])
    ave_values = df_norm.groupby(target).median()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
    sns.color_palette("pastel")
    ave_values.plot(kind="bar", figsize=(15,7), ax=ax)
    st.pyplot(fig)
