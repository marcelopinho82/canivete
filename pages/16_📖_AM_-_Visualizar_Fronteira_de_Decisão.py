
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import random

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

def amostragem(X_train, y_train, key):

  # Selecionar aleatoriamente instâncias para explorar a visualização da fronteira de decisão
  ninstances_training = st.selectbox('Selecione o tamanho da amostra', list(range(int(len(X_train)*10/100), len(X_train), 1)), key=key)
  selected_instances_training = list(random.sample(range(len(X_train)), ninstances_training))
  X_train_subset = X_train[selected_instances_training, :]
  y_train_subset = y_train[selected_instances_training]

  # ----------------------------------------------------------------------------

  # Analisar distribuição das instâncias entre as classes no conjunto de treino
  st.write("Analisar distribuição das instâncias entre as classes no conjunto de treino da amostra")
  unique, counts = np.unique(y_train_subset, return_counts=True)
  df_amostra = pd.DataFrame(list(dict(zip(unique, counts)).items()))
  df_amostra.columns = ['classe',target]
  df_amostra.set_index('classe', inplace=True)
  st.dataframe(df_amostra)

  return X_train_subset, y_train_subset, ninstances_training

# ------------------------------------------------------------------------------

def plot_scatter_plot(df, key):

  x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(exclude=['object']).drop(target, axis=1).columns.tolist(), key=f"x_{key}")
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes(exclude=['object']).drop(target, axis=1).columns.tolist()[::-1], key=f"y_{key}")
  atr = np.argwhere(df.drop(target, axis=1).columns.isin([x_column,y_column])).ravel()
  mychart = alt.Chart(df).mark_point().encode(x=x_column, y=y_column, color=alt.Column(field=target, type="nominal"), shape=alt.Column(field=target, type="nominal"), tooltip=df.columns.tolist())
  st.altair_chart(mychart.interactive(), use_container_width=True)

  return x_column, y_column, atr

# ------------------------------------------------------------------------------

st.markdown("# Aprendizado Supervisionado - Visualizar Fronteira de Decisão")

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

if not df[target].dtype.kind in 'iufc': # https://www.skytowner.com/explore/checking_if_column_is_numeric_in_pandas_dataframe
  valores_unicos = df[target].unique()
  valores_unicos_dict = dict(zip(valores_unicos, range(len(valores_unicos))))
  st.write("Conversão classe texto para número")
  for key in valores_unicos_dict:
    st.write(key, '->', valores_unicos_dict[key])
    df[target] = df[target].replace(key, valores_unicos_dict[key])
  df[target] = pd.to_numeric(df[target])

# ------------------------------------------------------------------------------

st.write(df[target].value_counts())

# ------------------------------------------------------------------------------

labels = df[target].unique().tolist()
pos_label = st.selectbox('Pos label', labels[::-1], key = "pos_label")
labels.remove(pos_label)
neg_label = st.selectbox('Neg label', labels[::1], key = "neg_label")

if df[target].nunique() == 2:
  st.write("Classificação binária")
  average = "binary"
  step=2
else:
  st.write("Classificação não binária")
  average = st.selectbox('Average', ['weighted','micro','macro'], key = "average")
  step=1

st.write(f"Average: {average}")
st.write(f"Step: {step}")

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

encoder = OrdinalEncoder(dtype=np.int64)
# Codifica variáveis categóricas usando inteiros
df_X = df
encoder.fit(df_X[categorical])
df_X[categorical] = encoder.transform(df_X[categorical])
df_X = df_X.drop(target, axis=1)
df_y = df.loc[:,[target]]

# ------------------------------------------------------------------------------

data_container = st.container()
with data_container:
  coluna_X, coluna_y = st.columns(2)
  with coluna_X:
    st.write("Atributos Preditores (X)")
    st.write(df_X.head())
  with coluna_y:
    st.write("Atributo Alvo (y)")
    st.write(df_y.head())

# ------------------------------------------------------------------------------
# Aprendizado Supervisionado
# ------------------------------------------------------------------------------

X = df_X.values
y = df_y.values

# ------------------------------------------------------------------------------

if st.checkbox('VISUALIZAR FRONTEIRA DE DECISÃO DO KNN'):

  x_column, y_column, atr = plot_scatter_plot(pd.concat([df_X, df_y], axis=1), "scatter_plot_1")
  X_train, X_test, y_train, y_test = mp.metodo_holdout_2(df_X, df_y.values, 0.90, True)
  X_train_subset, y_train_subset, ninstances_training = amostragem(X_train, y_train, "tamanho_amostra_1")

  n_neighbors = st.selectbox('Selecione o número de vizinhos', list(range(1, 60, step)), key = "n_neighbors")

  fig = plt.figure(figsize=(10, 8))
  plt.title("KNN: fronteira de decisão com {} instâncias".format(ninstances_training), fontsize=16)
  mp.plot_decision_boundaries(X_train_subset[:,atr], y_train_subset, x_column, y_column, 0.5, KNeighborsClassifier, n_neighbors=n_neighbors, metric='euclidean', weights='uniform')
  st.pyplot(fig)

# ------------------------------------------------------------------------------

if numerical:

  if (len(numerical) >= 2):

    if st.checkbox('VISUALIZAR FRONTEIRA DE DECISÃO DO MODELO NAÏVE BAYES GAUSSIANO'):

      st.write(f"Colunas disponíveis: {numerical}")

      x_column, y_column, atr = plot_scatter_plot(pd.concat([df_X[numerical], df_y], axis=1), "scatter_plot_2")
      X_train, X_test, y_train, y_test = mp.metodo_holdout_2(df_X[numerical], df_y.values, 0.90, True)
      X_train_subset, y_train_subset, ninstances_training = amostragem(X_train, y_train, "tamanho_amostra_2")

      fig = plt.figure(figsize=(10, 8))
      plt.title("Naïve Bayes Gaussiano: fronteira de decisão com {} instâncias".format(ninstances_training), fontsize=16)
      mp.plot_decision_boundaries(X_train_subset[:,atr], y_train_subset, x_column, y_column, 0.5, GaussianNB)
      st.pyplot(fig)

# ------------------------------------------------------------------------------

if categorical:

  if (len(categorical) >= 2):

    if st.checkbox('VISUALIZAR FRONTEIRA DE DECISÃO DO MODELO NAÏVE BAYES COM DADOS CATEGÓRICOS'):

      st.write(f"Colunas disponíveis: {categorical}")

      x_column, y_column, atr = plot_scatter_plot(pd.concat([df_X[categorical], df_y], axis=1), "scatter_plot_3")
      X_train, X_test, y_train, y_test = mp.metodo_holdout_2(df_X[categorical], df_y.values, 0.90, True)
      X_train_subset, y_train_subset, ninstances_training = amostragem(X_train, y_train, "tamanho_amostra_3")

      fig = plt.figure(figsize=(10, 8))
      plt.title("Naïve Bayes com Dados Categóricos: fronteira de decisão com {} instâncias".format(ninstances_training), fontsize=16)
      mp.plot_decision_boundaries(X_train_subset[:,atr], y_train_subset, x_column, y_column, 0.5, CategoricalNB)
      st.pyplot(fig)

# ------------------------------------------------------------------------------

if st.checkbox('VISUALIZAR FRONTEIRA DE DECISÃO DO MODELO SVM'):

  x_column, y_column, atr = plot_scatter_plot(pd.concat([df_X, df_y], axis=1), "scatter_plot_4")
  X_train, X_test, y_train, y_test = mp.metodo_holdout_2(df_X, df_y.values, 0.90, True)
  X_train_subset, y_train_subset, ninstances_training = amostragem(X_train, y_train, "tamanho_amostra_4")

  kernel = st.selectbox('Hiperparâmetro kernel', ['linear', 'rbf', 'poly', 'sigmoid'], key = "kernel")
  class_weight = st.selectbox('Hiperparâmetro class_weight', ['balanced', None], key = "class_weight")
  C = st.selectbox('Hiperparâmetro C', [0.1, 1, 5, 10, 50, 100], key = "C")

  # Visualizar a fronteira de decisão para o modelo SVM com kernel linear.
  # >>> Ajuste o valor de C para o melhor valor encontrado o GridSearchCV (1 é o valor padrão)

  fig = plt.figure(figsize=(10, 8))
  plt.title("SVM: fronteira de decisão com {} instâncias".format(ninstances_training), fontsize=16)
  mp.plot_decision_boundaries(X_train_subset[:,atr], y_train_subset, x_column, y_column, 0.1, SVC, kernel=kernel, class_weight=class_weight, C=C)
  st.pyplot(fig)

# ------------------------------------------------------------------------------

if st.checkbox('VISUALIZAR FRONTEIRA DE DECISÃO DA ÁRVORE DE DECISÃO - SEM PODA'):

  x_column, y_column, atr = plot_scatter_plot(pd.concat([df_X, df_y], axis=1), "scatter_plot_5")
  X_train, X_test, y_train, y_test = mp.metodo_holdout_2(df_X, df_y.values, 0.90, True)
  X_train_subset, y_train_subset, ninstances_training = amostragem(X_train, y_train, "tamanho_amostra_5")

  fig = plt.figure(figsize=(10, 8))
  plt.title("Árvore de Decisão: fronteira de decisão com {} instâncias".format(ninstances_training), fontsize=16)
  mp.plot_decision_boundaries(X_train_subset[:,atr], y_train_subset, x_column, y_column, 0.5, DecisionTreeClassifier, random_state=0)
  st.pyplot(fig)

# ------------------------------------------------------------------------------

if st.checkbox('VISUALIZAR FRONTEIRA DE DECISÃO DA ÁRVORE DE DECISÃO - PRÉ-PODA 1'):

  x_column, y_column, atr = plot_scatter_plot(pd.concat([df_X, df_y], axis=1), "scatter_plot_6")
  X_train, X_test, y_train, y_test = mp.metodo_holdout_2(df_X, df_y.values, 0.90, True)
  X_train_subset, y_train_subset, ninstances_training = amostragem(X_train, y_train, "tamanho_amostra_6")

  fig, axs = plt.subplots(2, 2, figsize = (15, 10), sharey=True, sharex=True)
  plt.sca(axs[0,0])
  mp.plot_decision_boundaries(X_train_subset[:,atr], y_train_subset, x_column, y_column, 0.5, DecisionTreeClassifier, random_state=0)
  plt.title("Sem restrição", fontsize=13)
  plt.sca(axs[0,1])
  mp.plot_decision_boundaries(X_train_subset[:,atr], y_train_subset, x_column, y_column, 0.5, DecisionTreeClassifier, random_state=0, max_depth=3)
  plt.title("max_depth=3", fontsize=13)
  plt.sca(axs[1,0])
  mp.plot_decision_boundaries(X_train_subset[:,atr], y_train_subset, x_column, y_column, 0.5, DecisionTreeClassifier, random_state=0, max_depth=2)
  plt.title("max_depth=2", fontsize=13)
  plt.sca(axs[1,1])
  mp.plot_decision_boundaries(X_train_subset[:,atr], y_train_subset, x_column, y_column, 0.5, DecisionTreeClassifier, random_state=0, max_depth=1)
  plt.title("max_depth=1", fontsize=13)
  plt.show()
  st.pyplot(fig)

# ------------------------------------------------------------------------------

if st.checkbox('VISUALIZAR FRONTEIRA DE DECISÃO DA ÁRVORE DE DECISÃO - PRÉ-PODA 2'):

  x_column, y_column, atr = plot_scatter_plot(pd.concat([df_X, df_y], axis=1), "scatter_plot_7")
  X_train, X_test, y_train, y_test = mp.metodo_holdout_2(df_X, df_y.values, 0.90, True)
  X_train_subset, y_train_subset, ninstances_training = amostragem(X_train, y_train, "tamanho_amostra_7")

  fig, axs = plt.subplots(ncols=3, figsize = (15, 4), sharey=True, sharex=True)
  plt.sca(axs[0])
  mp.plot_decision_boundaries(X_train_subset[:,atr], y_train_subset, x_column, y_column, 0.5, DecisionTreeClassifier, random_state=0)
  plt.title("Sem restrição", fontsize=13)
  plt.sca(axs[1])
  mp.plot_decision_boundaries(X_train_subset[:,atr], y_train_subset, x_column, y_column, 0.5, DecisionTreeClassifier, random_state=0, min_samples_leaf=3)
  plt.title("min_samples_leaf=3", fontsize=13)
  plt.sca(axs[2])
  mp.plot_decision_boundaries(X_train_subset[:,atr], y_train_subset, x_column, y_column, 0.5, DecisionTreeClassifier, random_state=0, min_samples_leaf=5)
  plt.title("min_samples_leaf=5", fontsize=13)
  plt.show()
  st.pyplot(fig)

# ------------------------------------------------------------------------------

if st.checkbox('VISUALIZAR FRONTEIRA DE DECISÃO DA ÁRVORE DE DECISÃO - PÓS-PODA'):

  x_column, y_column, atr = plot_scatter_plot(pd.concat([df_X, df_y], axis=1), "scatter_plot_8")
  X_train, X_test, y_train, y_test = mp.metodo_holdout_2(df_X, df_y.values, 0.90, True)
  X_train_subset, y_train_subset, ninstances_training = amostragem(X_train, y_train, "tamanho_amostra_8")

  fig, axs = plt.subplots(ncols=2, figsize = (15, 4), sharey=True, sharex=True)
  plt.sca(axs[0])
  mp.plot_decision_boundaries(X_train_subset[:,atr], y_train_subset, x_column, y_column, 0.5, DecisionTreeClassifier, random_state=0)
  plt.title("Sem restrição", fontsize=13)
  plt.sca(axs[1])
  mp.plot_decision_boundaries(X_train_subset[:,atr], y_train_subset, x_column, y_column, 0.5, DecisionTreeClassifier, random_state=0, ccp_alpha=0.05)
  plt.title("ccp_alpha=0.05", fontsize=13)
  plt.show()
  st.pyplot(fig)

# ------------------------------------------------------------------------------
