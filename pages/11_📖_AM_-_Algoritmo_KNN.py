
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

st.markdown("# Aprendizado Supervisionado - Algoritmo K-Nearest Neighbors e Introdução à avaliação de modelos")

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

X_train, y_train, X_test, y_test, X_valid, y_valid = mp.metodo_holdout_3(df_X, df_y.values, 0.70, True)

# ------------------------------------------------------------------------------

if st.checkbox('OTIMIZAÇÃO \"MANUAL\" DO HIPERPARÂMETRO K'):

  # Definindo arrays para armazenar as métricas de desempenho de cada modelo treinado e avaliado
  perf_valid = []

  k = st.slider("k", 1, int(len(X_train)*10/100), 1, key = "manual_K")

  # Definindo kmin e kmax
  k_min=1
  k_max=k

  # Treinando e avaliado os modelos com cada valor de hiperparâmetro especificado
  st.write(f"Calculando as métricas de desempenho para os modelos com k entre {k_min} e {k_max} (inclusive)")
  for ii in range(k_min,(k_max+1),2):

    knn = KNeighborsClassifier(n_neighbors=ii, metric='euclidean', weights='uniform')

    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_valid)
    perf_valid.append([ii,
                      accuracy_score(y_valid, pred_i),
                      recall_score(y_valid, pred_i, pos_label=pos_label, average=average),
                      precision_score(y_valid, pred_i, pos_label=pos_label, average=average),
                      f1_score(y_valid, pred_i, pos_label=pos_label, average=average)
                      ])

# ------------------------------------------------------------------------------

  mp.plotar_todas_as_metricas_na_mesma_figura(perf_valid, 'K', drop_duplicates = True)

# ------------------------------------------------------------------------------

if st.checkbox('MODELO KNN'):

  n_neighbors = st.selectbox('Selecione o número de vizinhos', list(range(1, int(len(X_train)*10/100), step)), key = "n_neighbors")

  dados_concatenados_x = np.concatenate((X_train, X_valid), axis=0)
  dados_concatenados_y = np.concatenate((y_train, y_valid), axis=0)

  # ----------------------------------------------------------------------------

  knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean', weights='uniform')
  knn.fit(dados_concatenados_x, dados_concatenados_y)

  y_pred = knn.predict(X_test)

  # ----------------------------------------------------------------------------

  mp.plota_todos_os_resultados(df[target].nunique(), knn.classes_, pos_label, neg_label, y_test, y_pred)

  # ----------------------------------------------------------------------------
