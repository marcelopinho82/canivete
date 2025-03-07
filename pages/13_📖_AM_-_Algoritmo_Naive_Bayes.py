
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

st.markdown("# Aprendizado Supervisionado - Algoritmo Naïve Bayes")

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

# https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas

# ------------------------------------------------------------------------------

target = st.selectbox('Selecione a coluna alvo (target)', df.columns.tolist()[::-1], key = "target")

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

if numerical:

  if st.checkbox('MODELO NAÏVE BAYES GAUSSIANO'):

    X_train, X_test, y_train, y_test = mp.metodo_holdout_2(df_X[numerical], df_y.values, 0.80, True)

    # Treinando o modelo com GaussianNB
    nbG_clf = GaussianNB()
    nbG_clf.fit(X_train, y_train)

    # Prevendo a classe de saída para as instâncias de teste
    # Por padrão, a classe retornada é aquela que maximiza a probabilidade a posteriori
    y_predG = nbG_clf.predict(X_test)

    # As probabilidades por classe podem ser observadas com o seguinte comando
    y_probaG = nbG_clf.predict_proba(X_test)

    # --------------------------------------------------------------------------

    mp.plota_todos_os_resultados(df[target].nunique(), nbG_clf.classes_, pos_label, neg_label, y_test, y_predG)

# ------------------------------------------------------------------------------

if categorical:

  if st.checkbox('MODELO NAÏVE BAYES COM DADOS CATEGÓRICOS'):

    X_train, X_test, y_train, y_test = mp.metodo_holdout_2(df_X[categorical], df_y.values, 0.80, True)

    # Treinando o modelo com CategoricalNB
    nbC_clf = CategoricalNB()
    nbC_clf.fit(X_train, y_train)

    # Prevendo a classe de saída para as instâncias de teste
    # Por padrão, a classe retornada é aquela que maximiza a probabilidade a posteriori
    y_predC = nbC_clf.predict(X_test)

    # As probabilidades por classe podem ser observadas com o seguinte comando
    y_probaC = nbC_clf.predict_proba(X_test)

    # --------------------------------------------------------------------------

    mp.plota_todos_os_resultados(df[target].nunique(), nbC_clf.classes_, pos_label, neg_label, y_test, y_predC)

  # ----------------------------------------------------------------------------

# ------------------------------------------------------------------------------
