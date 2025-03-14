
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

st.markdown("# Aprendizado Supervisionado - Regressão Logística")

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

# https://machinelearningmastery.com/spot-check-classification-machine-learning-algorithms-python-scikit-learn/

# Logistic Regression Classification
X = df_X.values
y = df_y.values

kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X, y, cv=kfold)

st.metric("Média", results.mean(), delta=None, delta_color="normal", help=None)

# Ajustar o modelo completo e fazer predições no conjunto original
model.fit(X, y.ravel())  # Treina o modelo no conjunto completo
predictions = model.predict(X)  # Faz as predições

# Adiciona as previsões ao DataFrame original
df['Predictions'] = predictions

# ------------------------------------------------------------------------------

# Gerar o nome do arquivo com base no nome do arquivo original
file_name = option.split('.')[0] + '_Predições.xlsx'

# Baixar o DataFrame original com as predições
output_file = "/tmp/" + file_name
df.to_excel(output_file, index=False)

# Botão para download
st.download_button(
    label="Baixar DataFrame com Previsões",
    data=open(output_file, 'rb').read(),
    file_name=file_name,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
