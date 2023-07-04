
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from PIL import Image

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

# https://docs.streamlit.io/library/api-reference/text
# https://blog.streamlit.io/introducing-multipage-apps/

st.markdown("# Página Inicial 🧰")
st.title("O Canivete Suiço do Cientista de Dados")
st.header("Conjunto de Dados")

# ------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------

arquivo = st.file_uploader("Upload conjunto de dados:", type=['csv','xlsx','pickle','json'])
if arquivo:
  nome_arquivo = arquivo.name.split('.')[0]
  extensao = arquivo.name.split('.')[1]

  if extensao.lower() == 'csv':
    df = pd.read_csv(arquivo, sep=',')

  elif extensao.lower() == 'xlsx':
    df = pd.read_excel(arquivo, engine='openpyxl')

  elif extensao.lower() == 'pickle':
    df = pd.read_pickle(arquivo)

  elif extensao.lower() == 'json':
    df = pd.read_json(arquivo)

  df.to_csv(nome_arquivo + '.csv', index=False, encoding='utf-8')

# ------------------------------------------------------------------------------

from sklearn import datasets

df_iris = mp.sklearn_to_df(datasets.load_iris())
df_iris.to_csv('iris.csv', index=False, encoding='utf-8')

df_breast_cancer = mp.sklearn_to_df(datasets.load_breast_cancer())
df_breast_cancer.to_csv('load_breast_cancer.csv', index=False, encoding='utf-8')

# ------------------------------------------------------------------------------

# https://www.pluralsight.com/guides/deploying-image-classification-on-the-web-with-streamlit-and-heroku
# https://www.geeksforgeeks.org/python-pil-image-save-method/
# https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader

arquivo = st.file_uploader("Upload imagens:", type=['png'])
if arquivo is not None:
  image = Image.open(arquivo)
  image.save(fp='./' + arquivo.name)

# ------------------------------------------------------------------------------
