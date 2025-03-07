
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from PIL import Image

# ------------------------------------------------------------------------------

import os
import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

# https://docs.streamlit.io/library/api-reference/text
# https://blog.streamlit.io/introducing-multipage-apps/

st.markdown("# Página Inicial 🧰")
st.title("O Canivete Suiço do Cientista de Dados")
st.header("Conjunto de Dados")

## Teste 12345

# ------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------

# Upload do arquivo
arquivo = st.file_uploader("Upload conjunto de dados:", type=['csv', 'xlsx', 'pickle', 'json'])

if arquivo:

    # nome_arquivo = arquivo.name.split('.')[0]
    # extensao = arquivo.name.split('.')[1]

    nome_arquivo, extensao = os.path.splitext(arquivo.name)
    extensao = extensao[1:].lower()  # Remove o ponto e converte para minúsculas

    try:
        # Carregamento do DataFrame conforme o tipo de arquivo
        if extensao == 'csv':
            df = pd.read_csv(arquivo, sep=',')
        elif extensao == 'xlsx':
            # Carrega todas as abas para seleção
            xls = pd.ExcelFile(arquivo, engine='openpyxl')
            aba_selecionada = st.selectbox("Selecione a aba:", xls.sheet_names)
            df = pd.read_excel(xls, sheet_name=aba_selecionada)
        elif extensao == 'pickle':
            df = pd.read_pickle(arquivo)
        elif extensao == 'json':
            df = pd.read_json(arquivo)
        else:
            st.error("Formato de arquivo não suportado!")

        if extensao == 'xlsx':
            # Salva o DataFrame como CSV
            csv_path = f"{nome_arquivo}_{aba_selecionada}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            st.success(f"Arquivo salvo como {csv_path}")
        else:
            # Salva o DataFrame como CSV
            csv_path = f"{nome_arquivo}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            st.success(f"Arquivo salvo como {csv_path}")
        
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")

# ------------------------------------------------------------------------------

from sklearn import datasets

# Lista de funções que carregam datasets em sklearn.datasets
dataset_loaders = [
    datasets.load_iris,
    datasets.load_breast_cancer,
    datasets.load_digits,
    datasets.load_diabetes,
    datasets.load_wine,
]

# Iterar sobre cada função de carregamento
for loader in dataset_loaders:
    dataset_name = loader.__name__.replace('load_', '')  # Nome do dataset
    try:
        # Carregar o dataset e converter para DataFrame
        data = loader()
        df = mp.sklearn_to_df(data)
        
        # Salvar o DataFrame como CSV
        csv_path = f'{dataset_name}.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"{dataset_name} salvo como {csv_path}")
        
    except Exception as e:
        print(f"Erro ao processar {dataset_name}: {e}")

# ------------------------------------------------------------------------------

# https://www.pluralsight.com/guides/deploying-image-classification-on-the-web-with-streamlit-and-heroku
# https://www.geeksforgeeks.org/python-pil-image-save-method/
# https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader

arquivo = st.file_uploader("Upload imagens:", type=['png'])
if arquivo is not None:
  image = Image.open(arquivo)
  image.save(fp='./' + arquivo.name)

# ------------------------------------------------------------------------------
