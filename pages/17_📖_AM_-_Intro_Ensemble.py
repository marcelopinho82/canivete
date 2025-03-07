
# https://colab.research.google.com/drive/1-FKsP3km5_FnqbQl7_89WErj6Bj5m8eh?usp=sharing#scrollTo=C4mdOXxHGk5e

# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV # Para auxiliar na otimização de hiperparâmetros
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn import tree
from sklearn.base import clone

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

st.markdown("# Aprendizado Supervisionado - Introdução ao Aprendizado Ensemble - \"Random Forest\" Manual")

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

X_train, y_train, X_test, y_test, X_valid, y_valid = mp.metodo_holdout_3(df_X, df_y.values, 0.80, True)

# ------------------------------------------------------------------------------

n_trees = 501

if (len(X_train)) > 100:
  n_instances = 100
else:
  n_instances = st.selectbox('Selecione o tamanho da amostra', list(range(int(len(X_train)*10/100), int(len(X_train)*50/100), 1))[::-1], key="tamanho_amostra")

mini_sets = []

# Amostrando 501 conjuntos de dados com 100 instâncias aleatórias (a partir dos 80% p/ treino)
# Cria diversos conjuntos de treino/teste independentes
# Deixa 'n_instances' instâncias para treinamento

rs = ShuffleSplit(n_splits=n_trees, train_size=n_instances, random_state=42)

for mini_train_index, mini_test_index in rs.split(X_train):
  X_mini_train = X_train[mini_train_index]
  y_mini_train = y_train[mini_train_index]
  mini_sets.append((X_mini_train, y_mini_train))

# ------------------------------------------------------------------------------

# Otimiza os hiperparâmetros do algoritmo de árvores de decisão usando
# função GridSearchCV do scikit-learn e X_train.

st.subheader("ENCONTRANDO A MELHOR CONFIGURAÇÃO DE HIPERPARÂMETROS")

metrica = st.selectbox('Métrica de desempenho buscada', ['Acurácia', 'Precisão', 'Recall'], key = "gridsearch_metrica")
scoring = {'Acurácia':'accuracy', 'Precisão': 'precision', 'Recall':'recall'}

# Hiperparâmetros
profundidade_maxima = st.slider("Profundidade máxima", 1, 10, (3, 6))
criterio = st.selectbox("Critério de divisão", ["gini", "entropy"])
min_samples_split = st.slider("Número mínimo de amostras para dividir um nó", 2, 20, (2, 10))
min_samples_leaf = st.slider("Número mínimo de amostras em uma folha", 1, 10, (1, 4))

params = {
  'max_depth': np.arange(profundidade_maxima[0],profundidade_maxima[1]).tolist(),
  'criterion': [criterio],
  'min_samples_split': np.arange(min_samples_split[0],min_samples_split[1]).tolist(),
  'min_samples_leaf': np.arange(min_samples_leaf[0],min_samples_leaf[1]).tolist()
}
st.write(params)

if st.button('Calcular', key=f"calcular_button_1"):

  estimator = DecisionTreeClassifier(random_state=42) # Define o algoritmo base da otimização de hiperparâmetros

  grid_search_cv = GridSearchCV(estimator=estimator, param_grid=params, verbose=1, cv=3, n_jobs=-1, scoring=scoring, refit=metrica, return_train_score=True)

  grid_search_cv.fit(X_train, y_train)

  # ----------------------------------------------------------------------------

  mp.plota_resultado_gridsearchcv(grid_search_cv)

  mp.plota_grafico_parametro_gridsearchcv(grid_search_cv, scoring, 'max_depth', float)

  mp.plota_grafico_parametro_gridsearchcv(grid_search_cv, scoring, 'min_samples_split', float)

  # ----------------------------------------------------------------------------

  y_pred = grid_search_cv.predict(X_test)

  mp.plota_todos_os_resultados(df[target].nunique(), grid_search_cv.classes_, pos_label, neg_label, y_test, y_pred)

  # ----------------------------------------------------------------------------

  # Constrói um novo 'estimador' do scikit-learn com os mesmos hiperparâmetros
  # mas sem estar ajustados aos dados (ou seja, não copia os dados, apenas hiperparâmetros)

  st.subheader("TREINANDO MÚLTIPLAS ÁRVORES DE DECISÃO")

  setTrees = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

  # ----------------------------------------------------------------------------

  # Faz o treinamento de cada uma das 'n_trees' árvores

  perf_scores = []

  df_proba = pd.DataFrame()
  df2 = pd.DataFrame()

  n = 0
  for tree, (X_mini_train, y_mini_train) in zip(setTrees, mini_sets):

    tree.fit(X_mini_train, y_mini_train)
    y_pred = tree.predict(X_test)
    y_pred_proba = tree.predict_proba(X_test)

    # --------------------------------------------------------------------------

    df_tree = pd.DataFrame(y_pred_proba, columns = tree.classes_)

    for coluna in df[target].unique().tolist():
      if coluna not in df_tree:
        df_tree[coluna] = 0

    df_proba = df_proba.append(df_tree)

    df3 = df_tree.loc[:,[pos_label]]
    df2 = df2.append(df3.transpose())

    # --------------------------------------------------------------------------

    perf_scores.append([n,
                        accuracy_score(y_test, y_pred),
                        recall_score(y_test, y_pred, pos_label=pos_label, average=average),
                        precision_score(y_test, y_pred, pos_label=pos_label, average=average),
                        f1_score(y_test, y_pred, pos_label=pos_label, average=average)
                        ])

    n = n + 1

  # ----------------------------------------------------------------------------

  mp.plotar_todas_as_metricas_na_mesma_figura(perf_scores, 'Árvore', drop_duplicates = True)

  # ----------------------------------------------------------------------------

  mp.plota_heatmap_probabilidades_classe(df_proba, classes = df[target].unique().tolist(), n_classes = df[target].nunique())

  # ----------------------------------------------------------------------------

  mp.plota_heatmap_probabilidade_classe_pos_label(df2, pos_label)

  # ----------------------------------------------------------------------------

  st.subheader("AGREGANDO AS PREDIÇÕES DE MÚLTIPLOS CLASSIFICADORES")

  # ------------------------------------------------------------------------------

  Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

  # Agrega a classe predita por todas as árvores em um array de 2D
  for tree_index, tree in enumerate(setTrees):
      Y_pred[tree_index] = tree.predict(X_test)

  st.write("Observando a saída gerada para as instâncias de teste")
  st.write(Y_pred[:,:])

  # ----------------------------------------------------------------------------

  from scipy.stats import mode

  # Determina a saída a partir da moda (valor mais frequente)
  y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

  st.write("Observa a classe predita por votação majoritária")
  st.write(y_pred_majority_votes)

  st.write("Número de votos que a classe majoritária recebeu")
  st.write(n_votes)

  # ----------------------------------------------------------------------------

  st.subheader("AVALIA O DESEMPENHO DA CLASSIFICAÇÃO A PARTIR DE UMA VOTAÇÃO MAJORITÁRIA")

  if df[target].nunique() == 2:
    mp.plota_metricas_duas_classes(y_test, y_pred_majority_votes.reshape([-1]), pos_label)
  else:
    mp.plota_metricas_multiclasse(y_test, y_pred_majority_votes.reshape([-1]), pos_label)

  # ----------------------------------------------------------------------------

  from sklearn.metrics import classification_report
  classification_report = classification_report(y_test, y_pred_majority_votes.reshape([-1]), output_dict=True)
  df_classification_report = pd.DataFrame(classification_report).transpose()
  st.write(df_classification_report)

  # ----------------------------------------------------------------------------
