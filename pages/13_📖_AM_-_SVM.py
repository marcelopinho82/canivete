
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC # Para treinar um SVM
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import PredefinedSplit, GridSearchCV # Para auxiliar na otimização de hiperparâmetros

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

st.markdown("# Aprendizado Supervisionado - Máquinas de Vetores de Suporte")

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

# https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas

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

if st.checkbox('OTIMIZAÇÃO DE HIPERPARÂMETROS COM GRIDSEARCHCV'):

  metrica = st.selectbox('Métrica de desempenho buscada', ['Acurácia', 'Precisão', 'Recall'], key = "gridsearch_metrica")

  # ----------------------------------------------------------------------------

  # Hiperparâmetros

  kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], key = "gridsearch_kernel")
  degree = st.slider("Grau", 1, 5, (2, 4), key = "gridsearch_degree")
  gamma = st.selectbox("Gamma", ["scale", "auto"], key = "gridsearch_gamma")
  probability = st.selectbox("Probabilidade", [True, False], key = "gridsearch_probability")
  tol = st.slider("Tolerância", 0.001, 0.1, (0.001, 0.01), key = "gridsearch_tol")
  class_weight = st.selectbox('Peso da classe', ['balanced', None])

  from sklearn.model_selection import PredefinedSplit, GridSearchCV # Para auxiliar na otimização de hiperparâmetros

  # Cria lista com os dados de treinamento com índice -1 e dados de validação com índice 0
  # Concatena os dados de treino e validação com as partições pré-definidas
  split_index = [-1]*len(X_train) + [0]*len(X_valid)
  X_gridSearch = np.concatenate((X_train, X_valid), axis=0)
  y_gridSearch = np.concatenate((y_train, y_valid), axis=0)
  pds = PredefinedSplit(test_fold = split_index)

  # Define métricas de desempenho a serem estimadas
  scoring = {'Acurácia':'accuracy', 'Precisão': 'precision', 'Recall':'recall'}
  estimator = SVC(class_weight=class_weight) # Define o algoritmo base da otimização de hiperparâmetros
  param_grid = {
    'C': [0.1, 1, 5, 10, 50, 100],
    'kernel': [kernel],
    'degree': np.arange(degree[0],degree[1]).tolist(),
    'gamma': [gamma],
    'probability': [probability],
    'tol': np.arange(tol[0],tol[1]).tolist()
  }

  # Aplica GridSearch com as partições de treino/validação pré-definidas
  grid_search_cv = GridSearchCV(estimator = estimator,
                   cv=pds,
                   n_jobs=-1,
                   param_grid=param_grid,
                   scoring=scoring,
                   refit=metrica, # Métrica a ser utilizada para definir o melhor modelo, retreinando-o com toda a base
                   return_train_score=True)
  grid_search_cv.fit(X_gridSearch, y_gridSearch)

  # ----------------------------------------------------------------------------

  mp.plota_resultado_gridsearchcv(grid_search_cv)

  # ----------------------------------------------------------------------------

  if kernel == 'linear':
    mp.plota_grafico_parametro_gridsearchcv(grid_search_cv, scoring, 'C', float)
  else:
    mp.plota_grafico_parametro_gridsearchcv(grid_search_cv, scoring, 'kernel', None)

# ------------------------------------------------------------------------------

  y_pred = grid_search_cv.predict(X_test)

  mp.plota_todos_os_resultados(df[target].nunique(), grid_search_cv.classes_, pos_label, neg_label, y_test, y_pred)

# ------------------------------------------------------------------------------

if st.checkbox('OTIMIZAÇÃO \"MANUAL\" DO HIPERPARÂMETRO C'):

  # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

  kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], key = "manual_kernel")
  C = st.slider("C", 0.01, 100.0, (0.1, 1.0), key = "manual_C")
  degree = None
  if ((kernel == "rbf") or (kernel == "poly") or (kernel == "sigmoid")):
    degree = st.slider("Grau", 1, 5, 3, key = "manual_degree")
  gamma = None
  if ((kernel == "rbf") or (kernel == "poly") or (kernel == "sigmoid")):
    gamma = st.selectbox("Gamma", ["scale", "auto"], key = "manual_gamma")
  probability = st.selectbox("Probabilidade", [True, False], key = "manual_probability")
  tol = st.slider("Tolerância", 0.001, 0.1, 0.01, key = "manual_tol")
  class_weight = st.selectbox('Peso da classe', ['balanced', None])

  # Definindo arrays para armazenar as métricas de desempenho de cada modelo treinado e avaliado
  perf_valid = []

  # Definindo valores de C a serem testados
  param_grid_C = np.arange(C[0],C[1]).tolist() # Valores para C, termo de regularização

  # Treinando e avaliado os modelos com cada valor de hiperparâmetro especificado
  st.write(f"Calculando as métricas de desempenho para os modelos com C igual a: {param_grid_C}")
  for ii in range(len(param_grid_C)):

    if ((kernel == "rbf") or (kernel == "poly") or (kernel == "sigmoid")):
      clf = SVC(kernel=kernel, C=param_grid_C[ii], random_state=42, degree=degree, gamma=gamma, probability=probability, tol=tol, class_weight=class_weight)
    else:
      clf = SVC(kernel=kernel, C=param_grid_C[ii], random_state=42, probability=probability, tol=tol, class_weight=class_weight)

    clf.fit(X_train, y_train)
    pred_i = clf.predict(X_valid)
    perf_valid.append([param_grid_C[ii],
                      accuracy_score(y_valid, pred_i),
                      recall_score(y_valid, pred_i, pos_label=pos_label, average=average),
                      precision_score(y_valid, pred_i, pos_label=pos_label, average=average),
                      f1_score(y_valid, pred_i, pos_label=pos_label, average=average)
                      ])

  mp.plotar_todas_as_metricas_na_mesma_figura(perf_valid, 'C', drop_duplicates = True)

# ------------------------------------------------------------------------------
