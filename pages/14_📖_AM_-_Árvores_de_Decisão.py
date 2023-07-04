
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import PredefinedSplit, GridSearchCV # Para auxiliar na otimização de hiperparâmetros
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn import tree

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

def get_root_node(dt, feature_names):
    feature_idx = dt.tree_.feature[0]
    return feature_names[feature_idx]

def plota_arvore_de_decisao(dt, features_names, class_names):

  # ----------------------------------------------------------------------------

  root_node = get_root_node(dt, features_names)
  st.write(f"Nó raiz: {root_node}")

  # ----------------------------------------------------------------------------

  st.write("Modelo 1:")

  from sklearn.tree import plot_tree
  fig = plt.figure(figsize=(12,6))
  _ = plot_tree(dt, feature_names=features_names, class_names=class_names)
  st.pyplot(fig)

  # ----------------------------------------------------------------------------

  st.write("Modelo 2:")

  dot_data = tree.export_graphviz(dt,
                                out_file=None,
                                feature_names = features_names,
                                class_names = class_names,
                                filled=True)

  # https://docs.streamlit.io/library/api-reference/charts/st.graphviz_chart
  st.graphviz_chart(dot_data, use_container_width=True)

  graph = graphviz.Source(dot_data)
  graph.format = 'png'
  graph.render('DecisionTree', view = True)
  # https://docs.streamlit.io/library/api-reference/widgets/st.download_button
  with open("DecisionTree.png", "rb") as file:
    btn = st.download_button(label="Download árvore de decisão", data=file, file_name="DecisionTree.png", mime="image/png", key=f"download_button_{random.randint(0,100)}")

# ------------------------------------------------------------------------------

st.markdown("# Aprendizado Supervisionado - Árvores de Decisão")

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

if st.checkbox('ANÁLISE CAMINHO DE REDUÇÃO DE COMPLEXIDADE DE CUSTO'):

  # ----------------------------------------------------------------------------

  # Estima os valores de alpha efetivos e as respectivas impurezas.
  clf = DecisionTreeClassifier(random_state=0)
  path = clf.cost_complexity_pruning_path(X_train, y_train)
  ccp_alphas, impurities = path.ccp_alphas, path.impurities

  #st.write(f"ccp_alphas: {ccp_alphas.tolist()}")
  #st.write(f"impurities: {impurities.tolist()}")

  # Faz a análise visual da variação de impureza de acordo com o valor efetivo
  fig, ax = plt.subplots()
  ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
  ax.set_xlabel("ccp_alpha eficaz")
  ax.set_ylabel("Impureza total das folhas")
  ax.set_title("Impureza total vs alfa efetivo para o conjunto de treinamento")
  st.pyplot(fig)

  # ----------------------------------------------------------------------------

  # Treina diversas árvores de decisão, uma para cada valor efetivo de alpha
  clfs = []
  for ccp_alpha in ccp_alphas:
      clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
      clf.fit(X_train, y_train)
      clfs.append(clf)
  st.write(f"O número de nós na última árvore é: {clfs[-1].tree_.node_count} with ccp_alpha: {ccp_alphas[-1]}")

  # ----------------------------------------------------------------------------

  # Analisa a complexidade da árvore de acordo com a variação do valor efetivo de alpha
  # Retira o último valor de alpha do array, pois representa a poda mais drástica, que deixa
  # apenas o nó folha
  clfs = clfs[:-1]
  ccp_alphas = ccp_alphas[:-1]

  node_counts = [clf.tree_.node_count for clf in clfs]
  depth = [clf.tree_.max_depth for clf in clfs]
  fig, ax = plt.subplots(2, 1)
  ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
  ax[0].set_xlabel("ccp_alpha")
  ax[0].set_ylabel("Número de nós")
  ax[0].set_title("Número de nós vs ccp_alpha")
  ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
  ax[1].set_xlabel("ccp_alpha")
  ax[1].set_ylabel("Profundidade da árvore")
  ax[1].set_title("Profundidade x ccp_alpha")
  fig.tight_layout()
  st.pyplot(fig)

  # ----------------------------------------------------------------------------

  # Analisa a variação de desempenho para treino e teste.
  train_scores = [clf.score(X_train, y_train) for clf in clfs]
  test_scores = [clf.score(X_test, y_test) for clf in clfs]

  fig, ax = plt.subplots()
  ax.set_xlabel("ccp_alpha")
  ax.set_ylabel("Acurácia")
  ax.set_title("Acurácia vs ccp_alpha para conjuntos de treinamento e teste")
  ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
  ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
  ax.legend()
  plt.show()
  st.pyplot(fig)

  # ----------------------------------------------------------------------------

# ------------------------------------------------------------------------------

if st.checkbox('ÁRVORE DE DECISÃO COM PRÉ-PODA'):

  # ----------------------------------------------------------------------------

  # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

  criterion = st.selectbox("Critério", ["gini", "entropy"], key = "criterion")
  splitter = st.selectbox('Splitter', ["best", "random"], key = "splitter")
  max_depth = st.selectbox("Profundidade máxima", [None, 1, 2, 3, 4, 5], key = "max_depth")
  min_samples_split = st.selectbox("Número mínimo de amostras para dividir um nó", [2, 50], key = "min_samples_split")
  min_samples_leaf = st.slider("Número mínimo de amostras em uma folha", 1, 10, 4)
  max_leaf_nodes = st.selectbox('Hiperparâmetro max_leaf_nodes', [None, 50], key = "max_leaf_nodes")
  class_weight = st.selectbox('Hiperparâmetro class_weight', [None, 'balanced'], key = "class_weight")

  # Treina uma árvore de decisão
  dt = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=0.0, max_features=None, random_state=42, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=0.0, class_weight=class_weight, ccp_alpha=0.0)

  dt.fit(X_train, y_train)
  y_pred = dt.predict(X_test)

  # ----------------------------------------------------------------------------

  plota_arvore_de_decisao(dt, features_names, [str(x) for x in df[target].unique().tolist()])

  # ----------------------------------------------------------------------------

  mp.plota_todos_os_resultados(df[target].nunique(), dt.classes_, pos_label, neg_label, y_test, y_pred)

  # ----------------------------------------------------------------------------

if st.checkbox('TESTAR DIFERENTES VALORES PARA O HIPERPARÂMETRO CCP_ALPHA'):

  # ----------------------------------------------------------------------------

  # Para armazenar desempenhos em treino e teste

  perf_train = []

  perf_test = []

  # Definindo manualmente o intervalo de valores a ser testado para ccp
  ccps = [k * 0.001 for k in range(0, 75, 3)]

  for ccp in ccps:

    dt = DecisionTreeClassifier(ccp_alpha=ccp, random_state=42)
    dt.fit(X_train, y_train)

    # --------------------------------------------------------------------------

    y_pred_train = dt.predict(X_train)

    y_pred_test = dt.predict(X_test)

    # --------------------------------------------------------------------------

    perf_train.append([ccp,
                      accuracy_score(y_train, y_pred_train),
                      recall_score(y_train, y_pred_train, pos_label=pos_label, average=average),
                      precision_score(y_train, y_pred_train, pos_label=pos_label, average=average),
                      f1_score(y_train, y_pred_train, pos_label=pos_label, average=average)
                      ])

    # --------------------------------------------------------------------------

    perf_test.append([ccp,
                      accuracy_score(y_test, y_pred_test),
                      recall_score(y_test, y_pred_test, pos_label=pos_label, average=average),
                      precision_score(y_test, y_pred_test, pos_label=pos_label, average=average),
                      f1_score(y_test, y_pred_test, pos_label=pos_label, average=average)
                      ])

  # ----------------------------------------------------------------------------

  mp.plota_comparacao_treino_teste(perf_train, perf_test, "ccp_alpha")

  # ----------------------------------------------------------------------------

if st.checkbox('ÁRVORE DE DECISÃO COM PÓS-PODA: CUSTO-COMPLEXIDADE'):

  # ----------------------------------------------------------------------------

  # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

  ccp_alpha = list(dict.fromkeys([k * 0.001 for k in range(0, 100, 3)]))
  ccp_alpha.remove(0)

  st.write('ccp_alpha: Parâmetro de complexidade usado para remoção de complexidade de custo mínimo. A subárvore com maior complexidade de custo menor que ccp_alpha será escolhida. Por padrão, nenhuma remoção é executada. Consulte Corte de complexidade de custo mínimo para obter detalhes.')
  ccp_alpha = st.selectbox('Hiperparâmetro ccp_alpha', ccp_alpha[::-1], key = "ccp_alpha")

  # Treina uma árvore de decisão com configurações 'padrão'
  dt = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
  dt.fit(X_train, y_train)
  y_pred = dt.predict(X_test)

  # ----------------------------------------------------------------------------

  plota_arvore_de_decisao(dt, features_names, [str(x) for x in df[target].unique().tolist()])

  # ----------------------------------------------------------------------------

  mp.plota_todos_os_resultados(df[target].nunique(), dt.classes_, pos_label, neg_label, y_test, y_pred)

# ------------------------------------------------------------------------------
