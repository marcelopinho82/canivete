
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
from sklearn.model_selection import PredefinedSplit, GridSearchCV # Para auxiliar na otimização de hiperparâmetros
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from statistics import mode

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.tree import ExtraTreeClassifier
from sklearn.dummy import DummyClassifier

from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import NuSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import HuberRegressor, RidgeCV, BayesianRidge, Ridge, LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV, LassoLarsCV, LassoLarsIC, LarsCV, Lars
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LassoLars
from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import CategoricalNB
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import MultinomialNB

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

st.markdown("# Aprendizado Supervisionado - Aprendizado Ensemble - Spot-checking algorithms - Com otimização de hiperparâmetros")

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

X_train, y_train, X_test, y_test, X_valid, y_valid = mp.metodo_holdout_3(df_X, df_y.values, 0.80, True)

# ------------------------------------------------------------------------------

tarefa = st.selectbox('Selecione a tarefa', ['Classificação', 'Classificação Ensemble - BaggingClassifier', 'Classificação Ensemble - AdaBoostClassifier', 'Classificação Ensemble - VotingClassifier 1', 'Classificação Ensemble - VotingClassifier 2', 'Classificação Ensemble - StackingClassifier','Classificação Ensemble - ExtraTreesClassifier','Classificação Ensemble - GradientBoostingClassifier'], key = "tarefa")

if (tarefa == "Regressão"):

  # Crie um dicionário de regressores para testar
  dict_algoritmos = {
      'SVR': {'algoritmo': SVR(), 'param_grid': {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear', 'poly', 'rbf']}},
      'RandomForestRegressor': {'algoritmo': RandomForestRegressor(), 'param_grid': {'n_estimators': [10, 50, 100, 200], 'max_depth': [3, 5, 7, None]}},
      'ExtraTreesRegressor': {'algoritmo': ExtraTreesRegressor(), 'param_grid': {'n_estimators': [10, 50, 100, 200], 'max_depth': [3, 5, 7, None]}},
      'AdaBoostRegressor': {'algoritmo': AdaBoostRegressor(), 'param_grid': {'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.1, 0.5, 1]}},
      'NuSVR': {'algoritmo': NuSVR(), 'param_grid': {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear', 'poly', 'rbf']}},
      'GradientBoostingRegressor': {'algoritmo': GradientBoostingRegressor(), 'param_grid': {'learning_rate': [0.1, 0.5, 1], 'n_estimators': [10, 50, 100, 200]}},
      'KNeighborsRegressor': {'algoritmo': KNeighborsRegressor(), 'param_grid': {'n_neighbors': [3, 5, 7, 9]}},
      'HistGradientBoostingRegressor': {'algoritmo': HistGradientBoostingRegressor(), 'param_grid': {'learning_rate': [0.1, 0.5, 1], 'max_depth': [3, 5, 7, None], 'max_iter': [10, 50, 100, 200]}},
      'BaggingRegressor': {'algoritmo': BaggingRegressor(), 'param_grid': {'n_estimators': [10, 50, 100, 200], 'max_samples': [0.5, 1, 2]}},
      'MLPRegressor': {'algoritmo': MLPRegressor(), 'param_grid': {'hidden_layer_sizes': [(10,), (50,), (100,), (10,10,), (50,50,), (100,100,)], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}},
      'HuberRegressor': {'algoritmo': HuberRegressor(), 'param_grid': {'epsilon': [0.1, 1, 10], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}},
      'LinearSVR': {'algoritmo': LinearSVR(), 'param_grid': {'C': [0.1, 1, 10, 100, 1000], 'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']}},
      'RidgeCV': {'algoritmo': RidgeCV(), 'param_grid': {'alphas': [0.1, 1, 10, 100, 1000], 'cv': [5, 10, 15]}},
      'BayesianRidge': {'algoritmo': BayesianRidge(), 'param_grid': {'n_iter': [100, 500, 1000], 'tol': [0.0001, 0.001, 0.01]}},
      'Ridge': {'algoritmo': Ridge(), 'param_grid': {'alpha': [0.1, 1, 10, 100, 1000]}},
      'LinearRegression': {'algoritmo': LinearRegression(), 'param_grid': {}},
      'TransformedTargetRegressor': {'algoritmo': TransformedTargetRegressor(), 'param_grid': {'regressor': [], 'transformer': []}},
      'LassoCV': {'algoritmo': LassoCV(), 'param_grid': {'eps': [0.001, 0.01, 0.1], 'n_alphas': [100, 500, 1000], 'cv': [5, 10, 15]}},
      'ElasticNetCV': {'algoritmo': ElasticNetCV(), 'param_grid': {'l1_ratio': [0.1, 0.5, 0.7, 0.9], 'eps': [0.001, 0.01, 0.1], 'n_alphas': [100, 500, 1000], 'cv': [5, 10, 15]}},
      'LassoLarsCV': {'algoritmo': LassoLarsCV(), 'param_grid': {'eps': [0.001, 0.01, 0.1], 'n_alphas': [100, 500, 1000], 'cv': [5, 10, 15]}},
      'LassoLarsIC': {'algoritmo': LassoLarsIC(), 'param_grid': {'criterion': ['aic', 'bic'], 'eps': [0.001, 0.01, 0.1]}},
      'LarsCV': {'algoritmo': LarsCV(), 'param_grid': {'eps': [0.001, 0.01, 0.1], 'n_alphas': [100, 500, 1000], 'cv': [5, 10, 15]}},
      'Lars': {'algoritmo': Lars(), 'param_grid': {'eps': [0.001, 0.01, 0.1], 'n_alphas': [100, 500, 1000], 'normalize': [True, False]}},
      'SGDRegressor': {'algoritmo': SGDRegressor(), 'param_grid': {'loss': ['squared_loss', 'huber', 'epsilon_insensitive'], 'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [0.0001, 0.001, 0.01]}},
      'RANSACRegressor': {'algoritmo': RANSACRegressor(), 'param_grid': {'min_samples': [0.1, 0.5, 0.7], 'residual_threshold': [0.1, 0.5, 0.7]}},
      'ElasticNet': {'algoritmo': ElasticNet(), 'param_grid': {'alpha': [0.1, 0.5, 1], 'l1_ratio': [0.1, 0.5, 0.7, 0.9]}},
      'Lasso': {'algoritmo': Lasso(), 'param_grid': {'alpha': [0.1, 0.5, 1]}},
      'OrthogonalMatchingPursuitCV': {'algoritmo': OrthogonalMatchingPursuitCV(), 'param_grid': {'cv': [5, 10, 15], 'n_nonzero_coefs': [5, 10, 15]}},
      'ExtraTreeRegressor': {'algoritmo': ExtraTreeRegressor(), 'param_grid': {'max_depth': [3, 5, 7, None], 'min_samples_leaf': [1, 2, 5, 10]}},
      'PassiveAggressiveRegressor': {'algoritmo': PassiveAggressiveRegressor(), 'param_grid': {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 0.5]}},
      'GaussianProcessRegressor': {'algoritmo': GaussianProcessRegressor(), 'param_grid': {'alpha': [0.1, 0.5, 1], 'normalize_y': [True, False]}},
      'OrthogonalMatchingPursuit': {'algoritmo': OrthogonalMatchingPursuit(), 'param_grid': {'n_nonzero_coefs': [5, 10, 15]}},
      'DecisionTreeRegressor': {'algoritmo': DecisionTreeRegressor(), 'param_grid': {'max_depth': [3, 5, 7, None], 'min_samples_leaf': [1, 2, 5, 10]}},
      'DummyRegressor': {'algoritmo': DummyRegressor(), 'param_grid': {'strategy': ['mean', 'median']}},
      'LassoLars': {'algoritmo': LassoLars(), 'param_grid': {'alpha': [0.1, 0.5, 1], 'eps': [0.001, 0.01, 0.1], 'normalize': [True, False]}},
      'KernelRidge': {'algoritmo': KernelRidge(), 'param_grid': {'alpha': [0.1, 0.5, 1], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [2, 3, 4]}}
  }

  if st.checkbox('SVR', value=False):
    dict_algoritmos['SVR'] = {'algoritmo': SVR(), 'param_grid': {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear', 'poly', 'rbf']}}
  else:
    dict_algoritmos.pop('SVR')
  if st.checkbox('RandomForestRegressor', value=False):
    dict_algoritmos['RandomForestRegressor'] = {'algoritmo': RandomForestRegressor(), 'param_grid': {'n_estimators': [10, 50, 100, 200], 'max_depth': [3, 5, 7, None]}}
  else:
    dict_algoritmos.pop('RandomForestRegressor')
  if st.checkbox('ExtraTreesRegressor', value=False):
    dict_algoritmos['ExtraTreesRegressor'] = {'algoritmo': ExtraTreesRegressor(), 'param_grid': {'n_estimators': [10, 50, 100, 200], 'max_depth': [3, 5, 7, None]}}
  else:
    dict_algoritmos.pop('ExtraTreesRegressor')
  if st.checkbox('AdaBoostRegressor', value=False):
    dict_algoritmos['AdaBoostRegressor'] = {'algoritmo': AdaBoostRegressor(), 'param_grid': {'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.1, 0.5, 1]}}
  else:
    dict_algoritmos.pop('AdaBoostRegressor')
  if st.checkbox('NuSVR', value=False):
    dict_algoritmos['NuSVR'] = {'algoritmo': NuSVR(), 'param_grid': {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear', 'poly', 'rbf']}}
  else:
    dict_algoritmos.pop('NuSVR')
  if st.checkbox('GradientBoostingRegressor', value=False):
    dict_algoritmos['GradientBoostingRegressor'] = {'algoritmo': GradientBoostingRegressor(), 'param_grid': {'learning_rate': [0.1, 0.5, 1], 'n_estimators': [10, 50, 100, 200]}}
  else:
    dict_algoritmos.pop('GradientBoostingRegressor')
  if st.checkbox('KNeighborsRegressor', value=False):
    dict_algoritmos['KNeighborsRegressor'] = {'algoritmo': KNeighborsRegressor(), 'param_grid': {'n_neighbors': [3, 5, 7, 9]}}
  else:
    dict_algoritmos.pop('KNeighborsRegressor')
  if st.checkbox('HistGradientBoostingRegressor', value=False):
    dict_algoritmos['HistGradientBoostingRegressor'] = {'algoritmo': HistGradientBoostingRegressor(), 'param_grid': {'learning_rate': [0.1, 0.5, 1], 'max_depth': [3, 5, 7, None], 'max_iter': [10, 50, 100, 200]}}
  else:
    dict_algoritmos.pop('HistGradientBoostingRegressor')
  if st.checkbox('BaggingRegressor', value=False):
    dict_algoritmos['BaggingRegressor'] = {'algoritmo': BaggingRegressor(), 'param_grid': {'n_estimators': [10, 50, 100, 200], 'max_samples': [0.5, 1, 2]}}
  else:
    dict_algoritmos.pop('BaggingRegressor')
  if st.checkbox('MLPRegressor', value=False):
    dict_algoritmos['MLPRegressor'] = {'algoritmo': MLPRegressor(), 'param_grid': {'hidden_layer_sizes': [(10,), (50,), (100,), (10,10,), (50,50,), (100,100,)], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}}
  else:
    dict_algoritmos.pop('MLPRegressor')
  if st.checkbox('HuberRegressor', value=False):
    dict_algoritmos['HuberRegressor'] = {'algoritmo': HuberRegressor(), 'param_grid': {'epsilon': [0.1, 1, 10], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}}
  else:
    dict_algoritmos.pop('HuberRegressor')
  if st.checkbox('LinearSVR', value=False):
    dict_algoritmos['LinearSVR'] = {'algoritmo': LinearSVR(), 'param_grid': {'C': [0.1, 1, 10, 100, 1000], 'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']}}
  else:
    dict_algoritmos.pop('LinearSVR')
  if st.checkbox('RidgeCV', value=False):
    dict_algoritmos['RidgeCV'] = {'algoritmo': RidgeCV(), 'param_grid': {'alphas': [0.1, 1, 10, 100, 1000], 'cv': [5, 10, 15]}}
  else:
    dict_algoritmos.pop('RidgeCV')
  if st.checkbox('BayesianRidge', value=False):
    dict_algoritmos['BayesianRidge'] = {'algoritmo': BayesianRidge(), 'param_grid': {'n_iter': [100, 500, 1000], 'tol': [0.0001, 0.001, 0.01]}}
  else:
    dict_algoritmos.pop('BayesianRidge')
  if st.checkbox('Ridge', value=False):
    dict_algoritmos['Ridge'] = {'algoritmo': Ridge(), 'param_grid': {'alpha': [0.1, 1, 10, 100, 1000]}}
  else:
    dict_algoritmos.pop('Ridge')
  if st.checkbox('LinearRegression', value=False):
    dict_algoritmos['LinearRegression'] = {'algoritmo': LinearRegression(), 'param_grid': {}}
  else:
    dict_algoritmos.pop('LinearRegression')
  if st.checkbox('TransformedTargetRegressor', value=False):
    dict_algoritmos['TransformedTargetRegressor'] = {'algoritmo': TransformedTargetRegressor(), 'param_grid': {'regressor': [], 'transformer': []}}
  else:
    dict_algoritmos.pop('TransformedTargetRegressor')
  if st.checkbox('LassoCV', value=False):
    dict_algoritmos['LassoCV'] = {'algoritmo': LassoCV(), 'param_grid': {'eps': [0.001, 0.01, 0.1], 'n_alphas': [100, 500, 1000], 'cv': [5, 10, 15]}}
  else:
    dict_algoritmos.pop('LassoCV')
  if st.checkbox('ElasticNetCV', value=False):
    dict_algoritmos['ElasticNetCV'] = {'algoritmo': ElasticNetCV(), 'param_grid': {'l1_ratio': [0.1, 0.5, 0.7, 0.9], 'eps': [0.001, 0.01, 0.1], 'n_alphas': [100, 500, 1000], 'cv': [5, 10, 15]}}
  else:
    dict_algoritmos.pop('ElasticNetCV')
  if st.checkbox('LassoLarsCV', value=False):
    dict_algoritmos['LassoLarsCV'] = {'algoritmo': LassoLarsCV(), 'param_grid': {'eps': [0.001, 0.01, 0.1], 'n_alphas': [100, 500, 1000], 'cv': [5, 10, 15]}}
  else:
    dict_algoritmos.pop('LassoLarsCV')
  if st.checkbox('LassoLarsIC', value=False):
    dict_algoritmos['LassoLarsIC'] = {'algoritmo': LassoLarsIC(), 'param_grid': {'criterion': ['aic', 'bic'], 'eps': [0.001, 0.01, 0.1]}}
  else:
    dict_algoritmos.pop('LassoLarsIC')
  if st.checkbox('LarsCV', value=False):
    dict_algoritmos['LarsCV'] = {'algoritmo': LarsCV(), 'param_grid': {'eps': [0.001, 0.01, 0.1], 'n_alphas': [100, 500, 1000], 'cv': [5, 10, 15]}}
  else:
    dict_algoritmos.pop('LarsCV')
  if st.checkbox('Lars', value=False):
    dict_algoritmos['Lars'] = {'algoritmo': Lars(), 'param_grid': {'eps': [0.001, 0.01, 0.1], 'n_alphas': [100, 500, 1000], 'normalize': [True, False]}}
  else:
    dict_algoritmos.pop('Lars')
  if st.checkbox('SGDRegressor', value=False):
    dict_algoritmos['SGDRegressor'] = {'algoritmo': SGDRegressor(), 'param_grid': {'loss': ['squared_loss', 'huber', 'epsilon_insensitive'], 'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [0.0001, 0.001, 0.01]}}
  else:
    dict_algoritmos.pop('SGDRegressor')
  if st.checkbox('RANSACRegressor', value=False):
    dict_algoritmos['RANSACRegressor'] = {'algoritmo': RANSACRegressor(), 'param_grid': {'min_samples': [0.1, 0.5, 0.7], 'residual_threshold': [0.1, 0.5, 0.7]}}
  else:
    dict_algoritmos.pop('RANSACRegressor')
  if st.checkbox('ElasticNet', value=False):
    dict_algoritmos['ElasticNet'] = {'algoritmo': ElasticNet(), 'param_grid': {'alpha': [0.1, 0.5, 1], 'l1_ratio': [0.1, 0.5, 0.7, 0.9]}}
  else:
    dict_algoritmos.pop('ElasticNet')
  if st.checkbox('Lasso', value=False):
    dict_algoritmos['Lasso'] = {'algoritmo': Lasso(), 'param_grid': {'alpha': [0.1, 0.5, 1]}}
  else:
    dict_algoritmos.pop('Lasso')
  if st.checkbox('OrthogonalMatchingPursuitCV', value=False):
    dict_algoritmos['OrthogonalMatchingPursuitCV'] = {'algoritmo': OrthogonalMatchingPursuitCV(), 'param_grid': {'cv': [5, 10, 15], 'n_nonzero_coefs': [5, 10, 15]}}
  else:
    dict_algoritmos.pop('OrthogonalMatchingPursuitCV')
  if st.checkbox('ExtraTreeRegressor', value=False):
    dict_algoritmos['ExtraTreeRegressor'] = {'algoritmo': ExtraTreeRegressor(), 'param_grid': {'max_depth': [3, 5, 7, None], 'min_samples_leaf': [1, 2, 5, 10]}}
  else:
    dict_algoritmos.pop('ExtraTreeRegressor')
  if st.checkbox('PassiveAggressiveRegressor', value=False):
    dict_algoritmos['PassiveAggressiveRegressor'] = {'algoritmo': PassiveAggressiveRegressor(), 'param_grid': {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 0.5]}}
  else:
    dict_algoritmos.pop('PassiveAggressiveRegressor')
  if st.checkbox('GaussianProcessRegressor', value=False):
    dict_algoritmos['GaussianProcessRegressor'] = {'algoritmo': GaussianProcessRegressor(), 'param_grid': {'alpha': [0.1, 0.5, 1], 'normalize_y': [True, False]}}
  else:
    dict_algoritmos.pop('GaussianProcessRegressor')
  if st.checkbox('OrthogonalMatchingPursuit', value=False):
    dict_algoritmos['OrthogonalMatchingPursuit'] = {'algoritmo': OrthogonalMatchingPursuit(), 'param_grid': {'n_nonzero_coefs': [5, 10, 15]}}
  else:
    dict_algoritmos.pop('OrthogonalMatchingPursuit')
  if st.checkbox('DecisionTreeRegressor', value=False):
    dict_algoritmos['DecisionTreeRegressor'] = {'algoritmo': DecisionTreeRegressor(), 'param_grid': {'max_depth': [3, 5, 7, None], 'min_samples_leaf': [1, 2, 5, 10]}}
  else:
    dict_algoritmos.pop('DecisionTreeRegressor')
  if st.checkbox('DummyRegressor', value=False):
    dict_algoritmos['DummyRegressor'] = {'algoritmo': DummyRegressor(), 'param_grid': {'strategy': ['mean', 'median']}}
  else:
    dict_algoritmos.pop('DummyRegressor')
  if st.checkbox('LassoLars', value=False):
    dict_algoritmos['LassoLars'] = {'algoritmo': LassoLars(), 'param_grid': {'alpha': [0.1, 0.5, 1], 'eps': [0.001, 0.01, 0.1], 'normalize': [True, False]}}
  else:
    dict_algoritmos.pop('LassoLars')
  if st.checkbox('KernelRidge', value=False):
    dict_algoritmos['KernelRidge'] = {'algoritmo': KernelRidge(), 'param_grid': {'alpha': [0.1, 0.5, 1], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [2, 3, 4]}}
  else:
    dict_algoritmos.pop('KernelRidge')

elif (tarefa == "Classificação"):

  # Crie um dicionário de classificadores para testar
  dict_algoritmos = {
      "AdaBoostClassifier": {
          "algoritmo": AdaBoostClassifier(),
          "param_grid": {
              "n_estimators": [10, 50, 100, 200],
              "learning_rate": [0.1, 0.5, 1],
          },
      },
      "BaggingClassifier": {
          "algoritmo": BaggingClassifier(),
          "param_grid": {"n_estimators": [10, 50, 100, 200]},
      },
      "BernoulliNB": {
          "algoritmo": BernoulliNB(),
          "param_grid": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1]},
      },
      "CalibratedClassifierCV": {
          "algoritmo": CalibratedClassifierCV(base_estimator=LogisticRegression()),
          "param_grid": {"base_estimator__C": [0.1, 1, 10, 100, 1000]},
      },
      "DecisionTreeClassifier": {
          "algoritmo": DecisionTreeClassifier(),
          "param_grid": {
              "max_depth": [None, 5, 10],
              "min_samples_split": [2, 5, 10],
              "min_samples_leaf": [1, 2, 4],
          },
      },
      "DummyClassifier": {
          "algoritmo": DummyClassifier(),
          "param_grid": {
              "strategy": ["most_frequent", "prior", "stratified", "uniform", "constant"]
          },
      },
      "ExtraTreeClassifier": {
          "algoritmo": ExtraTreeClassifier(),
          "param_grid": {
              "max_depth": [None, 5, 10],
              "min_samples_split": [2, 5, 10],
              "min_samples_leaf": [1, 2, 4],
          },
      },
      "ExtraTreesClassifier": {
          "algoritmo": ExtraTreesClassifier(),
          "param_grid": {
              "n_estimators": [10, 50, 100, 200],
              "max_depth": [3, 5, 7, None],
          },
      },
      "GaussianNB": {
          "algoritmo": GaussianNB(),
          "param_grid": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
      },
      "GradientBoostingClassifier": {
          "algoritmo": GradientBoostingClassifier(),
          "param_grid": {
              "learning_rate": [0.1, 0.5, 1],
              "n_estimators": [10, 50, 100, 200],
          },
      },
      "HistGradientBoostingClassifier": {
          "algoritmo": HistGradientBoostingClassifier(),
          "param_grid": {
              "learning_rate": [0.1, 0.5, 1],
              "max_depth": [3, 5, 7, None],
              "max_iter": [10, 50, 100, 200],
          },
      },
      "KNeighborsClassifier": {
          "algoritmo": KNeighborsClassifier(),
          "param_grid": {
              "n_neighbors": [3, 5, 7, 9, 11],
              "weights": ["uniform", "distance"],
          },
      },
      "LabelPropagation": {
          "algoritmo": LabelPropagation(),
          "param_grid": {"kernel": ["knn", "rbf"]},
      },
      "LabelSpreading": {
          "algoritmo": LabelSpreading(),
          "param_grid": {"kernel": ["knn", "rbf"]},
      },
      "LinearDiscriminantAnalysis": {
          "algoritmo": LinearDiscriminantAnalysis(),
          "param_grid": {},
      },
      "LinearSVC": {
          "algoritmo": LinearSVC(),
          "param_grid": {"C": [0.1, 1, 10, 100, 1000]},
      },
      "LogisticRegression": {
          "algoritmo": LogisticRegression(),
          "param_grid": {"C": [0.1, 1, 10, 100, 1000]},
      },
      "LogisticRegressionCV": {"algoritmo": LogisticRegressionCV(), "param_grid": {}},
      "MLPClassifier": {
          "algoritmo": MLPClassifier(),
          "param_grid": {
              "hidden_layer_sizes": [
                  (10,),
                  (50,),
                  (100,),
                  (
                      10,
                      10,
                  ),
                  (
                      50,
                      50,
                  ),
                  (
                      100,
                      100,
                  ),
              ]
          },
      },
      "NearestCentroid": {"algoritmo": NearestCentroid(), "param_grid": {}},
      "PassiveAggressiveClassifier": {
          "algoritmo": PassiveAggressiveClassifier(),
          "param_grid": {"C": [0.1, 1, 10, 100, 1000]},
      },
      "Perceptron": {
          "algoritmo": Perceptron(),
          "param_grid": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1]},
      },
      "QuadraticDiscriminantAnalysis": {
          "algoritmo": QuadraticDiscriminantAnalysis(),
          "param_grid": {},
      },
      "RandomForestClassifier": {
          "algoritmo": RandomForestClassifier(),
          "param_grid": {
              "n_estimators": [10, 50, 100, 200],
              "max_depth": [3, 5, 7, None],
          },
      },
      "RidgeClassifier": {
          "algoritmo": RidgeClassifier(),
          "param_grid": {"alpha": [0.1, 1, 10, 100, 1000]},
      },
      "RidgeClassifierCV": {"algoritmo": RidgeClassifierCV(), "param_grid": {}},
      "SGDClassifier": {
          "algoritmo": SGDClassifier(),
          "param_grid": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1]},
      },
      "SVC": {
          "algoritmo": SVC(class_weight="balanced"),
          "param_grid": {"C": [0.1, 1, 10, 100, 1000], "kernel": ["linear", "rbf"]},
      },
  }

  if st.checkbox('AdaBoostClassifier', value=False):
    dict_algoritmos['AdaBoostClassifier'] = {'algoritmo': AdaBoostClassifier(), 'param_grid': {'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.1, 0.5, 1]}}
  else:
    dict_algoritmos.pop('AdaBoostClassifier')
  if st.checkbox('BaggingClassifier', value=False):
    dict_algoritmos['BaggingClassifier'] = {'algoritmo': BaggingClassifier(), 'param_grid': {'n_estimators': [10, 50, 100, 200]}}
  else:
    dict_algoritmos.pop('BaggingClassifier')
  if st.checkbox('BernoulliNB', value=False):
    dict_algoritmos['BernoulliNB'] = {'algoritmo': BernoulliNB(), 'param_grid': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}}
  else:
    dict_algoritmos.pop('BernoulliNB')
  if st.checkbox('CalibratedClassifierCV', value=False):
    dict_algoritmos['CalibratedClassifierCV'] = {'algoritmo': CalibratedClassifierCV(base_estimator=LogisticRegression()), 'param_grid': {'base_estimator__C': [0.1, 1, 10, 100, 1000]}}
  else:
    dict_algoritmos.pop('CalibratedClassifierCV')
  if st.checkbox('DecisionTreeClassifier', value=False):
    dict_algoritmos['DecisionTreeClassifier'] = {'algoritmo': DecisionTreeClassifier(), 'param_grid': {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}}
  else:
    dict_algoritmos.pop('DecisionTreeClassifier')
  if st.checkbox('DummyClassifier', value=False):
    dict_algoritmos['DummyClassifier'] = {'algoritmo': DummyClassifier(), 'param_grid': {'strategy': ['most_frequent', 'prior', 'stratified', 'uniform', 'constant']}}
  else:
    dict_algoritmos.pop('DummyClassifier')
  if st.checkbox('ExtraTreeClassifier', value=False):
    dict_algoritmos['ExtraTreeClassifier'] = {'algoritmo': ExtraTreeClassifier(), 'param_grid': {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}}
  else:
    dict_algoritmos.pop('ExtraTreeClassifier')
  if st.checkbox('ExtraTreesClassifier', value=False):
    dict_algoritmos['ExtraTreesClassifier'] = {'algoritmo': ExtraTreesClassifier(), 'param_grid': {'n_estimators': [10, 50, 100, 200], 'max_depth': [3, 5, 7, None]}}
  else:
    dict_algoritmos.pop('ExtraTreesClassifier')
  if st.checkbox('GaussianNB', value=False):
    dict_algoritmos['GaussianNB'] = {'algoritmo': GaussianNB(), 'param_grid': {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}}
  else:
    dict_algoritmos.pop('GaussianNB')
  if st.checkbox('GradientBoostingClassifier', value=False):
    dict_algoritmos['GradientBoostingClassifier'] = {'algoritmo': GradientBoostingClassifier(), 'param_grid': {'learning_rate': [0.1, 0.5, 1], 'n_estimators': [10, 50, 100, 200]}}
  else:
    dict_algoritmos.pop('GradientBoostingClassifier')
  if st.checkbox('HistGradientBoostingClassifier', value=False):
    dict_algoritmos['HistGradientBoostingClassifier'] = {'algoritmo': HistGradientBoostingClassifier(), 'param_grid': {'learning_rate': [0.1, 0.5, 1], 'max_depth': [3, 5, 7, None], 'max_iter': [10, 50, 100, 200]}}
  else:
    dict_algoritmos.pop('HistGradientBoostingClassifier')
  if st.checkbox('KNeighborsClassifier', value=False):
    dict_algoritmos['KNeighborsClassifier'] = {'algoritmo': KNeighborsClassifier(), 'param_grid': {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}}
  else:
    dict_algoritmos.pop('KNeighborsClassifier')
  if st.checkbox('LabelPropagation', value=False):
    dict_algoritmos['LabelPropagation'] = {'algoritmo': LabelPropagation(), 'param_grid': {'kernel': ['knn', 'rbf']}}
  else:
    dict_algoritmos.pop('LabelPropagation')
  if st.checkbox('LabelSpreading', value=False):
    dict_algoritmos['LabelSpreading'] = {'algoritmo': LabelSpreading(), 'param_grid': {'kernel': ['knn', 'rbf']}}
  else:
    dict_algoritmos.pop('LabelSpreading')
  if st.checkbox('LinearDiscriminantAnalysis', value=False):
    dict_algoritmos['LinearDiscriminantAnalysis'] = {'algoritmo': LinearDiscriminantAnalysis(), 'param_grid': {}}
  else:
    dict_algoritmos.pop('LinearDiscriminantAnalysis')
  if st.checkbox('LinearSVC', value=False):
    dict_algoritmos['LinearSVC'] = {'algoritmo': LinearSVC(), 'param_grid': {'C': [0.1, 1, 10, 100, 1000]}}
  else:
    dict_algoritmos.pop('LinearSVC')
  if st.checkbox('LogisticRegression', value=False):
    dict_algoritmos['LogisticRegression'] = {'algoritmo': LogisticRegression(), 'param_grid': {'C': [0.1, 1, 10, 100, 1000]}}
  else:
    dict_algoritmos.pop('LogisticRegression')
  if st.checkbox('LogisticRegressionCV', value=False):
    dict_algoritmos['LogisticRegressionCV'] = {'algoritmo': LogisticRegressionCV(), 'param_grid': {}}
  else:
    dict_algoritmos.pop('LogisticRegressionCV')
  if st.checkbox('MLPClassifier', value=False):
    dict_algoritmos['MLPClassifier'] = {'algoritmo': MLPClassifier(), 'param_grid': {'hidden_layer_sizes': [(10,), (50,), (100,), (10,10,), (50,50,), (100,100,)]}}
  else:
    dict_algoritmos.pop('MLPClassifier')
  if st.checkbox('NearestCentroid', value=False):
    dict_algoritmos['NearestCentroid'] = {'algoritmo': NearestCentroid(), 'param_grid': {}}
  else:
    dict_algoritmos.pop('NearestCentroid')
  if st.checkbox('PassiveAggressiveClassifier', value=False):
    dict_algoritmos['PassiveAggressiveClassifier'] = {'algoritmo': PassiveAggressiveClassifier(), 'param_grid': {'C': [0.1, 1, 10, 100, 1000]}}
  else:
    dict_algoritmos.pop('PassiveAggressiveClassifier')
  if st.checkbox('Perceptron', value=False):
    dict_algoritmos['Perceptron'] = {'algoritmo': Perceptron(), 'param_grid': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}}
  else:
    dict_algoritmos.pop('Perceptron')
  if st.checkbox('QuadraticDiscriminantAnalysis', value=False):
    dict_algoritmos['QuadraticDiscriminantAnalysis'] = {'algoritmo': QuadraticDiscriminantAnalysis(), 'param_grid': {}}
  else:
    dict_algoritmos.pop('QuadraticDiscriminantAnalysis')
  if st.checkbox('RandomForestClassifier', value=False):
    dict_algoritmos['RandomForestClassifier'] = {'algoritmo': RandomForestClassifier(), 'param_grid': {'n_estimators': [10, 50, 100, 200], 'max_depth': [3, 5, 7, None]}}
  else:
    dict_algoritmos.pop('RandomForestClassifier')
  if st.checkbox('RidgeClassifier', value=False):
    dict_algoritmos['RidgeClassifier'] = {'algoritmo': RidgeClassifier(), 'param_grid': {'alpha': [0.1, 1, 10, 100, 1000]}}
  else:
    dict_algoritmos.pop('RidgeClassifier')
  if st.checkbox('RidgeClassifierCV', value=False):
    dict_algoritmos['RidgeClassifierCV'] = {'algoritmo': RidgeClassifierCV(), 'param_grid': {}}
  else:
    dict_algoritmos.pop('RidgeClassifierCV')
  if st.checkbox('SGDClassifier', value=False):
    dict_algoritmos['SGDClassifier'] = {'algoritmo': SGDClassifier(), 'param_grid': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}}
  else:
    dict_algoritmos.pop('SGDClassifier')
  if st.checkbox('SVC', value=False):
    dict_algoritmos['SVC'] = {'algoritmo': SVC(class_weight="balanced"), 'param_grid': {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear', 'rbf']}}
  else:
    dict_algoritmos.pop('SVC')

elif (tarefa == "Classificação Ensemble - BaggingClassifier"):

  # Hiperparâmetros
  n_estimators = st.slider("Número de estimadores", 10, 200, 50)
  max_samples = st.slider("Número máximo de amostras", 0.1, 1.0, (0.1, 0.5))
  max_features = st.slider("Número máximo de características", 0.1, 1.0, (0.1, 0.5))
  bootstrap = st.selectbox("Amostragem bootstrap", [True, False])
  bootstrap_features = st.selectbox("Características de amostragem bootstrap", [True, False])
  oob_score = st.selectbox("OOB score", [True, False])
  n_jobs = st.selectbox("Número de trabalhos", [-1, 1, 2])

  # Crie um dicionário de classificadores para testar
  dict_algoritmos = {
      "BaggingClassifier": {
          "algoritmo": BaggingClassifier(),
          "param_grid": {
              "base_estimator": [
                  DecisionTreeClassifier(),
                  LinearSVC(),
                  SGDClassifier(),
                  MLPClassifier(),
                  Perceptron(),
                  LogisticRegression(),
                  LogisticRegressionCV(),
                  SVC(),
                  CalibratedClassifierCV(),
                  PassiveAggressiveClassifier(),
                  LabelPropagation(),
                  LabelSpreading(),
                  RandomForestClassifier(),
                  GradientBoostingClassifier(),
                  QuadraticDiscriminantAnalysis(),
                  HistGradientBoostingClassifier(),
                  RidgeClassifierCV(),
                  RidgeClassifier(),
                  AdaBoostClassifier(),
                  ExtraTreesClassifier(),
                  KNeighborsClassifier(),
                  BernoulliNB(),
                  LinearDiscriminantAnalysis(),
                  GaussianNB(),
                  NuSVC(),
                  DecisionTreeClassifier(),
                  NearestCentroid(),
                  ExtraTreeClassifier(),
                  DummyClassifier(),
              ],
              "n_estimators": [n_estimators],
              "max_samples": np.arange(max_samples[0],max_samples[1]).tolist(),
              "max_features": np.arange(max_features[0],max_features[1]).tolist(),
              "bootstrap": [bootstrap],
              "bootstrap_features": [bootstrap_features],
              "oob_score": [oob_score],
              "n_jobs": [n_jobs]
          },
      }
  }
  st.write(dict_algoritmos)

elif (tarefa == "Classificação Ensemble - AdaBoostClassifier"):

  # Hiperparâmetros
  n_estimators = st.slider("Número de estimadores", 10, 200, 50)
  learning_rate = st.slider("Taxa de aprendizado", 0.1, 1.0, (0.1, 0.5))
  algorithm = st.selectbox("Algoritmo", ["SAMME.R", "SAMME"])

  # Crie um dicionário de classificadores para testar
  dict_algoritmos = {
      "AdaBoostClassifier": {
          "algoritmo": AdaBoostClassifier(),
          "param_grid": {
              "base_estimator": [
                  DecisionTreeClassifier(),
                  LinearSVC(),
                  SGDClassifier(),
                  MLPClassifier(),
                  Perceptron(),
                  LogisticRegression(),
                  LogisticRegressionCV(),
                  SVC(),
                  CalibratedClassifierCV(),
                  PassiveAggressiveClassifier(),
                  LabelPropagation(),
                  LabelSpreading(),
                  RandomForestClassifier(),
                  GradientBoostingClassifier(),
                  QuadraticDiscriminantAnalysis(),
                  HistGradientBoostingClassifier(),
                  RidgeClassifierCV(),
                  RidgeClassifier(),
                  KNeighborsClassifier(),
                  BernoulliNB(),
                  LinearDiscriminantAnalysis(),
                  GaussianNB(),
                  NuSVC(),
                  DecisionTreeClassifier(),
                  NearestCentroid(),
                  ExtraTreeClassifier(),
                  DummyClassifier(),
              ],
              "n_estimators": [n_estimators],
              "learning_rate": np.arange(learning_rate[0],learning_rate[1]).tolist(),
              "algorithm": [algorithm],
          },
      }
  }

  st.write(dict_algoritmos)

elif (tarefa == "Classificação Ensemble - VotingClassifier 1"):

  # Hiperparâmetros
  estimators = st.multiselect("Estimadores", ["DecisionTreeClassifier", "SVC", "RandomForestClassifier"])
  voting = st.selectbox("Método de votação", ["hard", "soft"])
  n_estimators = st.slider("Número de estimadores", 10, 200, 50)

  # Inicializando os estimadores
  estimators_list = []
  for estimator in estimators:
    if estimator == "DecisionTreeClassifier":
      from sklearn.tree import DecisionTreeClassifier
      estimators_list.append(("dt", DecisionTreeClassifier(random_state=42)))
    elif estimator == "SVC":
      from sklearn.svm import SVC
      estimators_list.append(("svc", SVC(random_state=42)))
    else:
      from sklearn.ensemble import RandomForestClassifier
      estimators_list.append(("rf", RandomForestClassifier(random_state=42)))

  # Crie um dicionário de classificadores para testar
  dict_algoritmos = {
    "VotingClassifier": {
      "algoritmo": VotingClassifier(
        estimators = [estimators_list]
      ),
      'param_grid': {
        'voting': [voting],
        'n_estimators': n_estimators
      },
    }
  }
  st.write(dict_algoritmos)

elif (tarefa == "Classificação Ensemble - VotingClassifier 2"):

  # Hiperparâmetros
  voting = st.selectbox("Voting", ["hard", "soft"])
  weights = st.slider("Pesos", 0.0, 1.0, 0.5)

  # Crie um dicionário de classificadores para testar
  dict_algoritmos = {
      "VotingClassifier": {
          "algoritmo": VotingClassifier(
              estimators=[
                  ("dt", DecisionTreeClassifier()),
                  ("svc", LinearSVC()),
                  ("sgd", SGDClassifier()),
                  ("mlp", MLPClassifier()),
                  ("per", Perceptron()),
                  ("lr", LogisticRegression()),
                  ("lrcv", LogisticRegressionCV()),
                  ("svc2", SVC()),
                  ("cc", CalibratedClassifierCV()),
                  ("pa", PassiveAggressiveClassifier()),
                  ("lp", LabelPropagation()),
                  ("ls", LabelSpreading()),
                  ("rf", RandomForestClassifier()),
                  ("gb", GradientBoostingClassifier()),
                  ("qda", QuadraticDiscriminantAnalysis()),
                  ("hgb", HistGradientBoostingClassifier()),
                  ("rcv", RidgeClassifierCV()),
                  ("rc", RidgeClassifier()),
                  ("knn", KNeighborsClassifier()),
                  ("bnb", BernoulliNB()),
                  ("lda", LinearDiscriminantAnalysis()),
                  ("gnb", GaussianNB()),
                  ("nusvc", NuSVC()),
                  ("dtc", DecisionTreeClassifier()),
                  ("nc", NearestCentroid()),
                  ("etc", ExtraTreeClassifier()),
                  ("dc", DummyClassifier()),
              ]
          ),
          "param_grid": {"weights": [[weights]], "voting": [voting]},
      }
  }
  st.write(dict_algoritmos)

elif (tarefa == "Classificação Ensemble - StackingClassifier"):

  # Hiperparâmetros
  final_estimator = st.selectbox("Estimador final", ["lr", "dt", "svc"])
  cv = st.slider("Número de folds para validação cruzada", 2, 10, 5)
  passthrough = st.selectbox("Passar através", [True, False])

  if final_estimator == "lr":
    final_estimator = LogisticRegression()
  elif final_estimator == "dt":
    final_estimator = DecisionTreeClassifier()
  else:
    final_estimator = SVC()

  # Crie um dicionário de classificadores para testar
  dict_algoritmos = {
      "StackingClassifier": {
          "algoritmo": StackingClassifier(
              estimators=[
                  ("dt", DecisionTreeClassifier()),
                  ("svc", LinearSVC()),
                  ("sgd", SGDClassifier()),
                  ("mlp", MLPClassifier()),
                  ("per", Perceptron()),
                  ("lr", LogisticRegression()),
                  ("lrcv", LogisticRegressionCV()),
                  ("svc2", SVC()),
                  ("cc", CalibratedClassifierCV()),
                  ("pa", PassiveAggressiveClassifier()),
                  ("lp", LabelPropagation()),
                  ("ls", LabelSpreading()),
                  ("rf", RandomForestClassifier()),
                  ("gb", GradientBoostingClassifier()),
                  ("qda", QuadraticDiscriminantAnalysis()),
                  ("hgb", HistGradientBoostingClassifier()),
                  ("rcv", RidgeClassifierCV()),
                  ("rc", RidgeClassifier()),
                  ("knn", KNeighborsClassifier()),
                  ("bnb", BernoulliNB()),
                  ("lda", LinearDiscriminantAnalysis()),
                  ("gnb", GaussianNB()),
                  ("nusvc", NuSVC()),
                  ("dtc", DecisionTreeClassifier()),
                  ("nc", NearestCentroid()),
                  ("etc", ExtraTreeClassifier()),
                  ("dc", DummyClassifier()),
              ]
          ),
          "param_grid": {
              "stack_method": ["auto", "predict_proba"],
              "final_estimator": [final_estimator],
              "cv": [cv],
              "passthrough": [passthrough],
          },
      }
  }

  st.write(dict_algoritmos)

elif (tarefa == "Classificação Ensemble - ExtraTreesClassifier"):

  # Hiperparâmetros
  n_estimators = st.slider("Número de estimadores", 10, 200, 50)
  criterion = st.selectbox("Critério", ["gini", "entropy"], key = "criterion")
  max_depth = st.slider("Profundidade máxima", 2, 20, (5, 10))
  min_samples_split = st.slider("Número mínimo de amostras para dividir um nó", 2, 20, (2, 10))
  min_samples_leaf = st.slider("Número mínimo de amostras em uma folha", 1, 10, (1, 4))

  # Crie um dicionário de classificadores para testar
  dict_algoritmos = {
      "ExtraTreesClassifier": {
          "algoritmo": ExtraTreesClassifier(),
          "param_grid": {
              "n_estimators": [n_estimators],
              "criterion": ["gini", "entropy"],
              "max_depth": np.arange(max_depth[0],max_depth[1]).tolist(),
              "min_samples_split": np.arange(min_samples_split[0],min_samples_split[1]).tolist(),
              "min_samples_leaf": np.arange(min_samples_leaf[0],min_samples_leaf[1]).tolist(),
              "max_features": ["auto", "sqrt", "log2"],
          },
      }
  }
  st.write(dict_algoritmos)

elif (tarefa == "Classificação Ensemble - GradientBoostingClassifier"):

  # Hiperparâmetros
  n_estimators = st.slider("Número de estimadores", 10, 200, 50)
  max_depth = st.slider("Profundidade máxima", 2, 20, (5, 10))
  min_samples_split = st.slider("Número mínimo de amostras para dividir um nó", 2, 20, (2, 10))
  min_samples_leaf = st.slider("Número mínimo de amostras em uma folha", 1, 10, (1, 4))
  learning_rate = st.slider("Taxa de aprendizado", 0.1, 1.0, (0.1, 0.5))
  loss = st.selectbox("Função de perda", ["deviance", "exponential"])

  # Crie um dicionário de classificadores para testar
  dict_algoritmos = {
      "GradientBoostingClassifier": {
          "algoritmo": GradientBoostingClassifier(),
          "param_grid": {
              "n_estimators": [n_estimators],
              "max_depth": np.arange(max_depth[0],max_depth[1]).tolist(),
              "learning_rate": np.arange(learning_rate[0],learning_rate[1]).tolist(),
              "loss": [loss],
              "subsample": [0.8, 1.0],
              "criterion": ["friedman_mse", "mse", "mae"],
              "min_samples_split": np.arange(min_samples_split[0],min_samples_split[1]).tolist(),
              "min_samples_leaf": np.arange(min_samples_leaf[0],min_samples_leaf[1]).tolist(),
              "max_features": ["auto", "sqrt", "log2"],
          },
      }
  }
  st.write(dict_algoritmos)

# ------------------------------------------------------------------------------

if st.button('Calcular', key=f"calcular_button_1"):

  # Crie uma lista de algoritmos para testar
  algoritmos = []
  for name, algoritmo_dict in dict_algoritmos.items():
    alg = algoritmo_dict['algoritmo']
    algoritmos.append(alg)

  # ------------------------------------------------------------------------------

  models = {}

  # ------------------------------------------------------------------------------

  st.subheader("ENCONTRANDO A MELHOR CONFIGURAÇÃO DE HIPERPARÂMETROS")

  metrica = st.selectbox('Métrica de desempenho buscada', ['Acurácia', 'Precisão', 'Recall'], key = "gridsearch_metrica")

  scoring = {'Acurácia':'accuracy', 'Precisão': 'precision', 'Recall':'recall'}

  # Definindo arrays para armazenar as métricas de desempenho de cada modelo treinado e avaliado
  perf_scores = []

  # Cria lista com os dados de treinamento com índice -1 e dados de validação com índice 0
  # Concatena os dados de treino e validação com as partições pré-definidas
  split_index = [-1]*len(X_train) + [0]*len(X_valid)
  X_gridSearch = np.concatenate((X_train, X_valid), axis=0)
  y_gridSearch = np.concatenate((y_train, y_valid), axis=0)
  pds = PredefinedSplit(test_fold = split_index)

  if (tarefa in ['Classificação','Regressão']):
    Y_pred = np.empty([len(algoritmos), len(X_test)], dtype=np.uint8)
  else:
    Y_pred = np.empty([len(algoritmos), len(X)], dtype=np.uint8)

  n = 0

  ada_base_estimator = []
  ada_base_estimators = []
  bag_base_estimator = []
  bag_base_estimators = []
  estimators = []

  # Itere sobre cada classificador e otimize seus hiperparâmetros
  for name, algoritmo_dict in dict_algoritmos.items():

    try:

      alg = algoritmo_dict['algoritmo']
      param_grid = algoritmo_dict['param_grid']

      st.subheader(type(alg).__name__)

      if (tarefa in ['Classificação','Regressão']):
        # Crie o objeto GridSearchCV
        grid_search_cv = GridSearchCV(alg, param_grid, verbose=1, cv=pds, n_jobs=-1, scoring=scoring, refit=metrica, return_train_score=True)
        # Treine o modelo
        grid_search_cv.fit(X_gridSearch, y_gridSearch)

      else:
        # Crie o objeto GridSearchCV
        grid_search_cv = GridSearchCV(alg, param_grid, verbose=1, cv=5, n_jobs=-1, scoring=scoring, refit=metrica, return_train_score=True)
        # Treine o modelo
        grid_search_cv.fit(X, y)

      # Plota resultados do GridSearchCV
      mp.plota_resultado_gridsearchcv(grid_search_cv)

      if (tarefa in ['Classificação','Regressão']):
        for param in param_grid:
          try:
            mp.plota_grafico_parametro_gridsearchcv(grid_search_cv, scoring, param, float)
          except:
            pass

      if (tarefa in ['Classificação','Regressão']):
        y_pred = grid_search_cv.predict(X_test)
      else:
        # Faça previsões com cross-validation
        y_pred = cross_val_predict(grid_search_cv, X, y, cv=5, n_jobs=-1)

      if (tarefa in ['Classificação','Regressão']):
        # Armazene as métricas de desempenho
        perf_scores.append([type(alg).__name__,
                          accuracy_score(y_test, y_pred),
                          recall_score(y_test, y_pred, pos_label=pos_label, average=average),
                          precision_score(y_test, y_pred, pos_label=pos_label, average=average),
                          f1_score(y_test, y_pred, pos_label=pos_label, average=average)
                          ])
      else:
        # Armazene as métricas de desempenho
        perf_scores.append([type(alg).__name__,
                        accuracy_score(y, y_pred),
                        recall_score(y, y_pred, pos_label=pos_label, average=average),
                        precision_score(y, y_pred, pos_label=pos_label, average=average),
                        f1_score(y, y_pred, pos_label=pos_label, average=average)
                        ])


      #if (tarefa in ['Classificação','Regressão']):
        #mp.plota_todos_os_resultados(df[target].nunique(), grid_search_cv.classes_, pos_label, neg_label, y_test, y_pred)
      #else:
        #mp.plota_todos_os_resultados(df[target].nunique(), grid_search_cv.classes_, pos_label, neg_label, y, y_pred)

      models[type(alg).__name__] = grid_search_cv.best_estimator_

      if type(alg).__name__ not in ['AdaBoostClassifier','BaggingClassifier','ExtraTreesClassifier','GradientBoostingClassifier','RandomForestClassifier','VotingClassifier','StackingClassifier','LinearDiscriminantAnalysis','KNeighborsClassifier','LabelPropagation','LabelSpreading','LinearSVC','MLPClassifier','NearestCentroid','PassiveAggressiveClassifier','Perceptron','QuadraticDiscriminantAnalysis','RidgeClassifier','RidgeClassifierCV','SGDClassifier','SVC','GaussianProcessClassifier','NuSVC','RadiusNeighborsClassifier','OneVsOneClassifier','OneVsRestClassifier','OutputCodeClassifier']:
        ada_base_estimator.append(grid_search_cv.best_estimator_)
        ada_base_estimators.append((type(alg).__name__, grid_search_cv.best_estimator_))

      if type(alg).__name__ not in ['AdaBoostClassifier','BaggingClassifier','ExtraTreesClassifier','GradientBoostingClassifier','RandomForestClassifier','VotingClassifier','StackingClassifier','RadiusNeighborsClassifier']:
        bag_base_estimator.append(grid_search_cv.best_estimator_)
        bag_base_estimators.append((type(alg).__name__, grid_search_cv.best_estimator_))

      estimators.append((type(alg).__name__, grid_search_cv.best_estimator_))

      Y_pred[n] = y_pred
      n = n + 1

    except Exception as e:
      st.write(f"Erro: {name} - {algoritmo_dict}")
      st.write(str(e))
      pass

  # ------------------------------------------------------------------------------

  st.subheader("DESEMPENHO DOS MODELOS")

  st.write(models)

  mp.plotar_todas_as_metricas_na_mesma_figura(perf_scores, 'Modelo', drop_duplicates = False)

  # ------------------------------------------------------------------------------

  st.subheader("AGREGANDO AS PREDIÇÕES DE MÚLTIPLOS CLASSIFICADORES")

  # ------------------------------------------------------------------------------

  st.write("Observando a saída gerada para as instâncias de teste")
  st.write(Y_pred[:,:])

  # ------------------------------------------------------------------------------

  from scipy.stats import mode

  # Determina a saída a partir da moda (valor mais frequente)
  y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

  st.write("Observa a classe predita por votação majoritária")
  st.write(y_pred_majority_votes)

  st.write("Número de votos que a classe majoritária recebeu")
  st.write(n_votes)

  # ------------------------------------------------------------------------------

  st.subheader("AVALIA O DESEMPENHO DA CLASSIFICAÇÃO A PARTIR DE UMA VOTAÇÃO MAJORITÁRIA")

  if df[target].nunique() == 2:
    if (tarefa in ['Classificação','Regressão']):
      mp.plota_metricas_duas_classes(y_test, y_pred_majority_votes.reshape([-1]), pos_label)
    else:
      mp.plota_metricas_duas_classes(y, y_pred_majority_votes.reshape([-1]), pos_label)
  else:
    if (tarefa in ['Classificação','Regressão']):
      mp.plota_metricas_multiclasse(y_test, y_pred_majority_votes.reshape([-1]), pos_label)
    else:
      mp.plota_metricas_multiclasse(y, y_pred_majority_votes.reshape([-1]), pos_label)

  # ----------------------------------------------------------------------------

  from sklearn.metrics import classification_report
  if (tarefa in ['Classificação','Regressão']):
    classification_report = classification_report(y_test, y_pred_majority_votes.reshape([-1]), output_dict=True)
  else:
    classification_report = classification_report(y, y_pred_majority_votes.reshape([-1]), output_dict=True)
  df_classification_report = pd.DataFrame(classification_report).transpose()
  st.write(df_classification_report)

  # ------------------------------------------------------------------------------

  if (tarefa in ['Classificação','Regressão']):

    st.subheader("AVALIA O DESEMPENHO DA CLASSIFICAÇÃO COM ENSEMBLE")

    st.write("MELHORES MODELOS")
    st.write(models)

    # Crie um dicionário de classificadores para testar
    dict_algoritmos_ensemble = {}

    adaboost_estimators = []
    bagging_estimators = []

    if bag_base_estimator:
      dict_algoritmos_ensemble['BaggingClassifier'] = {'algoritmo': BaggingClassifier(), 'param_grid': {'base_estimator': bag_base_estimator, 'n_estimators': [10], 'max_samples': [0.1, 0.5, 1.0], 'max_features': [0.1, 0.5, 1.0]}}
    if ada_base_estimator:
      dict_algoritmos_ensemble['AdaBoostClassifier'] = {'algoritmo': AdaBoostClassifier(), 'param_grid': {'base_estimator': ada_base_estimator, 'n_estimators': [10], 'learning_rate':[0.01,0.1,1.0], 'algorithm': ['SAMME', 'SAMME.R']}}
    if estimators:
      dict_algoritmos_ensemble['VotingClassifier'] = {'algoritmo': VotingClassifier(estimators=estimators), 'param_grid': {'voting':['hard','soft'], 'weights':[None, 1]}}
      dict_algoritmos_ensemble['StackingClassifier'] = {'algoritmo': StackingClassifier(estimators=estimators), 'param_grid': {'final_estimator': [LogisticRegression(), DecisionTreeClassifier()], 'stack_method': ['auto', 'predict_proba'], 'cv': [3,5,7]}}

    # ----------------------------------------------------------------------------

    # Crie uma lista de algoritmos para testar
    algoritmos = []
    for name, algoritmo_dict in dict_algoritmos_ensemble.items():
      alg = algoritmo_dict['algoritmo']
      algoritmos.append(alg)

    # ------------------------------------------------------------------------------

    models = {}

    # ------------------------------------------------------------------------------

    st.subheader("ENCONTRANDO A MELHOR CONFIGURAÇÃO DE HIPERPARÂMETROS")

    # Definindo arrays para armazenar as métricas de desempenho de cada modelo treinado e avaliado
    perf_scores = []

    # Cria lista com os dados de treinamento com índice -1 e dados de validação com índice 0
    # Concatena os dados de treino e teste com as partições pré-definidas
    split_index = [-1]*len(X_train) + [0]*len(X_test)
    X_gridSearch = np.concatenate((X_train, X_test), axis=0)
    y_gridSearch = np.concatenate((y_train, y_test), axis=0)
    pds = PredefinedSplit(test_fold = split_index)

    #Y_pred = np.empty([len(algoritmos), len(X)], dtype=np.uint8)
    Y_pred = np.empty([len(algoritmos), len(X_valid)], dtype=np.uint8)
    n = 0

    # Itere sobre cada classificador e otimize seus hiperparâmetros
    for name, algoritmo_dict in dict_algoritmos_ensemble.items():

      try:

        alg = algoritmo_dict['algoritmo']
        param_grid = algoritmo_dict['param_grid']

        st.subheader(type(alg).__name__)

        # Crie o objeto GridSearchCV
        #grid_search_cv = GridSearchCV(alg, param_grid, verbose=1, cv=5, n_jobs=-1, scoring=scoring, refit=metrica, return_train_score=True)
        grid_search_cv = GridSearchCV(alg, param_grid, verbose=1, cv=pds, n_jobs=-1, scoring=scoring, refit=metrica, return_train_score=True)

        # Treine o modelo
        #grid_search_cv.fit(X, y)
        grid_search_cv.fit(X_gridSearch, y_gridSearch)

        # Plota resultados do GridSearchCV
        mp.plota_resultado_gridsearchcv(grid_search_cv)

        # Faça previsões com cross-validation
        #y_pred = cross_val_predict(grid_search_cv, X, y, cv=5, n_jobs=-1)
        y_pred = grid_search_cv.predict(X_valid)

        # Armazene as métricas de desempenho
        # perf_scores.append([type(alg).__name__,
        #                  accuracy_score(y, y_pred),
        #                  recall_score(y, y_pred, pos_label=pos_label, average=average),
        #                  precision_score(y, y_pred, pos_label=pos_label, average=average),
        #                  f1_score(y, y_pred, pos_label=pos_label, average=average)
        #                  ])

        perf_scores.append([type(alg).__name__,
                          accuracy_score(y_valid, y_pred),
                          recall_score(y_valid, y_pred, pos_label=pos_label, average=average),
                          precision_score(y_valid, y_pred, pos_label=pos_label, average=average),
                          f1_score(y_valid, y_pred, pos_label=pos_label, average=average)
                          ])

        #mp.plota_todos_os_resultados(df[target].nunique(), grid_search_cv.classes_, pos_label, neg_label, y, y_pred)
        #mp.plota_todos_os_resultados(df[target].nunique(), grid_search_cv.classes_, pos_label, neg_label, y_valid, y_pred)

        models[type(alg).__name__] = grid_search_cv.best_estimator_

        Y_pred[n] = y_pred
        n = n + 1

      except Exception as e:
        st.write(f"Erro: {name} - {algoritmo_dict}")
        st.write(str(e))
        pass

    # ------------------------------------------------------------------------------

    st.subheader("DESEMPENHO DOS MODELOS")

    st.write("MELHORES MODELOS ENCONTRADOS")
    st.write(models)

    mp.plotar_todas_as_metricas_na_mesma_figura(perf_scores, 'Modelo', drop_duplicates = False)

  # ------------------------------------------------------------------------------
