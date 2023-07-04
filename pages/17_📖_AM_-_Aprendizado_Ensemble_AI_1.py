
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
from sklearn.ensemble import RandomForestClassifier

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

from sklearn.svm import SVR
from sklearn.svm import LinearSVR
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

st.markdown("# Aprendizado Supervisionado - Aprendizado Ensemble - Spot-checking algorithms - Sem otimização de hiperparâmetros")

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

tarefa = st.selectbox('Selecione a tarefa', ['Classificação', 'Classificação Ensemble'], key = "tarefa")

if (tarefa == "Regressão"):

  # Crie um dicionário de regressores para testar
  dict_algoritmos = {
      'SVR': SVR(),
      'RandomForestRegressor': RandomForestRegressor(),
      'ExtraTreesRegressor': ExtraTreesRegressor(),
      'AdaBoostRegressor': AdaBoostRegressor(),
      'NuSVR': NuSVR(),
      'GradientBoostingRegressor': GradientBoostingRegressor(),
      'KNeighborsRegressor': KNeighborsRegressor(),
      'HistGradientBoostingRegressor': HistGradientBoostingRegressor(),
      'BaggingRegressor': BaggingRegressor(),
      'MLPRegressor': MLPRegressor(),
      'HuberRegressor': HuberRegressor(),
      'LinearSVR': LinearSVR(),
      'RidgeCV': RidgeCV(),
      'BayesianRidge': BayesianRidge(),
      'Ridge': Ridge(),
      'LinearRegression': LinearRegression(),
      'TransformedTargetRegressor': TransformedTargetRegressor(),
      'LassoCV': LassoCV(),
      'ElasticNetCV': ElasticNetCV(),
      'LassoLarsCV': LassoLarsCV(),
      'LassoLarsIC': LassoLarsIC(),
      'LarsCV': LarsCV(),
      'Lars': Lars(),
      'SGDRegressor': SGDRegressor(),
      'RANSACRegressor': RANSACRegressor(),
      'ElasticNet': ElasticNet(),
      'Lasso': Lasso(),
      'OrthogonalMatchingPursuitCV': OrthogonalMatchingPursuitCV(),
      'ExtraTreeRegressor': ExtraTreeRegressor(),
      'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
      'GaussianProcessRegressor': GaussianProcessRegressor(),
      'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
      'DecisionTreeRegressor': DecisionTreeRegressor(),
      'DummyRegressor': DummyRegressor(),
      'LassoLars': LassoLars(),
      'KernelRidge': KernelRidge()
  }

  if st.checkbox('SVR', value=True):
    dict_algoritmos['SVR'] = SVR()
  else:
    dict_algoritmos.pop('SVR')
  if st.checkbox('RandomForestRegressor', value=True):
    dict_algoritmos['RandomForestRegressor'] = RandomForestRegressor()
  else:
    dict_algoritmos.pop('RandomForestRegressor')
  if st.checkbox('ExtraTreesRegressor', value=True):
    dict_algoritmos['ExtraTreesRegressor'] = ExtraTreesRegressor()
  else:
    dict_algoritmos.pop('ExtraTreesRegressor')
  if st.checkbox('AdaBoostRegressor', value=True):
    dict_algoritmos['AdaBoostRegressor'] = AdaBoostRegressor()
  else:
    dict_algoritmos.pop('AdaBoostRegressor')
  if st.checkbox('NuSVR', value=True):
    dict_algoritmos['NuSVR'] = NuSVR()
  else:
    dict_algoritmos.pop('NuSVR')
  if st.checkbox('GradientBoostingRegressor', value=True):
    dict_algoritmos['GradientBoostingRegressor'] = GradientBoostingRegressor()
  else:
    dict_algoritmos.pop('GradientBoostingRegressor')
  if st.checkbox('KNeighborsRegressor', value=True):
    dict_algoritmos['KNeighborsRegressor'] = KNeighborsRegressor()
  else:
    dict_algoritmos.pop('KNeighborsRegressor')
  if st.checkbox('HistGradientBoostingRegressor', value=True):
    dict_algoritmos['HistGradientBoostingRegressor'] = HistGradientBoostingRegressor()
  else:
    dict_algoritmos.pop('HistGradientBoostingRegressor')
  if st.checkbox('BaggingRegressor', value=True):
    dict_algoritmos['BaggingRegressor'] = BaggingRegressor()
  else:
    dict_algoritmos.pop('BaggingRegressor')
  if st.checkbox('MLPRegressor', value=True):
    dict_algoritmos['MLPRegressor'] = MLPRegressor()
  else:
    dict_algoritmos.pop('MLPRegressor')
  if st.checkbox('HuberRegressor', value=True):
    dict_algoritmos['HuberRegressor'] = HuberRegressor()
  else:
    dict_algoritmos.pop('HuberRegressor')
  if st.checkbox('LinearSVR', value=True):
    dict_algoritmos['LinearSVR'] = LinearSVR()
  else:
    dict_algoritmos.pop('LinearSVR')
  if st.checkbox('RidgeCV', value=True):
    dict_algoritmos['RidgeCV'] = RidgeCV()
  else:
    dict_algoritmos.pop('RidgeCV')
  if st.checkbox('BayesianRidge', value=True):
    dict_algoritmos['BayesianRidge'] = BayesianRidge()
  else:
    dict_algoritmos.pop('BayesianRidge')
  if st.checkbox('Ridge', value=True):
    dict_algoritmos['Ridge'] = Ridge()
  else:
    dict_algoritmos.pop('Ridge')
  if st.checkbox('LinearRegression', value=True):
    dict_algoritmos['LinearRegression'] = LinearRegression()
  else:
    dict_algoritmos.pop('LinearRegression')
  if st.checkbox('TransformedTargetRegressor', value=True):
    dict_algoritmos['TransformedTargetRegressor'] = TransformedTargetRegressor()
  else:
    dict_algoritmos.pop('TransformedTargetRegressor')
  if st.checkbox('LassoCV', value=True):
    dict_algoritmos['LassoCV'] = LassoCV()
  else:
    dict_algoritmos.pop('LassoCV')
  if st.checkbox('ElasticNetCV', value=True):
    dict_algoritmos['ElasticNetCV'] = ElasticNetCV()
  else:
    dict_algoritmos.pop('ElasticNetCV')
  if st.checkbox('LassoLarsCV', value=True):
    dict_algoritmos['LassoLarsCV'] = LassoLarsCV()
  else:
    dict_algoritmos.pop('LassoLarsCV')
  if st.checkbox('LassoLarsIC', value=True):
    dict_algoritmos['LassoLarsIC'] = LassoLarsIC()
  else:
    dict_algoritmos.pop('LassoLarsIC')
  if st.checkbox('LarsCV', value=True):
    dict_algoritmos['LarsCV'] = LarsCV()
  else:
    dict_algoritmos.pop('LarsCV')
  if st.checkbox('Lars', value=True):
    dict_algoritmos['Lars'] = Lars()
  else:
    dict_algoritmos.pop('Lars')
  if st.checkbox('SGDRegressor', value=True):
    dict_algoritmos['SGDRegressor'] = SGDRegressor()
  else:
    dict_algoritmos.pop('SGDRegressor')
  if st.checkbox('RANSACRegressor', value=True):
    dict_algoritmos['RANSACRegressor'] = RANSACRegressor()
  else:
    dict_algoritmos.pop('RANSACRegressor')
  if st.checkbox('ElasticNet', value=True):
    dict_algoritmos['ElasticNet'] = ElasticNet()
  else:
    dict_algoritmos.pop('ElasticNet')
  if st.checkbox('Lasso', value=True):
    dict_algoritmos['Lasso'] = Lasso()
  else:
    dict_algoritmos.pop('Lasso')
  if st.checkbox('OrthogonalMatchingPursuitCV', value=True):
    dict_algoritmos['OrthogonalMatchingPursuitCV'] = OrthogonalMatchingPursuitCV()
  else:
    dict_algoritmos.pop('OrthogonalMatchingPursuitCV')
  if st.checkbox('ExtraTreeRegressor', value=True):
    dict_algoritmos['ExtraTreeRegressor'] = ExtraTreeRegressor()
  else:
    dict_algoritmos.pop('ExtraTreeRegressor')
  if st.checkbox('PassiveAggressiveRegressor', value=True):
    dict_algoritmos['PassiveAggressiveRegressor'] = PassiveAggressiveRegressor()
  else:
    dict_algoritmos.pop('PassiveAggressiveRegressor')
  if st.checkbox('GaussianProcessRegressor', value=True):
    dict_algoritmos['GaussianProcessRegressor'] = GaussianProcessRegressor()
  else:
    dict_algoritmos.pop('GaussianProcessRegressor')
  if st.checkbox('OrthogonalMatchingPursuit', value=True):
    dict_algoritmos['OrthogonalMatchingPursuit'] = OrthogonalMatchingPursuit()
  else:
    dict_algoritmos.pop('OrthogonalMatchingPursuit')
  if st.checkbox('DecisionTreeRegressor', value=True):
    dict_algoritmos['DecisionTreeRegressor'] = DecisionTreeRegressor()
  else:
    dict_algoritmos.pop('DecisionTreeRegressor')
  if st.checkbox('DummyRegressor', value=True):
    dict_algoritmos['DummyRegressor'] = DummyRegressor()
  else:
    dict_algoritmos.pop('DummyRegressor')
  if st.checkbox('LassoLars', value=True):
    dict_algoritmos['LassoLars'] = LassoLars()
  else:
    dict_algoritmos.pop('LassoLars')
  if st.checkbox('KernelRidge', value=True):
    dict_algoritmos['KernelRidge'] = KernelRidge()
  else:
    dict_algoritmos.pop('KernelRidge')

elif (tarefa == "Classificação"):

  # ----------------------------------------------------------------------------

  # Crie um dicionário de classificadores para testar
  dict_algoritmos = {}

  from sklearn.utils import all_estimators
  classifiers = all_estimators(type_filter='classifier') # Obtendo todos os classificadores disponíveis
  for name, ClassifierClass in classifiers:
    if name not in ['CategoricalNB','ClassifierChain','MultiOutputClassifier','OneVsOneClassifier','OneVsRestClassifier','OutputCodeClassifier','VotingClassifier','StackingClassifier','RadiusNeighborsClassifier']:
      classifier = ClassifierClass()
      dict_algoritmos[type(classifier).__name__] = classifier

  # ----------------------------------------------------------------------------

  dict_algoritmos["OneVsOneClassifier"] = OneVsOneClassifier(estimator=LinearSVC(random_state=0), n_jobs=-1)
  dict_algoritmos["OneVsRestClassifier"] = OneVsRestClassifier(estimator=SVC(), n_jobs=-1)
  dict_algoritmos["OutputCodeClassifier"] = OutputCodeClassifier(estimator=RandomForestClassifier(), code_size=2, random_state=42)
  dict_algoritmos["RadiusNeighborsClassifier"] =  RadiusNeighborsClassifier(radius=2.0, weights='uniform', algorithm='auto')

  if st.checkbox('OneVsOneClassifier', value=True):
    dict_algoritmos['OneVsOneClassifier'] = OneVsOneClassifier(estimator=LogisticRegression(), n_jobs=-1)
  else:
    dict_algoritmos.pop('OneVsOneClassifier')
  if st.checkbox('OneVsRestClassifier', value=True):
    dict_algoritmos['OneVsRestClassifier'] = OneVsRestClassifier(estimator=LogisticRegression(), n_jobs=-1)
  else:
    dict_algoritmos.pop('OneVsRestClassifier')
  if st.checkbox('OutputCodeClassifier', value=True):
    dict_algoritmos['OutputCodeClassifier'] = OutputCodeClassifier(estimator=LogisticRegression(), code_size=2, random_state=42)
  else:
    dict_algoritmos.pop('OutputCodeClassifier')
  if st.checkbox('RadiusNeighborsClassifier', value=True):
    dict_algoritmos['RadiusNeighborsClassifier'] = RadiusNeighborsClassifier(radius=2.0, weights='uniform', algorithm='auto')
  else:
    dict_algoritmos.pop('RadiusNeighborsClassifier')

  # ----------------------------------------------------------------------------

  if st.checkbox('AdaBoostClassifier', value=True):
    dict_algoritmos['AdaBoostClassifier'] = AdaBoostClassifier()
  else:
    dict_algoritmos.pop('AdaBoostClassifier')
  if st.checkbox('BaggingClassifier', value=True):
    dict_algoritmos['BaggingClassifier'] = BaggingClassifier()
  else:
    dict_algoritmos.pop('BaggingClassifier')
  if st.checkbox('BernoulliNB', value=True):
    dict_algoritmos['BernoulliNB'] = BernoulliNB()
  else:
    dict_algoritmos.pop('BernoulliNB')
  if st.checkbox('CalibratedClassifierCV', value=True):
    dict_algoritmos['CalibratedClassifierCV'] = CalibratedClassifierCV()
  else:
    dict_algoritmos.pop('CalibratedClassifierCV')
  if st.checkbox('ComplementNB', value=True):
    dict_algoritmos['ComplementNB'] = ComplementNB()
  else:
    dict_algoritmos.pop('ComplementNB')
  if st.checkbox('DecisionTreeClassifier', value=True):
    dict_algoritmos['DecisionTreeClassifier'] = DecisionTreeClassifier()
  else:
    dict_algoritmos.pop('DecisionTreeClassifier')
  if st.checkbox('DummyClassifier', value=True):
    dict_algoritmos['DummyClassifier'] = DummyClassifier()
  else:
    dict_algoritmos.pop('DummyClassifier')
  if st.checkbox('ExtraTreeClassifier', value=True):
    dict_algoritmos['ExtraTreeClassifier'] = ExtraTreeClassifier()
  else:
    dict_algoritmos.pop('ExtraTreeClassifier')
  if st.checkbox('ExtraTreesClassifier', value=True):
    dict_algoritmos['ExtraTreesClassifier'] = ExtraTreesClassifier()
  else:
    dict_algoritmos.pop('ExtraTreesClassifier')
  if st.checkbox('GaussianNB', value=True):
    dict_algoritmos['GaussianNB'] = GaussianNB()
  else:
    dict_algoritmos.pop('GaussianNB')
  if st.checkbox('GaussianProcessClassifier', value=True):
    dict_algoritmos['GaussianProcessClassifier'] = GaussianProcessClassifier()
  else:
    dict_algoritmos.pop('GaussianProcessClassifier')
  if st.checkbox('GradientBoostingClassifier', value=True):
    dict_algoritmos['GradientBoostingClassifier'] = GradientBoostingClassifier()
  else:
    dict_algoritmos.pop('GradientBoostingClassifier')
  if st.checkbox('HistGradientBoostingClassifier', value=True):
    dict_algoritmos['HistGradientBoostingClassifier'] = HistGradientBoostingClassifier()
  else:
    dict_algoritmos.pop('HistGradientBoostingClassifier')
  if st.checkbox('KNeighborsClassifier', value=True):
    dict_algoritmos['KNeighborsClassifier'] = KNeighborsClassifier()
  else:
    dict_algoritmos.pop('KNeighborsClassifier')
  if st.checkbox('LabelPropagation', value=True):
    dict_algoritmos['LabelPropagation'] = LabelPropagation()
  else:
    dict_algoritmos.pop('LabelPropagation')
  if st.checkbox('LabelSpreading', value=True):
    dict_algoritmos['LabelSpreading'] = LabelSpreading()
  else:
    dict_algoritmos.pop('LabelSpreading')
  if st.checkbox('LinearDiscriminantAnalysis', value=True):
    dict_algoritmos['LinearDiscriminantAnalysis'] = LinearDiscriminantAnalysis()
  else:
    dict_algoritmos.pop('LinearDiscriminantAnalysis')
  if st.checkbox('LinearSVC', value=True):
    dict_algoritmos['LinearSVC'] = LinearSVC()
  else:
    dict_algoritmos.pop('LinearSVC')
  if st.checkbox('LogisticRegression', value=True):
    dict_algoritmos['LogisticRegression'] = LogisticRegression()
  else:
    dict_algoritmos.pop('LogisticRegression')
  if st.checkbox('LogisticRegressionCV', value=True):
    dict_algoritmos['LogisticRegressionCV'] = LogisticRegressionCV()
  else:
    dict_algoritmos.pop('LogisticRegressionCV')
  if st.checkbox('MLPClassifier', value=True):
    dict_algoritmos['MLPClassifier'] = MLPClassifier()
  else:
    dict_algoritmos.pop('MLPClassifier')
  if st.checkbox('MultinomialNB', value=True):
    dict_algoritmos['MultinomialNB'] = MultinomialNB()
  else:
    dict_algoritmos.pop('MultinomialNB')
  if st.checkbox('NearestCentroid', value=True):
    dict_algoritmos['NearestCentroid'] = NearestCentroid()
  else:
    dict_algoritmos.pop('NearestCentroid')
  if st.checkbox('NuSVC', value=True):
    dict_algoritmos['NuSVC'] = NuSVC()
  else:
    dict_algoritmos.pop('NuSVC')
  if st.checkbox('PassiveAggressiveClassifier', value=True):
    dict_algoritmos['PassiveAggressiveClassifier'] = PassiveAggressiveClassifier()
  else:
    dict_algoritmos.pop('PassiveAggressiveClassifier')
  if st.checkbox('Perceptron', value=True):
    dict_algoritmos['Perceptron'] = Perceptron()
  else:
    dict_algoritmos.pop('Perceptron')
  if st.checkbox('QuadraticDiscriminantAnalysis', value=True):
    dict_algoritmos['QuadraticDiscriminantAnalysis'] = QuadraticDiscriminantAnalysis()
  else:
    dict_algoritmos.pop('QuadraticDiscriminantAnalysis')
  if st.checkbox('RandomForestClassifier', value=True):
    dict_algoritmos['RandomForestClassifier'] = RandomForestClassifier()
  else:
    dict_algoritmos.pop('RandomForestClassifier')
  if st.checkbox('RidgeClassifier', value=True):
    dict_algoritmos['RidgeClassifier'] = RidgeClassifier()
  else:
    dict_algoritmos.pop('RidgeClassifier')
  if st.checkbox('RidgeClassifierCV', value=True):
    dict_algoritmos['RidgeClassifierCV'] = RidgeClassifierCV()
  else:
    dict_algoritmos.pop('RidgeClassifierCV')
  if st.checkbox('SGDClassifier', value=True):
    dict_algoritmos['SGDClassifier'] = SGDClassifier()
  else:
    dict_algoritmos.pop('SGDClassifier')
  if st.checkbox('SVC', value=True):
    dict_algoritmos['SVC'] = SVC(class_weight='balanced')
  else:
    dict_algoritmos.pop('SVC')

elif (tarefa == "Classificação Ensemble"):

  # Crie um dicionário de classificadores para testar
  dict_algoritmos = {

      'VotingClassifier': VotingClassifier(estimators=[
                                                         ('dt', DecisionTreeClassifier()),
                                                         ('svc', LinearSVC()),
                                                         ('sgd', SGDClassifier()),
                                                         ('mlp', MLPClassifier()),
                                                         ('per', Perceptron()),
                                                         ('lr', LogisticRegression()),
                                                         ('lrcv', LogisticRegressionCV()),
                                                         ('svc2', SVC()),
                                                         ('cc', CalibratedClassifierCV()),
                                                         ('pa', PassiveAggressiveClassifier()),
                                                         ('lp', LabelPropagation()),
                                                         ('ls', LabelSpreading()),
                                                         ('rf', RandomForestClassifier()),
                                                         ('gb', GradientBoostingClassifier()),
                                                         ('qda', QuadraticDiscriminantAnalysis()),
                                                         ('hgb', HistGradientBoostingClassifier()),
                                                         ('rcv', RidgeClassifierCV()),
                                                         ('rc', RidgeClassifier()),
                                                         ('knn', KNeighborsClassifier()),
                                                         ('bnb', BernoulliNB()),
                                                         ('lda', LinearDiscriminantAnalysis()),
                                                         ('gnb', GaussianNB()),
                                                         ('nusvc', NuSVC()),
                                                         ('dtc', DecisionTreeClassifier()),
                                                         ('nc', NearestCentroid()),
                                                         ('etc', ExtraTreeClassifier()),
                                                         ('dc', DummyClassifier())
                                                         ]),

      'StackingClassifier': StackingClassifier(estimators=[
                                                          ('dt', DecisionTreeClassifier()),
                                                          ('svc', LinearSVC()),
                                                          ('sgd', SGDClassifier()),
                                                          ('mlp', MLPClassifier()),
                                                          ('per', Perceptron()),
                                                          ('lr', LogisticRegression()),
                                                          ('lrcv', LogisticRegressionCV()),
                                                          ('svc2', SVC()),
                                                          ('cc', CalibratedClassifierCV()),
                                                          ('pa', PassiveAggressiveClassifier()),
                                                          ('lp', LabelPropagation()),
                                                          ('ls', LabelSpreading()),
                                                          ('rf', RandomForestClassifier()),
                                                          ('gb', GradientBoostingClassifier()),
                                                          ('qda', QuadraticDiscriminantAnalysis()),
                                                          ('hgb', HistGradientBoostingClassifier()),
                                                          ('rcv', RidgeClassifierCV()),
                                                          ('rc', RidgeClassifier()),
                                                          ('knn', KNeighborsClassifier()),
                                                          ('bnb', BernoulliNB()),
                                                          ('lda', LinearDiscriminantAnalysis()),
                                                          ('gnb', GaussianNB()),
                                                          ('nusvc', NuSVC()),
                                                          ('dtc', DecisionTreeClassifier()),
                                                          ('nc', NearestCentroid()),
                                                          ('etc', ExtraTreeClassifier()),
                                                          ('dc', DummyClassifier())
                                                          ], final_estimator=RandomForestClassifier(random_state=43), cv=5)
  }

  from sklearn.utils import all_estimators
  classifiers = all_estimators(type_filter='classifier') # Obtendo todos os classificadores disponíveis
  for name, ClassifierClass in classifiers:
    if name not in ['AdaBoostClassifier','BaggingClassifier','ExtraTreesClassifier','GradientBoostingClassifier','RandomForestClassifier','VotingClassifier','StackingClassifier','CategoricalNB','ClassifierChain','MultiOutputClassifier','OneVsOneClassifier','OneVsRestClassifier','OutputCodeClassifier']:
      classifier = ClassifierClass()
      if type(classifier).__name__ not in ['AdaBoostClassifier','BaggingClassifier','ExtraTreesClassifier','GradientBoostingClassifier','RandomForestClassifier','VotingClassifier','StackingClassifier','LinearDiscriminantAnalysis','KNeighborsClassifier','LabelPropagation','LabelSpreading','LinearSVC','MLPClassifier','NearestCentroid','PassiveAggressiveClassifier','Perceptron','QuadraticDiscriminantAnalysis','RidgeClassifier','RidgeClassifierCV','SGDClassifier','SVC','GaussianProcessClassifier','NuSVC','RadiusNeighborsClassifier','OneVsOneClassifier','OneVsRestClassifier','OutputCodeClassifier']:
        dict_algoritmos['AdaBoostClassifier' + '-' + type(classifier).__name__] = AdaBoostClassifier(base_estimator=classifier)
      if type(classifier).__name__ not in ['AdaBoostClassifier','BaggingClassifier','ExtraTreesClassifier','GradientBoostingClassifier','RandomForestClassifier','VotingClassifier','StackingClassifier','RadiusNeighborsClassifier']:
        dict_algoritmos['BaggingClassifier' + '-' + type(classifier).__name__] = BaggingClassifier(base_estimator=classifier, n_jobs=-1)
  st.write(dict_algoritmos)

# ------------------------------------------------------------------------------

if st.button('Calcular', key=f"calcular_button_1"):

  # ----------------------------------------------------------------------------

  # Crie uma lista de algoritmos para testar
  algoritmos = list(dict_algoritmos.values())

  # ----------------------------------------------------------------------------

  # Definindo arrays para armazenar as métricas de desempenho de cada modelo treinado e avaliado
  perf_scores = []

  Y_pred = np.empty([len(algoritmos), len(X_test)], dtype=np.uint8)
  n = 0

  ada_base_estimator = []
  ada_base_estimators = []
  bag_base_estimator = []
  bag_base_estimators = []

  for name, model in dict_algoritmos.items():
    try:
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      perf_scores.append([name,
                        accuracy_score(y_test, y_pred),
                        recall_score(y_test, y_pred, pos_label=pos_label, average=average),
                        precision_score(y_test, y_pred, pos_label=pos_label, average=average),
                        f1_score(y_test, y_pred, pos_label=pos_label, average=average)
                        ])
      if type(model).__name__ not in ['AdaBoostClassifier','BaggingClassifier','ExtraTreesClassifier','GradientBoostingClassifier','RandomForestClassifier','VotingClassifier','StackingClassifier','LinearDiscriminantAnalysis','KNeighborsClassifier','LabelPropagation','LabelSpreading','LinearSVC','MLPClassifier','NearestCentroid','PassiveAggressiveClassifier','Perceptron','QuadraticDiscriminantAnalysis','RidgeClassifier','RidgeClassifierCV','SGDClassifier','SVC','GaussianProcessClassifier','NuSVC','RadiusNeighborsClassifier','OneVsOneClassifier','OneVsRestClassifier','OutputCodeClassifier']:
        ada_base_estimator.append(model)
        ada_base_estimators.append((type(model).__name__, model))
      if type(model).__name__ not in ['AdaBoostClassifier','BaggingClassifier','ExtraTreesClassifier','GradientBoostingClassifier','RandomForestClassifier','VotingClassifier','StackingClassifier','RadiusNeighborsClassifier']:
        bag_base_estimator.append(model)
        bag_base_estimators.append((type(model).__name__, model))
      Y_pred[n] = y_pred
      n = n + 1
    except Exception as e:
      st.write(f"Erro: {name} - {model}")
      st.write(str(e))
      pass

  # ----------------------------------------------------------------------------

  st.subheader("DESEMPENHO DOS MODELOS")

  st.write(dict_algoritmos)

  mp.plotar_todas_as_metricas_na_mesma_figura(perf_scores, 'Modelo', drop_duplicates = False)

  # ----------------------------------------------------------------------------

  st.subheader("AGREGANDO AS PREDIÇÕES DE MÚLTIPLOS CLASSIFICADORES")

  # ----------------------------------------------------------------------------

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

  if (tarefa == "Classificação"):

    st.subheader("AVALIA O DESEMPENHO DA CLASSIFICAÇÃO COM ENSEMBLE")

    # Crie um dicionário de classificadores para testar
    dict_algoritmos_ensemble = {}

    adaboost_estimators = []
    bagging_estimators = []

    if bag_base_estimator:
      for alg in bag_base_estimator:
        dict_algoritmos_ensemble['BaggingClassifier' + '-' + type(alg).__name__] = BaggingClassifier(base_estimator=alg, n_jobs=-1, random_state=42)
        bagging_estimators.append(('BaggingClassifier' + '-' + type(alg).__name__, BaggingClassifier(base_estimator=alg, n_jobs=-1, random_state=42)))

      #dict_algoritmos_ensemble['VotingClassifier-BaggingClassifier'] = VotingClassifier(estimators=bagging_estimators)
      #dict_algoritmos_ensemble['StackingClassifier-BaggingClassifier'] = StackingClassifier(estimators=bagging_estimators, final_estimator=RandomForestClassifier(random_state=43), cv=5)

    if ada_base_estimator:
      for alg in ada_base_estimator:
        dict_algoritmos_ensemble['AdaBoostClassifier' + '-' + type(alg).__name__] = AdaBoostClassifier(base_estimator=alg, random_state=42)
        adaboost_estimators.append(('AdaBoostClassifier' + '-' + type(alg).__name__, AdaBoostClassifier(base_estimator=alg, random_state=42)))

      #dict_algoritmos_ensemble['VotingClassifier-AdaBoostClassifier'] = VotingClassifier(estimators=adaboost_estimators)
      #dict_algoritmos_ensemble['StackingClassifier-AdaBoostClassifier'] = StackingClassifier(estimators=adaboost_estimators, final_estimator=RandomForestClassifier(random_state=43), cv=5)

    estimators = [(k, v) for k, v in dict_algoritmos.items()]
    if 'CategoricalNB' in estimators:
      estimators.remove('CategoricalNB')
    if estimators:
      dict_algoritmos_ensemble['VotingClassifier'] = VotingClassifier(estimators=estimators)
      dict_algoritmos_ensemble['StackingClassifier'] = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(random_state=43), cv=5)

    # ----------------------------------------------------------------------------

    # Crie uma lista de algoritmos para testar
    algoritmos = list(dict_algoritmos_ensemble.values())

    # ----------------------------------------------------------------------------

    # Definindo arrays para armazenar as métricas de desempenho de cada modelo treinado e avaliado
    perf_scores = []

    Y_pred = np.empty([len(algoritmos), len(X_valid)], dtype=np.uint8)
    n = 0

    for name, model in dict_algoritmos_ensemble.items():
      try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        perf_scores.append([name,
                          accuracy_score(y_valid, y_pred),
                          recall_score(y_valid, y_pred, pos_label=pos_label, average=average),
                          precision_score(y_valid, y_pred, pos_label=pos_label, average=average),
                          f1_score(y_valid, y_pred, pos_label=pos_label, average=average)
                          ])
        Y_pred[n] = y_pred
        n = n + 1
      except Exception as e:
        st.write(f"Erro: {name} - {model}")
        st.write(str(e))
        pass

    # ----------------------------------------------------------------------------

    st.subheader("DESEMPENHO DOS MODELOS")

    st.write(dict_algoritmos_ensemble)

    mp.plotar_todas_as_metricas_na_mesma_figura(perf_scores, 'Modelo', drop_duplicates = False)

# ------------------------------------------------------------------------------
