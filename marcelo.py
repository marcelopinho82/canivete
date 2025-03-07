
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append("/content/marcelo/marcelo")
import the_confusion_matrix_visualized as tcmv
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------------------

# https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Utilities/ML-Python-utils.py

def plot_decision_boundaries(X, y, xlabel, ylabel, limit, model_class, **model_params):
    """
    Function to plot the decision boundaries of a classification model.
    This uses just the first two columns of the data for fitting
    the model as we need to find the predicted value for every point in
    scatter plot.
    Arguments:
            X: Feature data as a NumPy-type array.
            y: Label data as a NumPy-type array.
            model_class: A Scikit-learn ML estimator class
            e.g. GaussianNB (imported from sklearn.naive_bayes) or
            LogisticRegression (imported from sklearn.linear_model)
            **model_params: Model parameters to be passed on to the ML estimator

    Typical code example:
            plt.figure()
            plt.title("KNN decision boundary with neighbros: 5",fontsize=16)
            plot_decision_boundaries(X_train,y_train,KNeighborsClassifier,n_neighbors=5)
            plt.show()
    """
    try:
        X = np.array(X)
        y = np.array(y).flatten()
    except:
        print("Coercing input data to NumPy arrays failed")

    # Reduces to the first two columns of data - for a 2D plot!
    reduced_data = X[:, :2]

    # Instantiate the model object
    model = model_class(**model_params)

    # Fits the model with the reduced data
    model.fit(reduced_data, y)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.02  # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - limit, reduced_data[:, 0].max() + limit
    y_min, y_max = reduced_data[:, 1].min() - limit, reduced_data[:, 1].max() + limit
    # Meshgrid creation
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    x_min, x_max = X[:, 0].min() - limit, X[:, 0].max() + limit
    y_min, y_max = X[:, 1].min() - limit, X[:, 1].max() + limit
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Predictions to obtain the classification results
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plotting
    plt.contourf(xx, yy, Z, alpha=0.2, cmap="viridis")
    g = plt.scatter(
        X[:, 0], X[:, 1], c=y, alpha=0.6, s=50, edgecolor="k", cmap="viridis"
    )
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(handles=g.legend_elements()[0], labels=np.unique(y).tolist())
    return plt

# ------------------------------------------------------------------------------

# https://stackoverflow.com/questions/52645710/a-list-of-downloaded-files-names-in-google-colabarotary
# https://www.geeksforgeeks.org/python-filter-list-of-strings-based-on-the-substring-list/

def Filter(string, substr):

    # Essa função filtra uma lista de strings e retorna apenas aquelas que contêm alguma string especificada em outra lista de strings (substr).
    # Ela percorre cada string em string e verifica se alguma string em substr está contida na string atual.
    # Se sim, a string atual é adicionada à lista de resultados.

    return [str for str in string if any(sub in str for sub in substr)]

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------

# https://discuss.streamlit.io/t/how-do-i-use-a-background-image-on-streamlit/5067/5

import base64

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.

    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"

    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# -------------------------------------------------------------------------

# https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

# ------------------------------------------------------------------------------

def sklearn_to_df(sklearn_dataset):

    # https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset

    # Esta função tem como objetivo converter um conjunto de dados do scikit-learn para um dataframe do pandas.
    # A função recebe como argumento um conjunto de dados do scikit-learn.
    # A primeira linha cria um dataframe a partir dos dados do conjunto de dados e define os nomes das colunas como os nomes dos recursos do conjunto de dados.
    # A segunda linha adiciona uma nova coluna chamada 'target' ao dataframe, que contém os valores-alvo do conjunto de dados.
    # A última linha retorna o novo dataframe.

    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df["target"] = pd.Series(sklearn_dataset.target)
    return df

# ------------------------------------------------------------------------------

# Autor: Marcelo Vinicius Ludgero de Pinho

def legenda_matriz_confusao(TN, FN, TP, FP, pos_label, neg_label):

    # https://keytodatascience.com/confusion-matrix/
    # https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/#:~:text=Confusion%20Matrix%20for%20multiclass%20classification,number%20of%20classes%20or%20outputs.

    # Essa função recebe o número de verdadeiros negativos (TN), falsos negativos (FN), verdadeiros positivos (TP) e falsos positivos (FP)
    # bem como as etiquetas para as classes positiva e negativa.
    # Então ele escreve no aplicativo Streamlit uma legenda que descreve o que cada um desses valores representa.

    # Escreve um título indicando que essa é a legenda para a matriz de confusão
    st.write("Legenda:")

    # Escreve as etiquetas das classes positiva e negativa
    st.write(
        f'- A classe positiva é "{pos_label}" e a classe negativa é "{neg_label}".'
    )

    # Escreve a definição de verdadeiros positivos
    st.write(
        f'- Verdadeiro Positivo: {TP} instâncias foram classificadas corretamente como sendo da classe "{pos_label}" e não da classe "{neg_label}".'
    )
    st.write(
        f'O modelo previu positivo e é verdade. O modelo previu que a instância é da classe "{pos_label}" e realmente é.'
    )
    st.write(
        f'Isso significa que o valor real e também os valores previstos são os mesmos. A classe real é "{pos_label}" e a previsão do modelo também é "{pos_label}".'
    )

    # Escreve a definição de verdadeiros negativos
    st.write(
        f'- Verdadeiro Negativo: {TN} instâncias foram classificadas corretamente como sendo da classe "{neg_label}" e não da classe "{pos_label}".'
    )
    st.write(
        f'O modelo previu negativo e é verdade. O modelo previu que a instância é da classe "{neg_label}" e realmente é.'
    )
    st.write(
        f'Isso significa que o valor real e também os valores previstos são os mesmos. A classe real é "{neg_label}" e a previsão do modelo também é "{neg_label}".'
    )

    # Escreve a definição de falsos positivos
    st.write(
        f'- Falso Positivo (Erro Tipo 1): {FP} instâncias foram classificadas incorretamente como sendo da classe "{pos_label}".'
    )
    st.write(
        f'O modelo previu positivo e é falso. O modelo previu que a instância é da classe "{pos_label}", mas na verdade não é (é da classe "{neg_label}").'
    )
    st.write(
        f'Isso significa que o valor real é negativo em nosso caso, é da classe "{neg_label}", mas o modelo o previu como positivo, ou seja, "{pos_label}". Portanto, o modelo deu a previsão errada, deveria dar negativo ("{neg_label}"), mas deu positivo ("{pos_label}"), assim a saída positiva que obtivemos é falsa.'
    )

    # Escreve a definição de falsos negativos
    st.write(
        f'- Falso Negativo (Erro Tipo 2): {FN} instâncias foram classificadas incorretamente como sendo da classe "{neg_label}".'
    )
    st.write(
        f'O modelo previu negativo e é falso. O modelo previu que a instância é da classe "{neg_label}", mas na verdade não é (é da classe "{pos_label}").'
    )
    st.write(
        f'Isso significa que o valor real é positivo em nosso caso, é da classe "{pos_label}", mas o modelo o previu como negativo, ou seja, "{neg_label}". Portanto, o modelo deu a previsão errada. Era para dar um positivo ("{pos_label}"), mas deu um negativo ("{neg_label}"), assim o resultado negativo que obtivemos é falso.'
    )

# ------------------------------------------------------------------------------

def conclusao_matriz_confusao(TN, FN, TP, FP, pos_label, neg_label):

    # Essa função recebe o número de verdadeiros negativos (TN), falsos negativos (FN), verdadeiros positivos (TP) e falsos positivos (FP)
    # bem como as etiquetas para as classes positiva e negativa.
    # Então ele escreve no aplicativo Streamlit uma conclusão baseada nas métricas de acurácia, precisão, recall e especificidade.

    # Calcula as métricas de acurácia, precisão, recall e especificidade
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # Escreve a conclusão
    st.write("Nós podemos concluir que:")

    # Importa a biblioteca math para checar se as métricas são NaN
    import math

    # Verifica se a acurácia não é NaN e escreve a conclusão
    if not (math.isnan(accuracy)):
        st.write(
            f'A acurácia de {round(accuracy*100, 2)}% significa que a identificação de {round((100-int(accuracy*100))/10, 2)} de cada 10 instâncias da classe "{pos_label}" está incorreta e {round((accuracy*100)/10, 2)} está correta.'
        )

    # Verifica se a precisão não é NaN e escreve a conclusão
    if not (math.isnan(precision)):
        st.write(
            f'A precisão de {round(precision*100, 2)}% significa que a classificação de {round((100-int(precision*100))/10, 2)} de cada 10 instâncias da classe "{pos_label}" não são da classe "{pos_label}" (ou seja, são da classe "{neg_label}") e {round((precision*100)/10, 2)} são da classe "{pos_label}".'
        )

    # Verifica se o recall não é NaN e escreve a conclusão
    if not (math.isnan(recall)):
        st.write(
            f'O recall é de {round(recall*100, 2)}%, o que significa que {round((100-int(recall*100))/10, 2)} em cada 10 instâncias da classe "{pos_label}", na realidade, são perdidos pelo nosso modelo e {round((recall*100)/10, 2)} são corretamente identificados como sendo da classe "{pos_label}".'
        )

    # Verifica se a especificidade não é NaN e escreve a conclusão
    if not (math.isnan(specificity)):
        st.write(
            f'A especificidade é de {round(specificity*100, 2)}%, o que significa que {round((100-int(specificity*100))/10, 2)} em cada 10 instâncias da classe "{neg_label}" (ou seja, não são da classe "{pos_label}") na realidade são erroneamente rotulados como sendo da classe "{pos_label}" e {round((specificity*100)/10, 2)} são corretamente rotulados como sendo da classe "{neg_label}".'
        )

# ------------------------------------------------------------------------------

def plota_metricas_duas_classes(y_test, y_pred, pos_label):

    # Essa função recebe os rótulos reais (y_test) e os rótulos previstos pelo modelo (y_pred)
    # bem como a etiqueta para a classe positiva.
    # Então ele plota as métricas de acurácia, precisão, recall e F1 score no aplicativo Streamlit.
    # Importa as métricas de recall, precisão, acurácia e F1 score da biblioteca sklearn
    from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

    # Cria um container para armazenar as métricas
    data_container = st.container()
    with data_container:
        coluna_1, coluna_2, coluna_3, coluna_4 = st.columns(4)
        with coluna_1:
            # Plota a métrica de acurácia
            st.metric(
                "Acurácia",
                round(accuracy_score(y_test, y_pred), 3),
                delta=None,
                delta_color="normal",
                help=None,
            )
        with coluna_2:
            # Plota a métrica de precisão
            st.metric(
                "Precisão",
                round(
                    precision_score(
                        y_test, y_pred, pos_label=pos_label, average="binary"
                    ),
                    3,
                ),
                delta=None,
                delta_color="normal",
                help=None,
            )
        with coluna_3:
            # Plota a métrica de recall
            st.metric(
                "Recall",
                round(
                    recall_score(y_test, y_pred, pos_label=pos_label, average="binary"),
                    3,
                ),
                delta=None,
                delta_color="normal",
                help=None,
            )
        with coluna_4:
            # Plota a métrica de F1 score
            st.metric(
                "F1 score",
                round(
                    f1_score(y_test, y_pred, pos_label=pos_label, average="binary"), 3
                ),
                delta=None,
                delta_color="normal",
                help=None,
            )

# ------------------------------------------------------------------------------

def plota_metricas_multiclasse(y_test, y_pred, pos_label):

    # Essa função recebe os rótulos reais (y_test) e os rótulos previstos pelo modelo (y_pred)
    # bem como a etiqueta para a classe positiva.
    # Então ele plota as métricas de acurácia, precisão, recall e F1 score no aplicativo Streamlit.
    # Importa as métricas de recall, precisão, acurácia e F1 score da biblioteca sklearn

    from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

    # Cria um container para armazenar as métricas
    data_container = st.container()
    with data_container:
        coluna_1, coluna_2, coluna_3, coluna_4 = st.columns(4)
        with coluna_1:
            # Plota a métrica de acurácia
            st.metric(
                "Acurácia",
                round(accuracy_score(y_test, y_pred), 3),
                delta=None,
                delta_color="normal",
                help=None,
            )
        with coluna_2:
            # Plota a métrica de precisão micro
            st.metric(
                "Micro Precisão",
                round(
                    precision_score(
                        y_test, y_pred, pos_label=pos_label, average="micro"
                    ),
                    3,
                ),
                delta=None,
                delta_color="normal",
                help=None,
            )
            # Plota a métrica de precisão macro
            st.metric(
                "Macro Precisão",
                round(
                    precision_score(
                        y_test, y_pred, pos_label=pos_label, average="macro"
                    ),
                    3,
                ),
                delta=None,
                delta_color="normal",
                help=None,
            )
            # Plota a métrica de precisão weighted
            st.metric(
                "Weighted Precisão",
                round(
                    precision_score(
                        y_test, y_pred, pos_label=pos_label, average="weighted"
                    ),
                    3,
                ),
                delta=None,
                delta_color="normal",
                help=None,
            )
        with coluna_3:
            # Plota a métrica de recall micro
            st.metric(
                "Micro Recall",
                round(
                    recall_score(y_test, y_pred, pos_label=pos_label, average="micro"),
                    3,
                ),
                delta=None,
                delta_color="normal",
                help=None,
            )
            # Plota a métrica de recall macro
            st.metric(
                "Macro Recall",
                round(
                    recall_score(y_test, y_pred, pos_label=pos_label, average="macro"),
                    3,
                ),
                delta=None,
                delta_color="normal",
                help=None,
            )
            # Plota a métrica de recall weighted
            st.metric(
                "Weighted Recall",
                round(
                    recall_score(
                        y_test, y_pred, pos_label=pos_label, average="weighted"
                    ),
                    3,
                ),
                delta=None,
                delta_color="normal",
                help=None,
            )
        with coluna_4:
            # Plota a métrica de F1 score micro
            st.metric(
                "Micro F1 score",
                round(
                    f1_score(y_test, y_pred, pos_label=pos_label, average="micro"), 3
                ),
                delta=None,
                delta_color="normal",
                help=None,
            )
            # Plota a métrica de F1 score macro
            st.metric(
                "Macro F1 score",
                round(
                    f1_score(y_test, y_pred, pos_label=pos_label, average="macro"), 3
                ),
                delta=None,
                delta_color="normal",
                help=None,
            )
            # Plota a métrica de F1 score weighted
            st.metric(
                "Weighted F1 score",
                round(
                    f1_score(y_test, y_pred, pos_label=pos_label, average="weighted"), 3
                ),
                delta=None,
                delta_color="normal",
                help=None,
            )

# ------------------------------------------------------------------------------

# https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas


def top_entries(df):

    """
    Essa função cria uma matriz de correlação entre todos os atributos de um dataframe.
    Ela remove entradas duplicadas e de identidade, desempilha a matriz, classifica-a em ordem crescente e renomeia as colunas.
    """

    mat = df.corr().abs()

    # Remove duplicate and identity entries
    mat.loc[:, :] = np.tril(mat.values, k=-1)
    mat = mat[mat > 0]

    # Unstack, sort ascending, and reset the index, so features are in columns
    # instead of indexes (allowing e.g. a pretty print in Jupyter).
    # Also rename these it for good measure.
    return (
        mat.unstack()
        .sort_values(ascending=False)
        .reset_index()
        .rename(
            columns={"level_0": "Atributo X", "level_1": "Atributo Y", 0: "Correlação"}
        )
    )

# ------------------------------------------------------------------------------

# Autor: Marcelo Vinicius Ludgero de Pinho

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

def plota_todos_os_resultados(n_classes, classes, pos_label, neg_label, y_test, y_pred):

    # https://stackoverflow.com/questions/20927368/how-to-normalize-a-confusion-matrix
    # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html

    # cm = confusion_matrix(y_test, y_pred)
    # Normalise
    # cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # fig, ax = plt.subplots(figsize=(10,10))
    # sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show(block=False)
    # st.pyplot(fig)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    cm = confusion_matrix(
        y_test, y_pred, labels=classes, sample_weight=None, normalize="true"
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp = disp.plot(
        include_values=True, cmap="Blues", ax=ax, xticks_rotation="horizontal"
    )
    plt.grid(False)
    plt.show()
    st.pyplot(fig)

    # https://medium.com/analytics-vidhya/visually-interpreting-the-confusion-matrix-787a70b65678
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    cm = confusion_matrix(
        y_test, y_pred, labels=classes, sample_weight=None, normalize=None
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp = disp.plot(
        include_values=True, cmap="Blues", ax=ax, xticks_rotation="horizontal"
    )
    plt.grid(False)
    plt.show()
    st.pyplot(fig)

    # https://www.v7labs.com/blog/confusion-matrix-guide
    st.write(
        f"Como de costume, os elementos diagonais são as amostras previstas corretamente. Um total de {cm.diagonal().sum()} instâncias foi previsto corretamente do total de {cm.sum()} instâncias."
    )

    # --------------------------------------------------------------------------

    # True Positives (TP, blue distribution) are the people that truly have the virus.
    # True Negatives (TN, red distribution) are the people that truly DO NOT have the virus.
    # False Positives (FP) are the people that are truly NOT sick but based on the test, they were falsely (False) denoted as sick (Positives).
    # False Negatives (FN) are the people that are truly sick but based on the test, they were falsely (False) denoted as NOT sick (Negative).

    # Verdadeiros Positivos (TP, distribuição azul) são as pessoas que realmente têm o vírus.
    # Verdadeiros Negativos (TN, distribuição vermelha) são as pessoas que realmente NÃO têm o vírus.
    # Falsos Positivos (FP) são as pessoas que realmente NÃO estão doentes, mas com base no teste, foram falsamente (Falso) denotadas como doentes (Positivos).
    # Falsos Negativos (FN) são as pessoas que estão realmente doentes, mas com base no teste, foram falsamente (Falso) denotadas como NÃO doentes (Negativo).

    # --------------------------------------------------------------------------

    if n_classes == 2:
        TN = cm[0][0]  # TN (Verdadeiro Negativo)
        FN = cm[1][0]  # FN (Falso Negativo)
        TP = cm[1][1]  # TP (Verdadeiro Positivo)
        FP = cm[0][1]  # FP (Falso Positivo)
        legenda_matriz_confusao(TN, FN, TP, FP, pos_label, neg_label)

    # --------------------------------------------------------------------------

    else:
        # https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
        # https://keytodatascience.com/confusion-matrix/
        # https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/#:~:text=Confusion%20Matrix%20for%20multiclass%20classification,number%20of%20classes%20or%20outputs.

        FP = cm.sum(axis=0) - np.diag(cm)  # FP (Falso Positivo)
        FN = cm.sum(axis=1) - np.diag(cm)  # FN (Falso Negativo)
        TP = np.diag(cm)  # TP (Verdadeiro Positivo)
        TN = cm.sum() - (FP + FN + TP)  # TN (Verdadeiro Negativo)

        data = {
            "FP": FP.tolist(),
            "FN": FN.tolist(),
            "TP": TP.tolist(),
            "TN": TN.tolist(),
        }
        cm_df = pd.DataFrame(data, columns=["FP", "FN", "TP", "TN"], index=classes)
        st.dataframe(cm_df)

        # ----------------------------------------------------------------------

        data_container = st.container()
        with data_container:
            coluna_1, coluna_2 = st.columns(2)
            with coluna_1:
                # i = cm_df['FP'].idxmax() - 1
                # st.altair_chart(tcmv.confusion_matrix_chart(TN.tolist()[i], FP.tolist()[i], FN.tolist()[i], TP.tolist()[i]), use_container_width=True)
                if cm_df["FP"].loc[cm_df["FP"].idxmax()] > 0:
                    st.write("FP (Falso Positivo)")
                    st.write(
                        f"Para melhorar o desempenho do modelo, deve-se focar nos resultados preditivos da classe {cm_df['FP'].idxmax()}."
                    )
                    st.write(
                        f"Um total de {cm_df['FP'].loc[cm_df['FP'].idxmax()]} instâncias (somando os números nas caixas com exceção da diagonal na coluna {cm_df['FP'].idxmax()}) foram classificadas incorretamente pelo classificador, que é a maior taxa de classificação incorreta entre todas as classes."
                    )

            with coluna_2:
                # i = cm_df['FN'].idxmax() - 1
                # st.altair_chart(tcmv.confusion_matrix_chart(TN.tolist()[i], FP.tolist()[i], FN.tolist()[i], TP.tolist()[i]), use_container_width=True)
                if cm_df["FN"].loc[cm_df["FN"].idxmax()] > 0:
                    st.write("FN (Falso Negativo)")
                    st.write(
                        f"Para melhorar o desempenho do modelo, deve-se focar nos resultados preditivos da classe {cm_df['FN'].idxmax()}."
                    )
                    st.write(
                        f"Um total de {cm_df['FN'].loc[cm_df['FN'].idxmax()]} instâncias foram classificadas incorretamente pelo classificador, que é a maior taxa de classificação incorreta entre todas as classes."
                    )

        # ----------------------------------------------------------------------

        st.altair_chart(
            tcmv.confusion_matrix_chart(
                sum(TN.tolist()), sum(FP.tolist()), sum(FN.tolist()), sum(TP.tolist())
            ),
            use_container_width=True,
        )
        st.write(f"Total FP (Falso Positivo): {sum(FP.tolist())}")
        st.write(f"Total FN (Falso Negativo): {sum(FN.tolist())}")
        st.write(f"Total TP (Verdadeiro Positivo): {sum(TP.tolist())}")
        st.write(f"Total TN (Verdadeiro Negativo): {sum(TN.tolist())}")

    # --------------------------------------------------------------------------

    if n_classes == 2:
        plota_metricas_duas_classes(y_test, y_pred, pos_label)
    else:
        plota_metricas_multiclasse(y_test, y_pred, pos_label)

    # --------------------------------------------------------------------------

    from sklearn.metrics import classification_report

    classification_report = classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(classification_report).transpose()
    st.write(df_classification_report)

    # --------------------------------------------------------------------------

    if n_classes == 2:
        conclusao_matriz_confusao(TN, FN, TP, FP, pos_label, neg_label)

# ------------------------------------------------------------------------------

# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

# Definição da função para gerar um gráfico acurácia vs valores do hiperparâmetro
def plot_acc_vs_hypervalues(y1, y2, x, xlabel, ylabel):

    """
    Essa função cria um gráfico com os valores y1 e y2 em função do x, com as respectivas legendas xlabel e ylabel.
    Ele mostra tanto o desempenho no conjunto de treinamento quanto no conjunto de teste.
    """

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {xlabel} para conjuntos de treinamento e teste")
    ax.plot(x, y1, marker="o", label="train", drawstyle="steps-post")
    ax.plot(x, y2, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    ax.grid()
    plt.show()
    st.pyplot(fig)

# ------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Algumas proporções para a divisão dos dados em treino/teste são comumente adotadas na literatura: 80%/20% e 75%/25%.

# ------------------------------------------------------------------------------

def metodo_holdout_2(X, y, train_ratio, normalize):

    # A função acima é chamada "metodo_holdout_2" e ela é usada para dividir um conjunto de dados em dois conjuntos: um conjunto de treinamento e um conjunto de teste. Ela recebe como argumentos as características (X), as etiquetas (y), a proporção de treinamento (train_ratio) e uma flag para normalização (normalize).
    # A função usa a função train_test_split do scikit-learn para dividir o conjunto de dados em um conjunto de treinamento e teste. A proporção de treinamento é usada para determinar a proporção de dados que serão usados para treinamento e a proporção de teste é calculada subtraindo a proporção de treinamento de 1.
    # A flag de normalização é usada para determinar se os dados devem ser normalizados antes de serem divididos. Se a flag for True, os dados são normalizados usando o objeto MinMaxScaler do scikit-learn antes de serem divididos.
    # Finalmente, a função retorna os conjuntos de treinamento e teste divididos e normalizados (se aplicável) para serem usados em modelos de aprendizado de máquina.

    test_ratio = round(1 - train_ratio, 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )
    if normalize:
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# ------------------------------------------------------------------------------

def metodo_holdout_3(X, y, train_ratio, normalize):

    # Método Holdout 3
    # Este método divide o conjunto de dados em três partes: treinamento, validação e teste.
    # A divisão é feita em proporções específicas, onde o conjunto de treinamento é a maior parte, seguido pelo conjunto de validação e teste.
    # Isso é útil quando queremos verificar se o modelo generaliza bem para novos dados, além de avaliar sua capacidade de generalização
    # durante o processo de treinamento.
    # A normalização dos dados é opcional e pode ser habilitada como parâmetro.
    # O método retorna os conjuntos de treinamento, validação e teste para os dados de entrada e saída (X, y).

    test_ratio = round((1 - train_ratio) / 2, 3)
    validation_ratio = round((1 - train_ratio) / 2, 3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=None, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=validation_ratio / (train_ratio + test_ratio),
        stratify=y_train,
        random_state=None,
    )
    if normalize:
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test, X_valid, y_valid

# ------------------------------------------------------------------------------

def plota_grafico_parametro_gridsearchcv(gridS, scoring, parametro, dtype):

  # Essa função plota um gráfico mostrando o desempenho de diferentes valores de um parâmetro específico usado no GridSearchCV,
  # comparando o desempenho nos conjuntos de treinamento e teste.
  # Ele também mostra o valor do parâmetro que maximiza a métrica de desempenho escolhida.

  results = gridS.cv_results_

  # https://numpy.org/doc/stable/reference/generated/numpy.array.html
  X_axis = np.array(results['param_' + parametro].data, dtype=dtype)

  fig = plt.figure(figsize=(10, 7))
  plt.title(f"Resultados do GridSearchCV - Desempenho {parametro}", fontsize=16)
  plt.xlabel(f"{parametro}") # Nome do parâmetro a ser analisado
  plt.ylabel("Desempenho")

  ax = plt.gca()

  for scorer, color in zip(sorted(scoring), ['g', 'k', 'b', 'r']):
      for sample, style in (('train', '--'), ('test', '-')):
         sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
         sample_score_std = results['std_%s_%s' % (sample, scorer)]
         ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                      sample_score_mean + sample_score_std,
                      alpha=0.1 if sample == 'test' else 0, color=color)
         ax.plot(X_axis, sample_score_mean, style, color=color,
              alpha=1 if sample == 'test' else 0.7,
              label="%s (%s)" % (scorer, sample))

      best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
      best_score = results['mean_test_%s' % scorer][best_index]

      ## Plota uma linha vertical para o valor de hiperparâmetro que maximiza a métrica de desempenho
      ax.plot([X_axis[best_index], ] * 2, [0, best_score],
          linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)
      ## Anota o valor do melhor score
      ax.annotate("%0.3f" % best_score,
              (X_axis[best_index], best_score + 0.008))

  plt.legend(loc="best")
  plt.grid(False)
  plt.show()
  st.pyplot(fig)

# ------------------------------------------------------------------------------

def plota_resultado_gridsearchcv(grid_search_cv):

    # Essa função plota o resultado de uma busca de grid search,
    # mostrando os resultados em formato de tabela, o melhor classificador,
    # a melhor configuração de hiperparâmetros e a melhor pontuação.

    st.write("Resultados")
    results = grid_search_cv.cv_results_
    st.dataframe(results)

    st.write(f"Melhor classificador: {grid_search_cv.best_estimator_}")

    st.write("Melhor configuração de hiperparâmetros")
    st.write(grid_search_cv.best_params_)

    dict_best_param = grid_search_cv.best_params_
    df_best_param = pd.DataFrame.from_dict([dict_best_param])
    st.dataframe(df_best_param.transpose())

    st.metric(
        "Pontuação média de validação cruzada do best_estimator",
        round(grid_search_cv.best_score_, 3),
        delta=None,
        delta_color="normal",
        help=None,
    )

# ------------------------------------------------------------------------------

def plotar_metrica_individualmente(x, y, xlabel, ylabel):

    # Função para plotar uma métrica individualmente em relação a um parâmetro. Recebe como parâmetros:
    # x: valores do parâmetro
    # y: valores da métrica
    # xlabel: nome do parâmetro
    # ylabel: nome da métrica
    # Plota o gráfico com o título "ylabel vs xlabel", nomeando o eixo x com "xlabel" e o eixo y com "ylabel" e rotacionando os valores do eixo x para melhor visualização.

    st.write(ylabel)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(
        x,
        y,
        color="steelblue",
        linestyle="dashed",
        marker="o",
        markerfacecolor="darkblue",
        markersize=10,
    )
    plt.title(f"{ylabel} vs {xlabel}")
    plt.xlabel(xlabel)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    st.pyplot(fig)

# ------------------------------------------------------------------------------

def plotar_boxplot(y, ylabel):

    # A função "plotar_boxplot", tem como objetivo plotar um gráfico de caixa (box plot) dado um conjunto de dados "y" e uma label "ylabel" para o eixo y.
    # Utiliza a biblioteca "seaborn" para plotar o gráfico e o tamanho da figura é definido como (10,5).
    # A função também imprime o "ylabel" passado como parâmetro.
    # Por fim, utiliza a função "st.pyplot" do Streamlit para exibir o gráfico.

    st.write(ylabel)
    fig = plt.figure(figsize=(10, 5))
    ax = sns.boxplot(y=y)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

# ------------------------------------------------------------------------------

def plotar_todas_as_metricas_na_mesma_figura(perf_scores, parametro, drop_duplicates):

    # Essa função plota todas as métricas (acurácia, recall, precisão e f1 score) em uma única figura.
    # Ela também permite que o usuário escolha se deseja excluir ou não entradas duplicadas no gráfico.
    # Além disso, ela exibe gráficos de caixa (boxplots) individuais para cada métrica e calcula suas médias e desvios padrão.
    # Ela também exige que o usuário especifique qual parâmetro deve ser usado como eixo x no gráfico.

    # CRIAR O DATAFRAME EXCLUINDO OS DUPLICADOS

    perf_scores_df = pd.DataFrame(
        perf_scores, columns=[parametro, "accuracy", "recall", "precision", "f1_score"]
    )
    if drop_duplicates:
        perf_scores_df.drop_duplicates(
            subset=["accuracy", "recall", "precision", "f1_score"],
            keep="first",
            inplace=True,
        )
    perf_scores_df.set_index(parametro, inplace=True)

    st.write("Desempenho")
    st.dataframe(perf_scores_df)
    perf_scores_df.reset_index(inplace=True)

    st.write("Acurácia, Recall, Precisão, F1 score")
    fig = plt.figure(figsize=(12, 6))
    plt.xticks(rotation=45, ha="right")
    # Transforma o dataframe para facilitar plotar todas as métricas na mesma figura
    perf_scores_df_melt = pd.melt(
        perf_scores_df,
        id_vars=[parametro],
        value_vars=["accuracy", "recall", "precision", "f1_score"],
    )
    sns.lineplot(
        data=perf_scores_df_melt,
        x=parametro,
        y="value",
        hue="variable",
        palette="muted",
        marker="o",
    )
    st.pyplot(fig)

    plotar_metrica_individualmente(
        x=perf_scores_df[parametro],
        y=perf_scores_df["accuracy"],
        xlabel=parametro,
        ylabel="Acurácia",
    )
    plotar_metrica_individualmente(
        x=perf_scores_df[parametro],
        y=perf_scores_df["recall"],
        xlabel=parametro,
        ylabel="Recall",
    )
    plotar_metrica_individualmente(
        x=perf_scores_df[parametro],
        y=perf_scores_df["precision"],
        xlabel=parametro,
        ylabel="Precisão",
    )
    plotar_metrica_individualmente(
        x=perf_scores_df[parametro],
        y=perf_scores_df["f1_score"],
        xlabel=parametro,
        ylabel="F1 score",
    )

    # --------------------------------------------------------------------------

    # RECRIAR O DATAFRAME DE CIMA SEM EXCLUIR OS DUPLICADOS

    perf_scores_df = pd.DataFrame(
        perf_scores, columns=[parametro, "accuracy", "recall", "precision", "f1_score"]
    )
    perf_scores_df.set_index(parametro, inplace=True)
    perf_scores_df.reset_index(inplace=True)

    data_container = st.container()
    with data_container:
        coluna_1, coluna_2, coluna_3, coluna_4 = st.columns(4)
        with coluna_1:
            plotar_boxplot(y=perf_scores_df["accuracy"], ylabel="Acurácia")
        with coluna_2:
            plotar_boxplot(y=perf_scores_df["precision"], ylabel="Precisão")
        with coluna_3:
            plotar_boxplot(y=perf_scores_df["recall"], ylabel="Recall")
        with coluna_4:
            plotar_boxplot(y=perf_scores_df["f1_score"], ylabel="F1 score")

    # --------------------------------------------------------------------------

    data_container = st.container()
    with data_container:
        coluna_1, coluna_2, coluna_3, coluna_4 = st.columns(4)
        with coluna_1:
            st.metric(
                "Acurácia (Média)",
                round(np.mean(perf_scores_df["accuracy"]), 3),
                delta=None,
                delta_color="normal",
                help=None,
            )
        with coluna_2:
            st.metric(
                "Precisão (Média)",
                round(np.mean(perf_scores_df["precision"]), 3),
                delta=None,
                delta_color="normal",
                help=None,
            )
        with coluna_3:
            st.metric(
                "Recall (Média)",
                round(np.mean(perf_scores_df["recall"]), 3),
                delta=None,
                delta_color="normal",
                help=None,
            )
        with coluna_4:
            st.metric(
                "F1 score (Média)",
                round(np.mean(perf_scores_df["f1_score"]), 3),
                delta=None,
                delta_color="normal",
                help=None,
            )

# ------------------------------------------------------------------------------

def plota_heatmap_probabilidades_classe(df_proba, classes, n_classes):

    # Essa função plota uma heatmap das probabilidades para cada classe para cada amostra.
    # Ela recebe como entrada o dataframe com as probabilidades, as classes e o número de classes.
    # Ela reseta o índice do dataframe, remove as entradas duplicadas, e as transpõe.
    # Depois, é criada uma figura e é plotado uma heatmap das probabilidades utilizando a biblioteca seaborn.
    # Os ticks das classes são definidos de acordo com o número de classes e o índice do dataframe.
    # Os rótulos dos eixos x e y são definidos como "Probabilidade" e "Classes" respectivamente.
    # Por fim, a função mostra a figura e o dataframe transposto.

    st.subheader(f"HEATMAP COM AS PROBABILIDADE PARA AS CLASSES")

    df_proba.reset_index(drop=True, inplace=True)
    df_proba.drop_duplicates(subset=classes, keep="first", inplace=True)
    df_proba = df_proba.transpose()

    fig, ax = plt.subplots(figsize=(16, 16))
    ax = sns.heatmap(df_proba, vmin=0, vmax=1, cmap="YlGnBu")
    # https://www.programcreek.com/python/example/102350/matplotlib.pyplot.yticks
    tick_marks = np.arange(n_classes)
    plt.yticks(tick_marks, list(df_proba.index.values))
    plt.xlabel("Probabilidade")
    plt.ylabel("Classes")
    plt.show()
    st.pyplot(fig)

    st.dataframe(df_proba)

# ------------------------------------------------------------------------------

def plota_heatmap_probabilidade_classe_pos_label(df2, pos_label):

    # Essa função plota uma heatmap das probabilidades de cada instância de teste para ser classificada como a classe especificada em pos_label.
    # Ela recebe como entrada o dataframe df2 com as probabilidades previstas pelo classificador e o rótulo da classe positiva pos_label.
    # Ela remove os duplicados e transpõe o dataframe para facilitar a plotagem da heatmap.
    # Ela usa o método heatmap() do seaborn para plotar a heatmap, com as probabilidades variando entre 0 e 1 e usando a paleta "YlGnBu".
    # Ela também adiciona rótulos e títulos ao gráfico e exibe o dataframe transposto.

    st.subheader(f"HEATMAP COM A PROBABILIDADE PARA A CLASSE {pos_label}")

    df2.reset_index(drop=True, inplace=True)
    df2 = df2.transpose()

    # Visualizando as probabilidades para a classe ???.
    st.write(f"Quanto mais próximo de 1, mais provável de ser da classe {pos_label}.")

    fig, ax = plt.subplots(figsize=(16, 16))
    ax = sns.heatmap(df2, vmin=0, vmax=1, cmap="YlGnBu")
    tick_marks = np.arange(df2.shape[0])
    plt.yticks(tick_marks, list(df2.index.values))
    plt.xlabel("Árvores de decisão")
    plt.ylabel("Instâncias de teste")
    plt.show()
    st.pyplot(fig)

    st.dataframe(df2)

# ------------------------------------------------------------------------------

def plota_comparacao_treino_teste(perf_train, perf_test, parametro):

    # A função plota_comparacao_treino_teste compara o desempenho de um modelo nos conjuntos de dados de treinamento e teste.
    # Leva em 3 entradas: perf_train, perf_test e parametro.
    # perf_train e perf_test são listas que contêm as métricas de desempenho para os conjuntos de dados de treinamento e teste, respectivamente.
    # O parâmetro é uma string que representa o parâmetro que está sendo ajustado.

    # A função começa criando dois quadros de dados, um para o desempenho no conjunto de dados de treinamento e outro para o desempenho no conjunto de dados de teste.
    # Em seguida, chama a função plot_acc_vs_hypervalues ​​quatro vezes, uma vez para cada métrica de desempenho (exatidão, recuperação, precisão e f1_score) e
    # passa os respectivos valores y1 e y2 para o conjunto de dados de treinamento e teste, os valores x que representam o parâmetro que está sendo ajustado e
    # o xlabel e ylabel.

    # Em seguida, ele cria um contêiner com duas colunas, a primeira coluna exibe as métricas de desempenho do conjunto de dados de treinamento em
    # um dataframe e a segunda coluna exibe as métricas de desempenho do conjunto de dados de teste em um dataframe.

    # No geral, a função permite uma comparação do desempenho do modelo nos conjuntos de dados de treinamento e teste e
    # como o desempenho muda em relação ao valor do parâmetro que está sendo ajustado.

    perf_train_df = pd.DataFrame(
        perf_train, columns=[parametro, "accuracy", "recall", "precision", "f1_score"]
    )
    perf_test_df = pd.DataFrame(
        perf_test, columns=[parametro, "accuracy", "recall", "precision", "f1_score"]
    )

    plot_acc_vs_hypervalues(
        y1=perf_train_df["accuracy"],
        y2=perf_test_df["accuracy"],
        x=perf_train_df[parametro],
        xlabel=parametro,
        ylabel="Acurácia",
    )
    plot_acc_vs_hypervalues(
        y1=perf_train_df["recall"],
        y2=perf_test_df["recall"],
        x=perf_train_df[parametro],
        xlabel=parametro,
        ylabel="Recall",
    )
    plot_acc_vs_hypervalues(
        y1=perf_train_df["precision"],
        y2=perf_test_df["precision"],
        x=perf_train_df[parametro],
        xlabel=parametro,
        ylabel="Precisão",
    )
    plot_acc_vs_hypervalues(
        y1=perf_train_df["f1_score"],
        y2=perf_test_df["f1_score"],
        x=perf_train_df[parametro],
        xlabel=parametro,
        ylabel="F1 score",
    )

    # --------------------------------------------------------------------------

    data_container = st.container()
    with data_container:
        coluna_1, coluna_2 = st.columns(2)
        with coluna_1:
            st.write("TREINAMENTO")
            perf_train_df = pd.DataFrame(
                perf_train,
                columns=[parametro, "accuracy", "recall", "precision", "f1_score"],
            )
            perf_train_df.drop(index=perf_train_df.index[0], axis=0, inplace=True)
            perf_train_df.drop_duplicates(
                subset=["accuracy", "recall", "precision", "f1_score"],
                keep="first",
                inplace=True,
            )
            perf_train_df.set_index(parametro, inplace=True)
            st.dataframe(perf_train_df)

            perf_train_df2 = pd.DataFrame(
                perf_train,
                columns=[
                    parametro,
                    "accuracy_train",
                    "recall_train",
                    "precision_train",
                    "f1_score_train",
                ],
            )
            perf_train_df2.drop(index=perf_train_df2.index[0], axis=0, inplace=True)
            perf_train_df2.drop_duplicates(
                subset=[
                    "accuracy_train",
                    "recall_train",
                    "precision_train",
                    "f1_score_train",
                ],
                keep="first",
                inplace=True,
            )
            perf_train_df2.set_index(parametro, inplace=True)

        with coluna_2:
            st.write("TESTE")
            perf_test_df = pd.DataFrame(
                perf_test,
                columns=[parametro, "accuracy", "recall", "precision", "f1_score"],
            )
            perf_test_df.drop(index=perf_test_df.index[0], axis=0, inplace=True)
            perf_test_df.drop_duplicates(
                subset=["accuracy", "recall", "precision", "f1_score"],
                keep="first",
                inplace=True,
            )
            perf_test_df.set_index(parametro, inplace=True)
            st.dataframe(perf_test_df)

            perf_test_df2 = pd.DataFrame(
                perf_test,
                columns=[
                    parametro,
                    "accuracy_test",
                    "recall_test",
                    "precision_test",
                    "f1_score_test",
                ],
            )
            perf_test_df2.drop(index=perf_test_df2.index[0], axis=0, inplace=True)
            perf_test_df2.drop_duplicates(
                subset=[
                    "accuracy_test",
                    "recall_test",
                    "precision_test",
                    "f1_score_test",
                ],
                keep="first",
                inplace=True,
            )
            perf_test_df2.set_index(parametro, inplace=True)

    # --------------------------------------------------------------------------

    df_merge = pd.merge(
        perf_train_df2, perf_test_df2, left_index=True, right_index=True
    )
    st.dataframe(df_merge)

    # --------------------------------------------------------------------------

    st.write(f"MELHOR {parametro}")
    data_container = st.container()
    with data_container:
        coluna_1, coluna_2, coluna_3, coluna_4 = st.columns(4)
        with coluna_1:
            st.write("ACURÁCIA")
            df1 = df_merge[["accuracy_train", "accuracy_test"]]
            df1 = df1.loc[df1["accuracy_test"].idxmax()]
            st.dataframe(df1)
        with coluna_2:
            st.write("RECALL")
            df2 = df_merge[["recall_train", "recall_test"]]
            df2 = df2.loc[df2["recall_test"].idxmax()]
            st.dataframe(df2)
        with coluna_3:
            st.write("PRECISÃO")
            df3 = df_merge[["precision_train", "precision_test"]]
            df3 = df3.loc[df3["precision_test"].idxmax()]
            st.dataframe(df3)
        with coluna_4:
            st.write("F1 SCORE")
            df4 = df_merge[["f1_score_train", "f1_score_test"]]
            df4 = df4.loc[df4["f1_score_test"].idxmax()]
            st.dataframe(df4)

    # --------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
