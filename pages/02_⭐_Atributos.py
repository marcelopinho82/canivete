
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import altair as alt

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

st.markdown("# Atributos")

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
# https://www.visual-design.net/post/semi-automated-exploratory-data-analysis-process-in-python
df_datatypes = pd.DataFrame(df.dtypes)

# ------------------------------------------------------------------------------

st.write("ATRIBUTOS")
df_datatypes = pd.DataFrame({"Atributo": df.columns, "Tipo": df.dtypes.values})
st.dataframe(df_datatypes)

# ------------------------------------------------------------------------------

df_numericos = df.select_dtypes(exclude=['object']) # Dados Numéricos
df_categoricos = df.select_dtypes(include=['object']) # Dados Categóricos

if not df_numericos.empty:

  if st.checkbox('ATRIBUTOS NUMÉRICOS'):

    st.write(pd.DataFrame(df_numericos.describe().round(decimals=2).transpose()))

    for col in df.select_dtypes(include=['int16', 'int32', 'int64']):
      st.write(f"No atributo \"{col}\": mínima é de {df[col].min():.0f} e a máxima é de {df[col].max():.0f}.")
      st.write(f"\n")

      # https://altair-viz.github.io/gallery/simple_histogram.html

      mychart = alt.Chart(df_numericos).mark_bar().encode(alt.X(field=col, type='quantitative'), y='count()')
      st.altair_chart(mychart, use_container_width=True)

      mychart = alt.Chart(df_numericos).mark_bar().encode(alt.X(field=col, type='quantitative', bin=alt.Bin(maxbins=15)), y='count()')
      st.altair_chart(mychart, use_container_width=True)

    for col in df.select_dtypes(include=['float16', 'float32', 'float64']):
      st.write(f"No atributo \"{col}\": mínima é de {df[col].min():.0f} e a máxima é de {df[col].max():.0f}, com uma média de {df[col].mean():.2f} e desvio padrão de {df[col].std():.2f}.")
      st.write(f"\n")

      # https://altair-viz.github.io/gallery/simple_histogram.html

      mychart = alt.Chart(df_numericos).mark_bar().encode(alt.X(field=col, type='quantitative'), y='count()')
      st.altair_chart(mychart, use_container_width=True)

      mychart = alt.Chart(df_numericos).mark_bar().encode(alt.X(field=col, type='quantitative', bin=alt.Bin(maxbins=15)), y='count()')
      st.altair_chart(mychart, use_container_width=True)

    # --------------------------------------------------------------------------

  if st.checkbox('SCATTER PLOT'):

    x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(exclude=['object']).columns.tolist(), key = "x_scatter_plot")
    y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes(exclude=['object']).columns.tolist(), key = "y_scatter_plot")
    mychart = alt.Chart(df).mark_point().encode(x=x_column, y=y_column, tooltip=df.columns.tolist())
    st.altair_chart(mychart.interactive(), use_container_width=True)

if st.checkbox('SCATTER PLOT COM TARGET'):

  x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(exclude=['object']).columns.tolist(), key = "x_scatter_plot_com_target")
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes(exclude=['object']).columns.tolist(), key = "y_scatter_plot_com_target")

  if not df_categoricos.empty:
    target = st.selectbox('Selecione a coluna alvo (target)', df.select_dtypes(include=['object']).columns.tolist(), key = "scatter_plot_target_categorico")
    mychart = alt.Chart(df).mark_point().encode(x=x_column, y=y_column, color=alt.Column(field=target, type="nominal"), shape=alt.Column(field=target, type="nominal"), tooltip=df.columns.tolist())
    st.altair_chart(mychart.interactive(), use_container_width=True)
  elif not df_numericos.empty:
    target = st.selectbox('Selecione a coluna alvo (target)', df.select_dtypes(exclude=['object']).columns.tolist(), key = "scatter_plot_target_numerico")
    mychart = alt.Chart(df).mark_point().encode(x=x_column, y=y_column, color=alt.Column(field=target, type="nominal"), shape=alt.Column(field=target, type="nominal"), tooltip=df.columns.tolist())
    st.altair_chart(mychart.interactive(), use_container_width=True)

  if (set(['target']).issubset(df.columns)):
    target_max = df['target'].loc[df['target'].idxmax()] # Maximum in column
    target_min = df['target'].loc[df['target'].idxmin()] # Minimum in column
    select_target = alt.selection_single(name='Select', fields=['target'], init={'target': target_min}, bind=alt.binding_range(min=target_min, max=target_max, step=1))
    mychart = alt.Chart(df).mark_point().encode(x=x_column, y=y_column, color=alt.Column(field='target', type="nominal"), shape=alt.Column(field='target', type="nominal"), tooltip=df.columns.tolist())
    st.altair_chart(mychart.add_selection(select_target).transform_filter(select_target).interactive(), use_container_width=True)

# ------------------------------------------------------------------------------

if not df_categoricos.empty:

  if st.checkbox('ATRIBUTOS CATEGÓRICOS'):

    st.write(pd.DataFrame(df_categoricos.describe().round(decimals=2).transpose()))

    for col in df.select_dtypes(include=['object']).columns.tolist():

      # https://stackoverflow.com/questions/35392417/pandas-frequency-of-column-values
      modes = df[col].mode()
      modas = {item: len(df[col][df[col].isin(modes[modes == item])]) for item in modes}
      moda = list(modas.keys())
      freqmoda = list(modas.values())

      st.write(f"O atributo \"{col}\" possui {df[col].nunique()} valores únicos. A moda é \"{moda}\" com frequência {freqmoda}.")
      st.write(f"\n")

      # https://altair-viz.github.io/gallery/simple_histogram.html
      mychart = alt.Chart(df_categoricos).mark_bar().encode(alt.X(field=col, type='nominal'), y='count()')
      st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------
