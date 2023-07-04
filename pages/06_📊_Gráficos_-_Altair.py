
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import altair as alt
import nx_altair as nxa
import networkx as nx
from googletrans import Translator
translator = Translator()

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

st.markdown("# Gráficos - Altair")

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

# Dados
df_numericos = df.select_dtypes(include=['float64', 'int16', 'int32', 'int64', 'float16', 'float32']) # Dados Numéricos
df_categoricos = df.select_dtypes(include=['object']) # Dados Categóricos

# ------------------------------------------------------------------------------
# Gráficos
# ------------------------------------------------------------------------------

# https://altair-viz.github.io/gallery/index.html

# ------------------------------------------------------------------------------

if (set(['Year']).issubset(df.columns)):
  df.rename(columns = {'Year':'year'}, inplace = True)
if (set(['YEAR']).issubset(df.columns)):
  df.rename(columns = {'YEAR':'year'}, inplace = True)

if (set(['year']).issubset(df.columns)):
  year_max = df['year'].loc[df['year'].idxmax()] # Maximum in column
  year_min = df['year'].loc[df['year'].idxmin()] # Minimum in column
  select_year = alt.selection_single(
    name='Select', fields=['year'], init={'year': year_min},
    bind=alt.binding_range(min=year_min, max=year_max, step=1)
  )

# ------------------------------------------------------------------------------

if st.checkbox('HISTOGRAMA'):
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_histograma")
  mychart = alt.Chart(df).mark_bar().encode(
    alt.X(y_column, bin=True),
    y='count()',
    tooltip=df.columns.tolist()
  )
  if (set(['year']).issubset(df.columns)):
    st.altair_chart(mychart.add_selection(select_year).transform_filter(select_year).interactive(), use_container_width=True)
  else:
    st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------

if st.checkbox('GRÁFICO DE PIZZA'):
  x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(include=['object']).columns, key = "x_grafico_de_pizza")
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_grafico_de_pizza")
  mychart = alt.Chart(df).mark_arc().encode(
  theta=alt.Theta(field=y_column, type="quantitative"),
  color=alt.Color(field=x_column, type="nominal"),
  tooltip=df.columns.tolist()
  )
  if (set(['year']).issubset(df.columns)):
    st.altair_chart(mychart.add_selection(select_year).transform_filter(select_year).interactive(), use_container_width=True)
  else:
    st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------

if st.checkbox('GRÁFICO DE PIZZA COM LABELS'):
  x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(include=['object']).columns, key = "x_grafico_de_pizza_com_labels")
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_grafico_de_pizza_com_labels")
  base = alt.Chart(df).encode(
  theta=alt.Theta(y_column, stack=True), color=alt.Color(x_column, legend=None), tooltip=df.columns.tolist()
  )
  pie = base.mark_arc(outerRadius=120)
  text = base.mark_text(radius=140, size=20).encode(text=x_column)
  mychart = pie + text
  if (set(['year']).issubset(df.columns)):
    st.altair_chart(mychart.add_selection(select_year).transform_filter(select_year).interactive(), use_container_width=True)
  else:
    st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------

if st.checkbox('QUANTILE-QUANTILE PLOT'):
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_quantile_quantile_plot")

  base = alt.Chart(df).transform_quantile(
    y_column,
    step=0.01,
    as_ = ['p', 'v']
  ).transform_calculate(
  uniform = 'quantileUniform(datum.p)',
  normal = 'quantileNormal(datum.p)'
  ).mark_point().encode(
    alt.Y('v:Q')
  )
  mychart = base.encode(x='uniform:Q') | base.encode(x='normal:Q')
  if (set(['year']).issubset(df.columns)):
    st.altair_chart(mychart.add_selection(select_year).transform_filter(select_year), use_container_width=True)
  else:
    st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------

if st.checkbox('RADIAL CHART'):
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_radial_chart")

  base = alt.Chart(df).encode(
    theta=alt.Theta(y_column, stack=True),
    radius=alt.Radius(y_column, scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
    color=y_column,
    tooltip=df.columns.tolist()
  )
  c1 = base.mark_arc(innerRadius=20, stroke="#fff")
  c2 = base.mark_text(radiusOffset=10).encode(text=y_column)
  mychart = c1 + c2
  if (set(['year']).issubset(df.columns)):
    st.altair_chart(mychart.add_selection(select_year).transform_filter(select_year).interactive(), use_container_width=True)
  else:
    st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------

if st.checkbox('DONUT CHART'):
  x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(include=['object']).columns, key = "x_donut_chart")
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_donut_chart")

  mychart = alt.Chart(df).mark_arc(innerRadius=50).encode(
    theta=alt.Theta(field=y_column, type="quantitative"),
    color=alt.Color(field=x_column, type="nominal"),
    tooltip=df.columns.tolist()
  )
  if (set(['year']).issubset(df.columns)):
    st.altair_chart(mychart.add_selection(select_year).transform_filter(select_year).interactive(), use_container_width=True)
  else:
    st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------

if st.checkbox('GRÁFICO DE ÁREA'):
  x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(include=['object']).columns, key = "x_grafico_de_area")
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_grafico_de_area")
  a = alt.Chart(df).mark_area(opacity=1).encode(x=x_column, y=y_column)
  b = alt.Chart(df).mark_area(opacity=0.6).encode(x=x_column, y=y_column)
  c = alt.layer(a, b)
  mychart = c
  if (set(['year']).issubset(df.columns)):
    st.altair_chart(mychart.add_selection(select_year).transform_filter(select_year).interactive(), use_container_width=True)
  else:
    st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------

if st.checkbox('GRÁFICO DE BARRAS'):
  x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(include=['object']).columns, key = "x_grafico_de_barras")
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_grafico_de_barras")
  mychart = alt.Chart(df).mark_bar().encode(
  x=x_column,
  y=y_column,
  tooltip=df.columns.tolist()
  )
  if (set(['year']).issubset(df.columns)):
    st.altair_chart(mychart.add_selection(select_year).transform_filter(select_year).interactive(), use_container_width=True)
  else:
    st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------

if st.checkbox('MAPA DE CALOR'):
  x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(include=['object']).columns, key = "x_mapa_calor")
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_mapa_calor")
  mychart = alt.Chart(df).mark_rect().encode(
  x=x_column,
  y=y_column,
  tooltip=df.columns.tolist()
  )
  if (set(['year']).issubset(df.columns)):
    st.altair_chart(mychart.add_selection(select_year).transform_filter(select_year).interactive(), use_container_width=True)
  else:
    st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------

if st.checkbox('BOXPLOT'):
  x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(include=['object']).columns, key = "x_boxplot")
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_boxplot")
  mychart = alt.Chart(df).mark_boxplot().encode(
  x=x_column,
  y=y_column,
  tooltip=df.columns.tolist()
  )
  st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------

if st.checkbox('SCATTER MATRIX'):
  rows=df_numericos.columns.tolist()
  columns=rows[::-1]
  mychart = alt.Chart(df).mark_circle().encode(
      alt.X(alt.repeat("column"), type='quantitative'),
      alt.Y(alt.repeat("row"), type='quantitative'),
      tooltip=df.columns.tolist()
  ).properties(
      width=150,
      height=150
  ).repeat(
      row=rows,
      column=columns
  )

  if (set(['year']).issubset(df.columns)):
    st.altair_chart(mychart.add_selection(select_year).transform_filter(select_year).interactive(), use_container_width=True)
  else:
    st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------

if st.checkbox('SCATTER PLOT'):

  x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes('number').columns, key = "x_scatter_plot")
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_scatter_plot")
  mychart = alt.Chart(df).mark_point().encode(x=x_column, y=y_column, tooltip=df.columns.tolist())
  if (set(['year']).issubset(df.columns)):
    st.altair_chart(mychart.add_selection(select_year).transform_filter(select_year).interactive(), use_container_width=True)
  else:
    st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------

if st.checkbox('GRÁFICO DE REDE'):

  # https://github.com/Zsailer/nx_altair/blob/master/examples/nx_altair-tutorial.ipynb
  # https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_pandas_edgelist.html
  # https://towardsdatascience.com/customizing-networkx-graphs-f80b4e69bedf
  # https://infovis.fh-potsdam.de/tutorials/infovis7networks.html

  x_column = st.selectbox('Selecione o atributo x', df.select_dtypes(include=['object']).columns, key = "x_grafico_de_rede")
  y_column = st.selectbox('Selecione o atributo y', df.select_dtypes(include=['object']).columns, key = "y_grafico_de_rede")
  layout = st.selectbox('Layout', ['Layout Kamada Kawai','Layout Circular','Layout Aleatório','Layout Concha','Layout Primavera','Layout Spectral','Layout Espiral'], key = "layout_grafico_de_rede")

  G = nx.from_pandas_edgelist(df, source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())

  density = nx.density(G)
  st.write(f"Informações: {translator.translate(nx.info(G), src='en', dest='pt').text}")
  st.write("Densidade da rede:", density)

  if layout == "Layout Circular":
    pos = nx.circular_layout(G)

  elif layout == "Layout Kamada Kawai":
    pos = nx.kamada_kawai_layout(G)

  elif layout == "Layout Aleatório":
    pos = nx.random_layout(G)

  elif layout == "Layout Concha":
    pos = nx.shell_layout(G)

  elif layout == "Layout Primavera":
    pos = nx.spring_layout(G)

  elif layout == "Layout Spectral":
    pos = nx.spectral_layout(G)

  elif layout == "Layout Espiral":
    pos = nx.spiral_layout(G)

  else:
    pos = nx.spring_layout(G)

  d = {v:v for v in list(G.nodes())}
  degrees = dict(G.degree(G.nodes()))
  between = nx.betweenness_centrality(G)
  nx.set_node_attributes(G, d, 'name')
  nx.set_node_attributes(G, degrees, 'degree') # save the degrees as a node attribute
  nx.set_node_attributes(G, between, 'between') # save the between as a node attribute
  mychart = nxa.draw_networkx(G=G, pos=pos, node_size='degree:Q', node_color='degree:N', node_tooltip=["name", "degree", "between"], cmap='viridis', linewidths=0,).properties(width=500, height=500)
  st.altair_chart(mychart, use_container_width=True)

  # ------------------------------------------------------------------------

  sorted_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True) # reverse sort of the degrees
  sorted_between = sorted(between.items(), key=lambda x: x[1], reverse=True) # reverse sort of the between

  # ------------------------------------------------------------------------

  df_sorted_degree = pd.DataFrame(sorted_degree, columns=['Nó','N_Vizinhos'])

  mychart = alt.Chart(df_sorted_degree.head(10)).mark_bar().encode(
  x='N_Vizinhos:Q',
  y=alt.Y('Nó:N', sort='-x'),
    color=alt.Color('N_Vizinhos:Q', scale=alt.Scale(scheme='viridis')),
    tooltip=df_sorted_degree.columns.tolist()
  ).interactive()

  #https://discuss.streamlit.io/t/how-to-display-a-table-and-its-plot-side-by-side-with-an-adjusted-height/30214
  data_container = st.container()
  with data_container:
    table, plot = st.columns(2)
    with table:
      st.table(df_sorted_degree.head(10))
    with plot:
      st.altair_chart(mychart, use_container_width=True)

  # ------------------------------------------------------------------------

  df_sorted_between = pd.DataFrame(sorted_between, columns=['Nó','N_Between'])

  mychart = alt.Chart(df_sorted_between.head(10)).mark_bar().encode(
    x='N_Between:Q',
    y=alt.Y('Nó:N', sort='-x'),
    color=alt.Color('N_Between:Q', scale=alt.Scale(scheme='viridis')),
    tooltip=df_sorted_between.columns.tolist()
  ).interactive()

  #https://discuss.streamlit.io/t/how-to-display-a-table-and-its-plot-side-by-side-with-an-adjusted-height/30214
  data_container = st.container()
  with data_container:
    table, plot = st.columns(2)
  with table:
    st.table(df_sorted_between.head(10))
  with plot:
    st.altair_chart(mychart, use_container_width=True)

  # ------------------------------------------------------------------------
