
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

st.markdown("# Gráficos - Streamlit")

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

# https://docs.streamlit.io/library/api-reference/charts

# ------------------------------------------------------------------------------

# https://docs.streamlit.io/library/api-reference/charts/st.line_chart

if st.checkbox('GRÁFICO DE LINHA'):
  st.line_chart(data=df_numericos, x=None, y=None, width=0, height=0, use_container_width=True)
  x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(include=['object']).columns, key = "x_grafico_de_linha")
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_grafico_de_linha")
  st.line_chart(data=df, x=x_column, y=y_column, width=0, height=0, use_container_width=True)

# ------------------------------------------------------------------------------

# https://docs.streamlit.io/library/api-reference/charts/st.area_chart

if st.checkbox('GRÁFICO DE ÁREA'):
  st.area_chart(data=df_numericos, x=None, y=None, width=0, height=0, use_container_width=True)
  x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(include=['object']).columns, key = "x_grafico_de_area")
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_grafico_de_area")
  st.area_chart(data=df, x=x_column, y=y_column, width=0, height=0, use_container_width=True)

# ------------------------------------------------------------------------------

# https://docs.streamlit.io/library/api-reference/charts/st.bar_chart

if st.checkbox('GRÁFICO DE BARRAS'):
  st.bar_chart(data=df_numericos, x=None, y=None, width=0, height=0, use_container_width=True)
  x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(include=['object']).columns, key = "x_grafico_de_barras")
  y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_grafico_de_barras")
  st.bar_chart(data=df, x=x_column, y=y_column, width=0, height=0, use_container_width=True)

# ------------------------------------------------------------------------------

# https://docs.streamlit.io/library/api-reference/charts/st.map
# https://stackoverflow.com/questions/24870306/how-to-check-if-a-column-exists-in-pandas
# https://www.geeksforgeeks.org/how-to-rename-columns-in-pandas-dataframe/#:~:text=One%20way%20of%20renaming%20the,which%20are%20to%20be%20renamed.

if (set(['LAT','LON']).issubset(df.columns)):
  df.rename(columns = {'LAT':'lat'}, inplace = True)
  df.rename(columns = {'LON':'lon'}, inplace = True)

if (set(['LATITUDE','LONGITUDE']).issubset(df.columns)):
  df.rename(columns = {'LATITUDE':'latitude'}, inplace = True)
  df.rename(columns = {'LONGITUDE':'longitude'}, inplace = True)

# Must have columns called 'lat', 'lon', 'latitude', or 'longitude'.
if (set(['lat','lon']).issubset(df.columns)) or (set(['latitude','longitude']).issubset(df.columns)):
  if st.checkbox('MAPA'):
    st.map(data=df, zoom=None, use_container_width=True)

# ------------------------------------------------------------------------------
