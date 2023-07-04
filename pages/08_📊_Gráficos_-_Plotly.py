
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

import pycountry

countries = {}
for Country in pycountry.countries:
  countries[Country.name] = Country.alpha_3

# ------------------------------------------------------------------------------

def ajustaCodeCountryISO(df, campo):
  df[campo + 'ISO'] = df[campo]
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Ivory Coast', 'Côte d\'Ivoire')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Zaire', 'Congo, The Democratic Republic of the')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Bolivia', 'Bolivia, Plurinational State of')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Iran', 'Iran, Islamic Republic of')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Russian', 'Russian Federation')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Scotland', 'United Kingdom')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('England', 'United Kingdom')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Wales', 'United Kingdom')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Northern Ireland', 'United Kingdom')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('USA', 'United States')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Korea', 'Korea, Republic of')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Germany FR', 'Germany')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('East Germany', 'Germany')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('West Germany', 'Germany')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Czechoslovakia', 'Czechia')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Soviet Union', 'Russian Federation')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Yugoslavia', 'Serbia')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Federal Republic of Yugoslavia', 'Serbia')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Kingdom of Yugoslavia', 'Serbia')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Socialist Federal Republic of Yugoslavia', 'Serbia')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Korea Republic', 'Korea, Republic of')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('South Korea', 'Korea, Republic of')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('North Korea', 'Korea, Democratic People\'s Republic of')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Moldova', 'Moldova, Republic of')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Taiwan', 'Taiwan, Province of China')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Tanzania', 'Tanzania, United Republic of')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Venezuela', 'Venezuela, Bolivarian Republic of')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Vietnam', 'Viet Nam')

  df[campo + '_Code'] = [countries.get(country, 'Unknown code') for country in df[campo + 'ISO']]

  return(df)

# ------------------------------------------------------------------------------

st.markdown("# Gráficos - Plotly")

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
# Gráficos
# ------------------------------------------------------------------------------

# ???

# ------------------------------------------------------------------------------

if (set(['country']).issubset(df.columns)) or (set(['Country']).issubset(df.columns)) or (set(['COUNTRY']).issubset(df.columns)):

  if st.checkbox('MAPA'):

    atributos = []

    if (set(['country']).issubset(df.columns)):
      df = ajustaCodeCountryISO(df, 'country')
      atributos.append("country")

    if (set(['Country']).issubset(df.columns)):
      df = ajustaCodeCountryISO(df, 'Country')
      atributos.append("Country")

    if (set(['COUNTRY']).issubset(df.columns)):
      df = ajustaCodeCountryISO(df, 'COUNTRY')
      atributos.append("COUNTRY")

    x_column = st.selectbox('Selecione a coluna do eixo x', atributos, key = "x_map_ploty")

    fig = px.choropleth(
      df.sort_values(x_column),
      locations=x_column + "_Code",
      color=x_column,
      hover_name=x_column,
      hover_data = df.columns.tolist(),
      color_continuous_scale=px.colors.sequential.Plasma[::-1],
    )

    fig.show()
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
