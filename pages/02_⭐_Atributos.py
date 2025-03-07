
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import altair as alt
import nx_altair as nxa
import networkx as nx

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
  select_year = alt.selection_point(
    name='Select', fields=['year'], init={'year': year_min},
    bind=alt.binding_range(min=year_min, max=year_max, step=1)
)

# ------------------------------------------------------------------------------

if df.select_dtypes(include=['number']).columns.tolist():

    st.subheader('Atributos numéricos')
    
    # Selecionar colunas desejadas
    selected_cols = st.multiselect("Selecione as colunas numéricas:", df.select_dtypes(include=['number']).columns.tolist())
   
    if selected_cols:
        
        # Calcular estatísticas básicas para as colunas selecionadas
        df_stats = pd.DataFrame({
            'Count': df[selected_cols].count(),
            'Valores Únicos': df[selected_cols].nunique(),
            'Min': df[selected_cols].min(),
            '25%': df[selected_cols].quantile(0.25),
            'Mediana (50%)': df[selected_cols].quantile(0.50),
            '75%': df[selected_cols].quantile(0.75),
            'Max': df[selected_cols].max(),
            'Média': df[selected_cols].mean(),
            'Desvio Padrão': df[selected_cols].std()
        })

        # Exibir as estatísticas no Streamlit
        st.write("Estatísticas das colunas numéricas selecionadas:")
        st.write(df_stats.round(decimals=2).transpose())

        st.write("O histograma exibe a distribuição de uma variável numérica em intervalos ou 'bins'. Ele mostra quantas vezes os valores caem em cada intervalo. Barras altas indicam que muitos dados estão concentrados naquele intervalo, enquanto barras baixas indicam que os valores são menos frequentes. Útil para identificar padrões como simetria, assimetria ou picos nos dados.")

        for col in selected_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            std_val = df[col].std()

            if df[col].dtype in ['int16', 'int32', 'int64']:
                st.write(f"No atributo \"{col}\": mínima é de {min_val:.0f} e a máxima é de {max_val:.0f}.")
                st.write(f"\n")
                
            elif df[col].dtype in ['float16', 'float32', 'float64']:
                st.write(f"No atributo \"{col}\": mínima é de {min_val:.2f}, máxima é de {max_val:.2f}, média é de {mean_val:.2f}, e desvio padrão de {std_val:.2f}.")
                st.write(f"\n")

            # Histograma simples
            mychart = alt.Chart(df.select_dtypes(exclude=['object'])).mark_bar().encode(
                alt.X(field=col, type='quantitative'), 
                y='count()'
            )

            if (set(['year']).issubset(df.columns)):
                st.altair_chart(mychart.add_params(select_year).transform_filter(select_year).interactive(), use_container_width=True)
            else:
                st.altair_chart(mychart, use_container_width=True)            

            # Histograma com binning
            mychart_binned = alt.Chart(df.select_dtypes(exclude=['object'])).mark_bar().encode(
                alt.X(field=col, type='quantitative', bin=alt.Bin(maxbins=15)),
                y='count()'
            )

            if (set(['year']).issubset(df.columns)):
                st.altair_chart(mychart_binned.add_params(select_year).transform_filter(select_year).interactive(), use_container_width=True)
            else:
                st.altair_chart(mychart_binned, use_container_width=True)

        if len(selected_cols) > 1:
            if st.checkbox('Matriz de dispersão (scatter matrix)'):
            
                st.write("Exibe pares de gráficos de dispersão entre várias variáveis numéricas simultaneamente. Cada gráfico mostra a correlação entre duas variáveis. Padrões lineares indicam correlação forte, enquanto dispersões aleatórias sugerem pouca ou nenhuma correlação.")
            
                rows=df.select_dtypes(include=['number']).columns.tolist()
                rows=selected_cols
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
                    st.altair_chart(mychart.add_params(select_year).transform_filter(select_year).interactive(), use_container_width=True)
                else:
                    st.altair_chart(mychart, use_container_width=True)
            
    else:
        st.warning("Nenhuma coluna selecionada.")

    # -------------------------------------------------------------------------- 

    if st.checkbox('Gráfico quantil-quantil (q-q plot)'):
    
        st.write("O Q-Q Plot compara a distribuição dos dados com uma distribuição teórica, como uniforme ou normal. Se os pontos formarem uma linha reta, os dados seguem a distribuição esperada. Desvios significam diferenças significativas, indicando dados que podem ser mais dispersos ou concentrados.")
    
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
        mychart1 = base.encode(x='uniform:Q')
        mychart2 = base.encode(x='normal:Q')
        if (set(['year']).issubset(df.columns)):
            st.altair_chart(mychart1.add_params(select_year).transform_filter(select_year), use_container_width=True)
            st.altair_chart(mychart2.add_params(select_year).transform_filter(select_year), use_container_width=True)
        else:
            st.altair_chart(mychart1, use_container_width=True)
            st.altair_chart(mychart2, use_container_width=True)

    if st.checkbox('Gráfico radial'):
    
        st.write("Este gráfico organiza os dados em um círculo, com as fatias representando categorias e o raio mostrando a magnitude. Ele destaca proporções relativas, ajudando a visualizar como cada categoria contribui para o total. Cores distintas facilitam a identificação dos grupos.")
    
        y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_radial_chart")
          
        # Realizar a contagem de ocorrências agrupando por x
        df_grouped = df.groupby(y_column, as_index=False).count()

        base = alt.Chart(df_grouped).encode(
            theta=alt.Theta(y_column, stack=True),
            radius=alt.Radius(y_column, scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
            color=y_column,
            tooltip=df_grouped.columns.tolist()
        )
        c1 = base.mark_arc(innerRadius=20, stroke="#fff")
        c2 = base.mark_text(radiusOffset=10).encode(text=y_column)
        mychart = c1 + c2
        if (set(['year']).issubset(df.columns)):
            st.altair_chart(mychart.add_params(select_year).transform_filter(select_year).interactive(), use_container_width=True)
        else:
            st.altair_chart(mychart, use_container_width=True)
              
        # Exibir a contagem total para referência
        total_count = df[y_column].count()
        st.write(f"**Contagem total de {y_column}:** {total_count}")

# ------------------------------------------------------------------------------

if df.select_dtypes(include=['object', 'category']).columns.tolist():

    st.subheader('Atributos categóricos')
  
    # Selecionar colunas desejadas
    selected_cols = st.multiselect("Selecione as colunas categóricas:", df.select_dtypes(include=['object']).columns.tolist())

    if selected_cols:

        # Calcular estatísticas básicas para as colunas selecionadas
        stats_data = []
        for col in selected_cols:
            moda = df[col].mode()[0]  # Moda
            freq_moda = df[col].value_counts()[moda]  # Frequência da moda
            stats_data.append({
                'Coluna': col,
                'Count': df[col].count(),
                'Valores Únicos': df[col].nunique(),
                'Moda': moda,
                'Frequência da Moda': freq_moda
            })

        # Criar DataFrame com as estatísticas
        df_stats = pd.DataFrame(stats_data)
        
        # Criar DataFrame com as estatísticas e definir a coluna "Coluna" como índice
        df_stats = pd.DataFrame(stats_data).set_index('Coluna')

        # Exibir df_stats no Streamlit
        st.write("Estatísticas das colunas categóricas selecionadas:")
        st.dataframe(df_stats, use_container_width=True)    

        for col in selected_cols:

            # https://stackoverflow.com/questions/35392417/pandas-frequency-of-column-values
            modes = df[col].mode()
            modas = {item: len(df[col][df[col].isin(modes[modes == item])]) for item in modes}
            moda = list(modas.keys())
            freqmoda = list(modas.values())

            st.write(f"O atributo \"{col}\" possui {df[col].nunique()} valores únicos. A moda é \"{moda}\" com frequência {freqmoda}.")

            # https://altair-viz.github.io/gallery/simple_histogram.html
            mychart = alt.Chart(df.select_dtypes(include=['object'])).mark_bar().encode(alt.X(field=col, type='nominal'), y='count()')
            st.altair_chart(mychart, use_container_width=True)

    else:
        st.warning("Nenhuma coluna selecionada.")
            
    if st.checkbox('Visualizando valores únicos de uma coluna'):
    
        # Melt and count unique value occurrences
        df_pre_proc = (
            df[selected_cols]
            .melt(var_name='Atributo', value_name='Valor')
            .value_counts() # Series with counts of each (Atributo, Valor) pair
            .reset_index()  # Turn index into columns
            .sort_values(by=['Atributo', 'count'], ascending=False)
        )        
        
        # Display DataFrame in Streamlit
        st.dataframe(df_pre_proc, use_container_width=True)
        
        csv = df_pre_proc.to_csv(index=False).encode('utf-8')
        st.download_button(label='Download CSV', data=csv, mime='text/csv', file_name=option.split('.')[0] + '_Unique.csv')

# ------------------------------------------------------------------------------

    if st.checkbox('Gráfico de rede'):
    
        st.write("Representa relacionamentos entre nós, como conexões entre elementos ou entidades. A disposição dos nós e a espessura das conexões refletem a proximidade e a força dos relacionamentos. Útil para entender redes complexas como sociais ou hierárquicas.")

        # https://github.com/Zsailer/nx_altair/blob/master/examples/nx_altair-tutorial.ipynb
        # https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_pandas_edgelist.html
        # https://towardsdatascience.com/customizing-networkx-graphs-f80b4e69bedf
        # https://infovis.fh-potsdam.de/tutorials/infovis7networks.html

        x_column = st.selectbox('Selecione o atributo x', df.select_dtypes(include=['object']).columns, key = "x_grafico_de_rede")
        y_column = st.selectbox('Selecione o atributo y', df.select_dtypes(include=['object']).columns, key = "y_grafico_de_rede")
        layout = st.selectbox('Layout', ['Layout Kamada Kawai','Layout Circular','Layout Aleatório','Layout Concha','Layout Primavera','Layout Spectral','Layout Espiral'], key = "layout_grafico_de_rede")
      
        G = nx.from_pandas_edgelist(df, source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())

        density = nx.density(G)
        st.write(f"Informações: Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
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

# ------------------------------------------------------------------------------

if df.select_dtypes(include=['object', 'category']).columns.tolist() and df.select_dtypes(include=['number']).columns.tolist():

    if st.checkbox('Gráfico de pizza ou gráfico de donut (rosquinha)'):
        
        st.write("Representa a proporção de categorias em um conjunto de dados. Cada fatia mostra a parcela de uma categoria em relação ao total. Útil para destacar qual categoria é mais representativa no conjunto, mas funciona melhor com poucas categorias.")

        # Seleção de colunas
        x_column = st.selectbox('Selecione a coluna para categorias', df.select_dtypes(include=['object']).columns, key="x_grafico_de_pizza")
        y_column = st.selectbox('Selecione a métrica (opcional)', ['Contagem'] + list(df.select_dtypes('number').columns), key="y_grafico_de_pizza")

        # Calcular o número de instâncias por categoria
        df_instance_counts = df[x_column].value_counts().reset_index()
        df_instance_counts.columns = [x_column, 'Instâncias']

        # Adicionar slider para filtrar pelo número de instâncias
        min_count, max_count = df_instance_counts['Instâncias'].min(), df_instance_counts['Instâncias'].max()
        instance_threshold = st.slider('Filtrar categorias por número de instâncias', min_count, max_count, (min_count, max_count), step=1, key="slider_grafico_de_pizza")

        # Aplicar filtro no DataFrame agrupado
        filtered_df = df_instance_counts[
            (df_instance_counts['Instâncias'] >= instance_threshold[0]) & 
            (df_instance_counts['Instâncias'] <= instance_threshold[1])
        ]

        # Usar o DataFrame filtrado no gráfico
        if y_column == 'Contagem':
            df_grouped = filtered_df
            metric_column = 'Instâncias'
        else:
            df_grouped = df[df[x_column].isin(filtered_df[x_column])]
            df_grouped = df_grouped.groupby(x_column, as_index=False)[y_column].sum()
            metric_column = y_column

        # Parâmetros adicionais
        inner_radius = st.slider('Ajuste do innerRadius (gráfico de rosquinha)', 0.0, 100.0, 0.0, step=1.0)        

        # Construção do gráfico
        base = alt.Chart(df_grouped).encode(
            theta=alt.Theta(field=metric_column, type="quantitative"),
            color=alt.Color(field=x_column, type="nominal"),
            tooltip=[x_column, metric_column],
        )

        # Aplicação do innerRadius para gráfico de rosquinha
        mychart = base.mark_arc(innerRadius=inner_radius)

        if (set(['year']).issubset(df.columns)):
            st.altair_chart(mychart.add_params(select_year).transform_filter(select_year).interactive(), use_container_width=True)
        else:
            st.altair_chart(mychart, use_container_width=True)

        # Exibir a contagem total (se métrica for contagem)
        if y_column == 'Contagem':
            total_count = df[x_column].count()
            st.write(f"**Contagem total de categorias em {x_column}:** {total_count}")
        else:
            total_sum = df_grouped[metric_column].sum()
            st.write(f"**Soma total de {metric_column}:** {total_sum}")
        
    if st.checkbox('Gráfico de barras'):

        st.write("Mostra a comparação de valores entre categorias. Cada barra representa uma categoria e sua altura reflete o valor associado. Útil para destacar quais categorias possuem maiores ou menores valores.")

        # Seleção de colunas
        x_column = st.selectbox('Selecione a coluna para categorias', df.select_dtypes(include=['object']).columns, key="x_grafico_de_barras")
        y_column = st.selectbox('Selecione a métrica (opcional)', ['Contagem'] + list(df.select_dtypes('number').columns), key="y_grafico_de_barras")

        # Calcular o número de instâncias por categoria
        df_instance_counts = df[x_column].value_counts().reset_index()
        df_instance_counts.columns = [x_column, 'Instâncias']

        # Adicionar slider para filtrar pelo número de instâncias
        min_count, max_count = df_instance_counts['Instâncias'].min(), df_instance_counts['Instâncias'].max()
        instance_threshold = st.slider('Filtrar categorias por número de instâncias', min_count, max_count, (min_count, max_count), step=1, key="slider_grafico_de_barras")

        # Aplicar filtro no DataFrame agrupado
        filtered_df = df_instance_counts[
            (df_instance_counts['Instâncias'] >= instance_threshold[0]) & 
            (df_instance_counts['Instâncias'] <= instance_threshold[1])
        ]

        # Usar o DataFrame filtrado no gráfico
        if y_column == 'Contagem':
            df_grouped = filtered_df
            metric_column = 'Instâncias'
        else:
            df_grouped = df[df[x_column].isin(filtered_df[x_column])]
            df_grouped = df_grouped.groupby(x_column, as_index=False)[y_column].sum()
            metric_column = y_column

        # Construção do gráfico de barras
        bar_chart = alt.Chart(df_grouped).mark_bar().encode(
            x=alt.X(field=x_column, type="nominal", title=x_column),
            y=alt.Y(field=metric_column, type="quantitative", title=metric_column),
            color=alt.Color(field=x_column, type="nominal"),
            tooltip=[x_column, metric_column]
        )

        # Renderizar o gráfico no Streamlit
        st.altair_chart(bar_chart, use_container_width=True)

        # Exibir a contagem total (se métrica for contagem)
        if y_column == 'Contagem':
            total_count = df[x_column].count()
            st.write(f"**Contagem total de categorias em {x_column}:** {total_count}")
        else:
            total_sum = df_grouped[metric_column].sum()
            st.write(f"**Soma total de {metric_column}:** {total_sum}")
        
    if st.checkbox('Gráfico de caixa (boxplot)'):
        
        st.write("Visualiza a distribuição de uma variável numérica, destacando a mediana, os quartis e possíveis outliers. A linha central mostra a mediana, enquanto a caixa abrange o intervalo interquartil. Outliers aparecem como pontos fora dos 'bigodes'.")

        # Seleção de colunas
        x_column = st.selectbox('Selecione a coluna para categorias', df.select_dtypes(include=['object']).columns, key="x_boxplot")
        y_column = st.selectbox('Selecione a variável numérica', df.select_dtypes('number').columns, key="y_boxplot")

        # Calcular o número de instâncias por categoria
        df_instance_counts = df[x_column].value_counts().reset_index()
        df_instance_counts.columns = [x_column, 'Instâncias']

        # Adicionar slider para filtrar pelo número de instâncias
        min_count, max_count = df_instance_counts['Instâncias'].min(), df_instance_counts['Instâncias'].max()
        instance_threshold = st.slider('Filtrar categorias por número de instâncias', min_count, max_count, (min_count, max_count), step=1, key="slider_boxplot")

        # Aplicar filtro no DataFrame
        filtered_df = df[df[x_column].isin(
            df_instance_counts[
                (df_instance_counts['Instâncias'] >= instance_threshold[0]) & 
                (df_instance_counts['Instâncias'] <= instance_threshold[1])
            ][x_column]
        )]

        # Construção do gráfico de caixa
        boxplot_chart = alt.Chart(filtered_df).mark_boxplot().encode(
            x=alt.X(field=x_column, type="nominal", title=x_column),
            y=alt.Y(field=y_column, type="quantitative", title=y_column),
            color=alt.Color(field=x_column, type="nominal")
        )

        # Renderizar o gráfico no Streamlit
        st.altair_chart(boxplot_chart, use_container_width=True)

        # Exibir a contagem total de linhas filtradas
        filtered_count = filtered_df.shape[0]
        st.write(f"**Total de instâncias filtradas:** {filtered_count}")

# ------------------------------------------------------------------------------

if df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist():

    if st.checkbox('Gráfico de área'):
    
        st.write("Mostra a evolução de uma variável numérica ao longo do tempo. A área preenchida sob a curva facilita a visualização das mudanças acumulativas. Útil para destacar tendências ascendentes ou descendentes.")
    
        x_column = st.selectbox('Selecione a coluna do eixo x', df.select_dtypes(include=['datetime', 'datetimetz']).columns, key = "x_grafico_de_area")
        y_column = st.selectbox('Selecione a coluna do eixo y', df.select_dtypes('number').columns, key = "y_grafico_de_area")
        a = alt.Chart(df).mark_area(opacity=1).encode(x=x_column, y=y_column)
        b = alt.Chart(df).mark_area(opacity=0.6).encode(x=x_column, y=y_column)
        c = alt.layer(a, b)
        mychart = c
        if (set(['year']).issubset(df.columns)):
            st.altair_chart(mychart.add_params(select_year).transform_filter(select_year).interactive(), use_container_width=True)
        else:
            st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------

if df.select_dtypes(include=['object', 'category']).columns.tolist() or df.select_dtypes(include=['number']).columns.tolist():

    if st.checkbox('Gráfico de dispersão (scatter plot)'):

        st.write("Mostra a relação entre duas variáveis numéricas. Cada ponto representa uma observação. Padrões como linhas ou curvas sugerem correlação positiva ou negativa. A ausência de padrão indica correlação fraca ou inexistente. Além disso, pode ser utilizada para identificar agrupamentos (clusters) de dados, onde pontos próximos entre si podem indicar grupos com características semelhantes.")

        # Validar colunas disponíveis
        x_columns = df.columns.tolist()

        if not x_columns:
            st.warning("Não há colunas numéricas suficientes para construir o gráfico.")
        else:
            # Select X column
            x_column = st.selectbox('Selecione a coluna do eixo x', x_columns, key="x_scatter_plot_com_target")

            # Create Y column options (exclude selected X column)
            y_columns = [col for col in x_columns if col != x_column]

            if not y_columns:
                st.warning("Nenhuma outra coluna disponível para o eixo Y após a seleção do eixo X.")
            else:
                # Select Y column
                y_column = st.selectbox('Selecione a coluna do eixo y', y_columns, key="y_scatter_plot_com_target")

                # Select Target Column
                target = st.selectbox(
                    'Selecione a coluna alvo (target) ou deixe vazio para mostrar sem destaque:', 
                    [""] + df.columns.tolist(),  # Empty option
                    key="scatter_plot_target"
                )

                # Selecionar se quer aplicar filtro
                apply_filter = st.checkbox('Aplicar filtro ao gráfico com base no target?', value=True)

                # Função auxiliar para criar gráficos
                def create_chart(target_column=None, filter_enabled=False):
                    base = alt.Chart(df).mark_point().encode(
                        x=alt.X(x_column),
                        y=alt.Y(y_column),
                        tooltip=[alt.Tooltip(col) for col in df.columns]
                    )
                    if target_column:
                        if pd.api.types.is_numeric_dtype(df[target_column]):
                            # Numeric target
                            target_max = df[target_column].max()
                            target_min = df[target_column].min()
                            select_target = alt.selection_point(
                                name='Select',
                                fields=[target_column],
                                bind=alt.binding_range(min=target_min, max=target_max, step=1),
                                value=target_min
                            )
                        else:
                            # Categorical target
                            select_target = alt.selection_point(
                                name='Select',
                                fields=[target_column],
                                bind=alt.binding_select(options=df[target_column].unique().tolist()),
                                value=df[target_column].iloc[0]
                            )

                        chart = base.encode(
                            color=alt.Color(f'{target_column}:N'),
                            shape=alt.Shape(f'{target_column}:N')
                        ).add_params(select_target)

                        # Aplicar filtro somente se habilitado
                        if filter_enabled:
                            chart = chart.transform_filter(select_target)
                        return chart
                    else:
                        # Without target
                        return base.encode(color=alt.value("gray"))

                # Criar e exibir gráfico
                mychart = create_chart(target if target else None, filter_enabled=apply_filter).interactive()
                st.altair_chart(mychart, use_container_width=True)

# ------------------------------------------------------------------------------    