
# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

st.markdown("# Correlações")

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

df_numericos = df.select_dtypes(exclude=['object']) # Dados Numéricos

# ------------------------------------------------------------------------------

if len(df_numericos.columns) > 1:

    st.markdown("### Tabela de correlações")

    # Seleção do método de correlação
    method = st.selectbox(
        "Selecione o método de correlação",
        ["pearson", "kendall", "spearman"]
    )

    # Definição do número mínimo de períodos
    min_periods = st.number_input(
       "Número mínimo de observações por par de colunas (min_periods)", 
        min_value=1, 
        value=1, 
        step=1
    )

    # Opção de considerar apenas dados numéricos
    numeric_only = st.checkbox("Considerar apenas dados numéricos", value=False)

    # Botão para calcular correlações
    if st.button("Calcular correlação"):
        with st.spinner("Calculando correlações..."):
            try:
                # Cálculo da correlação
                corr_matrix = df.corr(
                    method=method, 
                    min_periods=min_periods, 
                    numeric_only=numeric_only
                )
                st.success("Correlação calculada com sucesso!")
                st.write("Matriz de correlação:")
                st.dataframe(corr_matrix)
            except Exception as e:
                st.error(f"Erro ao calcular a correlação: {e}")

    # ------------------------------------------------------------------------------

    st.markdown("### Correlações identificadas 🔍")
    with st.expander("Clique para exibir a tabela de correlações"):
        dfCorr = mp.top_entries(df)
        dfCorr.dropna(axis=0, inplace=True)
    
        # Aplicando estilo (opcional)
        styled_df = dfCorr.style.background_gradient(cmap="coolwarm")
        
        # Exibindo a tabela estilizada
        st.dataframe(styled_df, use_container_width=True)        
    
    # ------------------------------------------------------------------------------    

    st.markdown("### Tabela de correlações filtrada 🎯")
    
    with st.expander("Clique para visualizar a tabela filtrada de correlações"):
        # Filtragem das correlações
        dfCorr = mp.top_entries(df)
        dfCorr.dropna(axis=0, inplace=True)
        dfCorr = dfCorr.loc[
            ((dfCorr['Correlação'] >= 0.5) | (dfCorr['Correlação'] <= -0.5)) 
            & (dfCorr['Correlação'] != 1.000)
        ]
        
        # Aplicar estilo visual ao DataFrame
        styled_df = dfCorr.style.background_gradient(
            cmap="RdYlBu", subset=["Correlação"]
        )
        
        # Exibir a tabela com layout aprimorado
        st.dataframe(styled_df, use_container_width=True)

    # ------------------------------------------------------------------------------

    # https://stackoverflow.com/questions/43335973/how-to-generate-high-resolution-heatmap-using-seaborn
    # https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
    # https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
    
    st.markdown("### Mapa de calor das correlações 🔥")
    
    # Opções de filtro
    min_corr = st.slider("Selecione a correlação mínima", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    
    # Calcula a correlação do DataFrame
    dfCorr = df.corr()
    
    # Filtra as correlações acima ou abaixo do limiar especificado
    filteredDf = dfCorr[(dfCorr.abs() >= min_corr) & (dfCorr != 1.0)].dropna(how="all", axis=0).dropna(how="all", axis=1)

    # Cria o gráfico de calor
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    heatmap = sns.heatmap(
        filteredDf,
        vmin=-1, vmax=1, cbar=True, square=True,
        annot=True, cmap="PuOr", annot_kws={"size": 10},
        linewidths=0.5, linecolor='gray'
    )
    heatmap.set_title('Mapa de Calor das Correlações Filtradas', fontdict={'fontsize': 16}, pad=16)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    
    st.pyplot(fig)

    # ------------------------------------------------------------------------------

    st.markdown("### Mapa de Calor Triangular das Correlações 📊")
    
    # Máscara para exibir apenas a metade triangular inferior
    mask = np.triu(np.ones_like(filteredDf, dtype=bool))
    
    # Criação do gráfico de calor triangular
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    heatmap = sns.heatmap(
        filteredDf, 
        mask=mask, 
        vmin=-1, vmax=1, cbar=True, square=True, 
        annot=True, cmap="PuOr", annot_kws={"size": 10}, 
        linewidths=0.5, linecolor='gray'
    )
    heatmap.set_title('Mapa de Calor Triangular das Correlações', fontdict={'fontsize': 16}, pad=16)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    
    st.pyplot(fig)    

    # ------------------------------------------------------------------------------

    st.markdown("### Análise de Correlação com a Variável Alvo 🎯")

    # Seleção da coluna alvo (variável dependente)
    target = st.selectbox(
        "Selecione a coluna alvo (target)", 
        options=df.columns.tolist(), 
        index=0, 
        key="corr_target"
    )

    # Cálculo da matriz de correlação
    dfCorr = df.corr()

    # Filtrar apenas as correlações acima do limite definido pelo usuário
    filteredDf = dfCorr[[target]].dropna().loc[
        (dfCorr[target].abs() >= min_corr) & (dfCorr.index != target)
    ].sort_values(by=target, ascending=False)

    st.markdown(f"#### Tabela de Correlações com `{target}`")
    st.dataframe(filteredDf.style.background_gradient(cmap="coolwarm"))

    # Gráfico de calor para visualização das correlações
    st.markdown(f"#### Mapa de Calor: Correlação com `{target}`")
    fig, ax = plt.subplots(figsize=(8, 12), dpi=300)
    sns.heatmap(
        filteredDf, 
        vmin=-1, vmax=1, annot=True, cmap="BrBG", 
        linewidths=0.5, linecolor="gray", 
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title(
        f"Atributos Correlacionados com `{target}`", 
        fontdict={"fontsize": 16}, pad=16
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)

    
    # ------------------------------------------------------------------------------

    st.markdown("### Gráfico de Linha 📈")
    
    # Seleção da coluna alvo (target) para o eixo X
    target = st.selectbox(
        "Selecione a coluna alvo (eixo X)", 
        options=df.columns[::-1], 
        key="grafico_de_linha_target"
    )

    # Filtrar apenas colunas numéricas
    numerical = df.select_dtypes(exclude=["object"]).columns.tolist()

    # Remover o target das opções para o eixo Y
    if target in numerical:
        numerical.remove(target)

    if numerical:
        # Seleção da coluna do eixo Y
        y_column = st.selectbox(
            "Selecione a coluna para o eixo Y", 
            options=numerical, 
            key="y_line_plot_atributo_numerico"
        )

        # Plotando o gráfico
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        sns.lineplot(data=df, x=target, y=y_column, color="r", ax=ax)
        ax.set_title(f"Gráfico de Linha: {y_column} vs {target}", fontsize=16, pad=12)
        ax.set_xlabel(target, fontsize=14)
        ax.set_ylabel(y_column, fontsize=14)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("Não há colunas numéricas disponíveis para plotar o gráfico.")

    # ------------------------------------------------------------------------------
    
else:
    st.success("Número insuficiente de colunas numéricas para realizar análise de correlação.")
        
# ------------------------------------------------------------------------------
