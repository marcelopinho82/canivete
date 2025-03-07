# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, OPTICS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture

try:
    from hdbscan import HDBSCAN
except ImportError:
    st.warning("HDBSCAN não está disponível. Certifique-se de instalá-lo se precisar deste algoritmo.")

import altair as alt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from kneed import KneeLocator

# ------------------------------------------------------------------------------

import sys
sys.path.append('./')
import marcelo as mp

# ------------------------------------------------------------------------------

alt.themes.enable('opaque')  # Deixa o fundo dos gráficos transparente

# ------------------------------------------------------------------------------

st.markdown("# Aprendizado de Máquina Não Supervisionado (ANS)")

# ------------------------------------------------------------------------------
# Função para download do CSV
# ------------------------------------------------------------------------------

def downloadCSV(df, file_name):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Download CSV Processado',
        data=csv,
        mime='text/csv',
        file_name=file_name
    )

# ------------------------------------------------------------------------------
# Função para processar colunas de texto
# ------------------------------------------------------------------------------

def process_text_columns(df, selected_columns, method):
    """
    Processa colunas de texto usando TF-IDF ou embeddings.

    Args:
        df (pd.DataFrame): DataFrame com os dados.
        selected_columns (list): Lista de colunas selecionadas.
        method (str): Método de processamento ('tf-idf' ou 'embeddings').

    Returns:
        np.ndarray: Dados transformados.
    """
    # Combinar as colunas de texto em uma única coluna "Texto"
    df['Texto'] = df[selected_columns].astype(str).agg(' '.join, axis=1)

    if method == 'tf-idf':
        # Representação dos textos com TF-IDF
        vectorizer = TfidfVectorizer(max_features=10000)  # Limitar a 10.000 palavras mais frequentes
        transformed_data = vectorizer.fit_transform(df['Texto']).toarray()

    elif method == 'embeddings':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        def batched_embeddings(texts, model, batch_size=32):
            loader = DataLoader(texts, batch_size=batch_size, shuffle=False)
            embeddings = []
            for batch in loader:
                embeddings.extend(model.encode(batch, show_progress_bar=True))
            return np.array(embeddings)
        transformed_data = batched_embeddings(df['Texto'].tolist(), model, batch_size=32)
    else:
        raise ValueError("Método de processamento desconhecido: escolha 'tf-idf' ou 'embeddings'")

    return transformed_data

# ------------------------------------------------------------------------------
# Carregar Dados
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
# Seleção de colunas do dataset
# ------------------------------------------------------------------------------

st.markdown("### Seleção de Colunas")

# Opção para selecionar apenas colunas numéricas ou categóricas
column_type = st.radio(
    "Escolha o tipo de colunas para análise:",
    options=['Numéricas', 'Categóricas'],
    index=0
)

if column_type == 'Numéricas':
    filtered_columns = df.select_dtypes(include=['number']).columns.tolist()
elif column_type == 'Categóricas':
    filtered_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

if "Texto" in filtered_columns:
    features = st.multiselect('Selecione as colunas para clustering', options=filtered_columns, default=["Texto"])    
else:
    features = st.multiselect('Selecione as colunas para clustering', options=filtered_columns, default=filtered_columns)

# Mensagem de erro caso nenhuma coluna seja selecionada
if not features:
    st.error("Por favor, selecione ao menos uma coluna para clustering.")

if column_type == 'Categóricas':
    # Escolher o método de processamento para colunas de texto
    text_processing_method = st.selectbox('Selecione o método de processamento para colunas de texto', ['embeddings','tf-idf'])

# ------------------------------------------------------------------------------
# Configurações de Clustering
# ------------------------------------------------------------------------------

st.markdown("### Configurações de Clustering")

# Checkbox para selecionar se é para clusterizar ou não
clusterizar = st.checkbox(
    "Realizar Clustering?",
    value=False
)

if clusterizar:

    algorithms = st.multiselect(
        "Selecione os algoritmos a serem utilizados para clusterização:",
        options=['KMeans', 'MiniBatchKMeans', 'DBSCAN', 'Agglomerative', 'SpectralClustering', 'OPTICS', 'GaussianMixture', 'HDBSCAN'],
        default=['KMeans', 'MiniBatchKMeans', 'DBSCAN', 'Agglomerative', 'SpectralClustering', 'OPTICS', 'GaussianMixture', 'HDBSCAN']
    )

    if 'KMeans' in algorithms:
        # Checkbox para cálculo do silhouette no KMeans
        calcular_numero_ideal_clusters = st.checkbox(
            "Calcular o número ideal de Clusters?",
            value=False
        )

        if calcular_numero_ideal_clusters:
            # Slider para definir o range de busca dos grupos
            min_clusters, max_clusters = st.slider(
                "Selecione o intervalo de clusters (k)",
                min_value=2,
                max_value=100,
                value=(2, 10),
                step=1
            )

    # ------------------------------------------------------------------------------
    # Configurações de Hiperparâmetros
    # ------------------------------------------------------------------------------

    st.sidebar.header("Configurações de Hiperparâmetros")

    hyperparameters = {}

    # KMeans
    if 'KMeans' in algorithms:
        st.sidebar.subheader("KMeans")
        hyperparameters['KMeans'] = {
            'n_clusters': st.sidebar.slider("Número de Clusters (KMeans)", min_value=2, max_value=100, value=3),
            'init': st.sidebar.selectbox("Método de Inicialização", ['k-means++', 'random'], index=0),
            'max_iter': st.sidebar.number_input("Máximo de Iterações", min_value=100, max_value=1000, value=300),
            'random_state': 42
        }

    # MiniBatchKMeans
    if 'MiniBatchKMeans' in algorithms:
        st.sidebar.subheader("MiniBatchKMeans")
        hyperparameters['MiniBatchKMeans'] = {
            'n_clusters': st.sidebar.slider("Número de Clusters (MiniBatchKMeans)", min_value=2, max_value=100, value=3),
            'batch_size': st.sidebar.number_input("Tamanho do Lote", min_value=10, max_value=10000, value=1000),
            'random_state': 42
        }

    # DBSCAN
    if 'DBSCAN' in algorithms:
        st.sidebar.subheader("DBSCAN")
        hyperparameters['DBSCAN'] = {
            'eps': st.sidebar.slider("Distância Máxima (eps)", min_value=0.1, max_value=5.0, step=0.1, value=0.5),
            'min_samples': st.sidebar.number_input("Mínimo de Amostras", min_value=1, max_value=50, value=5, key = "dbscan_min_samples")
        }

    # Agglomerative Clustering
    if 'Agglomerative' in algorithms:
        st.sidebar.subheader("Agglomerative Clustering")
        hyperparameters['Agglomerative'] = {
            'n_clusters': st.sidebar.slider("Número de Clusters (Agglomerative)", min_value=2, max_value=100, value=3),
            'linkage': st.sidebar.selectbox("Tipo de Ligação", ['ward', 'complete', 'average', 'single'], index=0)
        }

    # Spectral Clustering
    if 'SpectralClustering' in algorithms:
        st.sidebar.subheader("Spectral Clustering")
        hyperparameters['SpectralClustering'] = {
            'n_clusters': st.sidebar.slider("Número de Clusters (SpectralClustering)", min_value=2, max_value=100, value=3),
            'affinity': st.sidebar.selectbox("Tipo de Afinidade", ['nearest_neighbors', 'rbf'], index=0),
            'random_state': 42
        }

    # OPTICS
    if 'OPTICS' in algorithms:
        st.sidebar.subheader("OPTICS")
        hyperparameters['OPTICS'] = {
            'min_samples': st.sidebar.number_input("Mínimo de Amostras", min_value=1, max_value=50, value=5, key = "optics_min_samples"),
            'max_eps': st.sidebar.slider("Máximo de eps", min_value=0.1, max_value=5.0, step=0.1, value=2.0),
            'metric': st.sidebar.selectbox("Métrica", ['euclidean', 'manhattan', 'cosine'], index=0, key = "optics_metric")
        }

    # Gaussian Mixture
    if 'GaussianMixture' in algorithms:
        st.sidebar.subheader("Gaussian Mixture")
        hyperparameters['GaussianMixture'] = {
            'n_components': st.sidebar.slider("Número de Componentes", min_value=2, max_value=20, value=3),
            'covariance_type': st.sidebar.selectbox("Tipo de Covariância", ['full', 'tied', 'diag', 'spherical'], index=0),
            'random_state': 42
        }

    # HDBSCAN
    if 'HDBSCAN' in algorithms:
        st.sidebar.subheader("HDBSCAN")
        hyperparameters['HDBSCAN'] = {
            'min_samples': st.sidebar.number_input("Mínimo de Amostras", min_value=1, max_value=50, value=5, key = "hdbscan_min_samples"),
            'min_cluster_size': st.sidebar.number_input("Tamanho Mínimo do Cluster", min_value=1, max_value=100, value=15),
            'metric': st.sidebar.selectbox("Métrica", ['euclidean', 'manhattan', 'cosine'], index=0, key = "hdbscan_metric")
        }

# Checkbox para visualizar instâncias e resultados em PCA
visualize_pca = st.checkbox(
    "Visualizar instâncias e resultados dos agrupamentos em PCA?",
    value=False
)

# Checkbox para visualizar instâncias e resultados em t-SNE
visualize_tsne = st.checkbox(
    "Visualizar instâncias e resultados dos agrupamentos em t-SNE?",
    value=True
) 

# ------------------------------------------------------------------------------

if features:  

    if st.button("Processar"):

        # Verificar se as colunas são de texto
        text_columns = [col for col in features if df[col].dtype == 'object']
        numeric_columns = [col for col in features if col not in text_columns]

        if text_columns:
            st.markdown(f"### Processando colunas de texto: {text_columns}")
            data_scaled = process_text_columns(df, text_columns, text_processing_method)
        else:
            data = df[features].to_numpy()
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

        if visualize_pca:

            st.markdown("### Visualização das Instâncias (PCA)")
            pca = PCA(n_components=2)
            with st.spinner("Aguarde enquanto processamos os dados..."):
                pca_data = pca.fit_transform(data_scaled)
            
            df['PCA1'] = pca_data[:, 0]
            df['PCA2'] = pca_data[:, 1]

            chart = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('PCA1', title='Componente Principal 1'),
                y=alt.Y('PCA2', title='Componente Principal 2'),
                tooltip=['PCA1', 'PCA2']
            ).properties(
                title='Instâncias Reduzidas para 2 Dimensões (PCA)'
            )
            st.altair_chart(chart, use_container_width=True)            

        if visualize_tsne:

            st.markdown("### Visualização das Instâncias (t-SNE)")
                
            tsne = TSNE(n_components=2, random_state=42)
            with st.spinner("Aguarde enquanto processamos os dados..."):
                tsne_data = tsne.fit_transform(data_scaled)
                
            df['tSNE1'] = tsne_data[:, 0]
            df['tSNE2'] = tsne_data[:, 1]

            tsne_chart = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('tSNE1', title='t-SNE Componente 1'),
                y=alt.Y('tSNE2', title='t-SNE Componente 2'),
                tooltip=['tSNE1', 'tSNE2']
            ).properties(
                title='Instâncias Reduzidas para 2 Dimensões (t-SNE)'
            )
            st.altair_chart(tsne_chart, use_container_width=True)
        
        if clusterizar:

            try:

                # 1. KMeans Clustering
                if 'KMeans' in algorithms:

                    if calcular_numero_ideal_clusters:

                        st.markdown("### Número Ideal de Clusters (KMeans)")

                        # Define os parâmetros-padrão do k-means
                        parametros_basicos_kmeans = {
                            "init": "random",
                            "n_init": 10,
                            "max_iter": 300,
                            "random_state": 42,
                        }

                        resultados_sse = []  # Lista que armazena o SSE de cada 'k'
                        resultados_silhouette = [] # Lista que armazena o Silhouette de cada 'k' 

                        kmeans = KMeans(n_clusters=1, **parametros_basicos_kmeans)
                        kmeans.fit(data_scaled)
                        resultados_sse.append(kmeans.inertia_)

                        # Executa o algoritmo para 'k' variando entre 2 e 10
                        # O Silhouette só funciona quando temos 2 ou mais clusters!
                        kmeans_scores = []                        
                        for k in range(max(2, min_clusters), max_clusters + 1):
                            kmeans = KMeans(n_clusters=k, **parametros_basicos_kmeans)
                            kmeans.fit(data_scaled)
                            resultados_sse.append(kmeans.inertia_)
                            
                            score = silhouette_score(data_scaled, kmeans.labels_)
                            resultados_silhouette.append(score)

                            labels = kmeans.fit_predict(data_scaled)
                            score = silhouette_score(data_scaled, labels)
                            kmeans_scores.append((k, score))

                        # Cria um DataFrame com os resultados
                        sse_data = pd.DataFrame({
                            "Número de clusters (k)": list(range(1, max_clusters + 1)),
                            "SSE": resultados_sse,
                            "Silhueta": [0] + resultados_silhouette
                        })

                        # Gera o gráfico usando Altair
                        grafico_sse = alt.Chart(sse_data).mark_line(point=True).encode(
                            x=alt.X("Número de clusters (k):O", title="Número de clusters (k)"),
                            y=alt.Y("SSE:Q", title="SSE"),
                            tooltip=["Número de clusters (k)", "SSE"]
                        ).properties(
                            title="Correlação entre Número de Clusters (k) e SSE",
                        )

                        # Adiciona uma linha reta em vermelho entre o menor e o maior SSE
                        linha_reta = alt.Chart(pd.DataFrame({
                            "x": [1, max_clusters],
                            "y": [sse_data["SSE"].max(), sse_data["SSE"].min()]
                        })).mark_line(color="red", strokeDash=[5, 5]).encode(
                            x=alt.X("x:O", title="Número de clusters (k)"),
                            y=alt.Y("y:Q", title="SSE")
                        )

                        st.altair_chart(grafico_sse + linha_reta, use_container_width=True)

                        kl = KneeLocator(range(1, max_clusters + 1), resultados_sse, curve="convex", direction="decreasing")
                        st.write(f"Número ideal de clusters pelo Método do Cotovelo (Elbow): {kl.elbow}")

                        # Gera o gráfico usando Altair
                        grafico_silhouette = alt.Chart(sse_data).mark_line(point=True).encode(
                            x=alt.X("Número de clusters (k):O", title="Número de clusters (k)"),
                            y=alt.Y("Silhueta:Q", title="Coeficiente de Silhueta"),
                            tooltip=["Número de clusters (k)", "Silhueta"]
                        ).properties(
                            title="Correlação entre Número de Clusters (k) e Coeficiente de Silhueta",
                        )

                        st.altair_chart(grafico_silhouette, use_container_width=True)

                        ideal_clusters_kmeans = max(kmeans_scores, key=lambda x: x[1])[0]
                        st.write(f"Número ideal de clusters sugerido pelo Coeficiente de Silhueta: {ideal_clusters_kmeans}")

                    kmeans = KMeans(**hyperparameters['KMeans'])
                    df['KMeans'] = kmeans.fit_predict(data_scaled)                

                # 2. MiniBatchKMeans
                if 'MiniBatchKMeans' in algorithms:
                    minibatch_kmeans = MiniBatchKMeans(**hyperparameters['MiniBatchKMeans'])
                    df['MiniBatchKMeans'] = minibatch_kmeans.fit_predict(data_scaled)

                # 3. DBSCAN Clustering
                if 'DBSCAN' in algorithms:
                    dbscan = DBSCAN(**hyperparameters['DBSCAN'])
                    df['DBSCAN'] = dbscan.fit_predict(data_scaled)

                # 4. Agglomerative Clustering    
                if 'Agglomerative' in algorithms:
                    agg = AgglomerativeClustering(**hyperparameters['Agglomerative'])
                    df['Agglomerative'] = agg.fit_predict(data_scaled)

                # 5. Spectral Clustering
                if 'SpectralClustering' in algorithms:
                    spectral = SpectralClustering(**hyperparameters['SpectralClustering'])
                    df['SpectralClustering'] = spectral.fit_predict(data_scaled)

                # 6. OPTICS Clustering
                if 'OPTICS' in algorithms:
                    optics = OPTICS(**hyperparameters['OPTICS'])
                    df['OPTICS'] = optics.fit_predict(data_scaled)  

                # 7. Gaussian Mixture Clustering
                if 'GaussianMixture' in algorithms:
                    gaussian = GaussianMixture(**hyperparameters['GaussianMixture'])
                    df['GaussianMixture'] = gaussian.fit_predict(data_scaled)

                # 8. HDBSCAN Clustering
                if 'HDBSCAN' in algorithms:
                    hdbscan = HDBSCAN(**hyperparameters['HDBSCAN'])
                    df['HDBSCAN'] = hdbscan.fit_predict(data_scaled)

                st.markdown("### Resultados do Clustering")
                st.dataframe(df)                
                
                for algorithm in algorithms:
                    if algorithm in df.columns:
                        if visualize_pca or visualize_tsne:
                            st.subheader(f"Visualização dos Clusters ({algorithm})")

                        if visualize_pca:
                            chart = alt.Chart(df).mark_circle(size=60).encode(
                                x=alt.X('PCA1', title='Componente Principal 1'),
                                y=alt.Y('PCA2', title='Componente Principal 2'),                        
                                color=alt.Color(field=algorithm, type="nominal", scale=alt.Scale(scheme='viridis'), title='Cluster'),
                                tooltip=['PCA1', 'PCA2', algorithm]
                            ).properties(
                                title=f'PCA - Clusters Identificados ({algorithm})'
                            )
                            st.altair_chart(chart, use_container_width=True)

                        if visualize_tsne:
                            chart = alt.Chart(df).mark_circle(size=60).encode(
                                x=alt.X('tSNE1', title='t-SNE Componente 1'),
                                y=alt.Y('tSNE2', title='t-SNE Componente 2'),                    
                                color=alt.Color(field=algorithm, type="nominal", scale=alt.Scale(scheme='viridis'), title='Cluster'),
                                tooltip=['tSNE1', 'tSNE2', algorithm]
                            ).properties(
                                title=f't-SNE - Clusters Identificados ({algorithm})'
                            )
                            st.altair_chart(chart, use_container_width=True)

                st.markdown("### Avaliação dos Modelos")
                results = {}
                for algorithm in algorithms:
                    if algorithm in df.columns:
                        try:
                            score = silhouette_score(data_scaled, df[algorithm])
                            results[algorithm] = score
                            st.write(f"Coeficiente de Silhueta para {algorithm}: {score:.2f}")
                        except ValueError as e:
                            st.warning(f"Não foi possível calcular o silhouette para {algorithm}: {e}")


                # Exibindo as métricas em um gráfico interativo
                results_df = pd.DataFrame(list(results.items()), columns=['Algoritmo', 'Silhueta'])
                chart = alt.Chart(results_df).mark_bar().encode(
                    x=alt.X('Algoritmo', title='Algoritmo de Clustering'),
                    y=alt.Y('Silhueta', title='Coeficiente de Silhueta'),
                    color='Algoritmo'
                ).properties(title='Qualidade dos Agrupamentos')
                st.altair_chart(chart, use_container_width=True)

                # Download do CSV com clusters        
                df_to_download = df.copy()
                cols_to_drop = ['PCA1', 'PCA2', 'tSNE1', 'tSNE2']
                df_to_download.drop(columns=[col for col in cols_to_drop if col in df_to_download.columns], inplace=True)

                downloadCSV(df_to_download, file_name=option.split('.')[0] + '_Clustering.csv')

            except Exception as e:
                st.error(f"Erro ao executar o clustering: {e}")

else:
    st.error("Por favor, selecione ao menos uma coluna para clustering.")
