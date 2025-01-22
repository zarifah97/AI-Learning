import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title('KMeans Clustering with Summary and Evaluation')

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file)

    # Display the uploaded dataset
    st.write("### Uploaded Dataset")
    st.write(df.head())

    # Select number of clusters (k)
    k = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

    # Select columns for clustering (use all numeric columns by default)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    selected_columns = st.multiselect("Select columns for clustering", numeric_columns.tolist(), default=numeric_columns.tolist())
    
    # Perform KMeans Clustering
    if len(selected_columns) > 0:
        X = df[selected_columns]
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)

        # Cluster labels
        labels = kmeans.labels_

        # Add the labels to the original dataframe
        df['Cluster'] = labels

        # Summary of clusters
        st.write("### KMeans Clustering Summary")

        # Show centroids
        centroids = kmeans.cluster_centers_
        st.write(f"Centroids of the {k} clusters:")
        centroids_df = pd.DataFrame(centroids, columns=selected_columns)
        st.write(centroids_df)

        # Show cluster sizes (number of points in each cluster)
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        st.write(f"Cluster Sizes (number of points in each cluster):")
        st.write(cluster_sizes)

        # Plotting: Scatter plot of the clusters
        st.write("### Cluster Plot")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[selected_columns[0]], y=df[selected_columns[1]], hue=labels, palette='viridis', s=100, alpha=0.7)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroids")
        plt.title(f"KMeans Clustering with {k} clusters")
        plt.xlabel(selected_columns[0])
        plt.ylabel(selected_columns[1])
        plt.legend()
        st.pyplot(plt)

        # Evaluation Metrics
        st.write("### Evaluation Metrics")

        # Inertia (within-cluster sum of squared distances)
        inertia = kmeans.inertia_
        st.write(f"Inertia (Sum of squared distances of samples to their closest cluster center): {inertia}")

        # Silhouette Score
        if len(np.unique(labels)) > 1:  # Silhouette score is only valid if there are at least two clusters
            silhouette_avg = silhouette_score(X, labels)
            st.write(f"Silhouette Score: {silhouette_avg}")
        else:
            st.write("Silhouette Score cannot be computed with a single cluster.")

        # Davies-Bouldin Index
        db_index = davies_bouldin_score(X, labels)
        st.write(f"Davies-Bouldin Index: {db_index}")

    else:
        st.write("Please select at least one column for clustering.")
else:
    st.write("Please upload a dataset to begin.")
