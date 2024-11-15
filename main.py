import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Title and description
st.title("Customer Segmentation Using KMeans")
st.write("This application performs customer segmentation using KMeans clustering on the Mall Customers dataset.")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Select features for clustering
    st.sidebar.header("Clustering Configuration")
    features = st.sidebar.multiselect(
        "Select features for clustering",
        options=data.columns,
        default=["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    )

    if len(features) < 2:
        st.error("Please select at least two features.")
    else:
        # Display selected features
        st.write("### Selected Features for Clustering")
        st.write(data[features].head())

        # Number of clusters
        n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=5)

        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data["Cluster"] = kmeans.fit_predict(data[features])
        
        # Display clustered data
        st.write("### Clustered Data")
        st.write(data.head())

        # Cluster Visualization
        if len(features) == 2:
            st.write("### Clustering Visualization")
            plt.figure(figsize=(10, 6))
            for cluster in range(n_clusters):
                cluster_data = data[data["Cluster"] == cluster]
                plt.scatter(cluster_data[features[0]], cluster_data[features[1]], label=f"Cluster {cluster}")
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            plt.title("Clusters")
            plt.legend()
            st.pyplot(plt)
        else:
            st.write("Clustering visualization is only available for 2 features.")

else:
    st.info("Please upload a CSV file to begin.")
