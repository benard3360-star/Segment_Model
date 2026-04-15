import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load models
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca_model.pkl")

# Load dataset for lookup
df_model = joblib.load("training_data.pkl")

# Cluster mapping
cluster_map = {
    0: "Dormant / Low Engagement Users",
    1: "High-Value Digital Power Users",
    2: "Active Mass Market Users"
}

st.title("Customer Segmentation System")

st.write("Enter Customer Number to identify segment")

# -------------------------
# USER INPUT
# -------------------------
cust_id = st.text_input("Customer Number")

if st.button("Predict Segment"):

    if cust_id not in df_model['customer_number'].values:
        st.error("Customer not found in dataset")
    else:
        # Extract customer row
        customer_row = df_model[df_model['customer_number'] == cust_id]

        # Drop ID column
        X = customer_row.drop(columns=['customer_number'])

        # Scale
        X_scaled = scaler.transform(X)

        # PCA
        X_pca = pca.transform(X_scaled)

        # Predict cluster
        cluster = kmeans.predict(X_pca)[0]

        # Output
        st.success(f"Customer Cluster: {cluster}")
        st.write(cluster_map.get(cluster))

        # Optional: show profile
        st.subheader("Customer Snapshot")
        st.dataframe(customer_row.T)