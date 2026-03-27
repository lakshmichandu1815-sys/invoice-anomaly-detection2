import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Invoice Anomaly Detection")

st.title("📄 Invoice Anomaly Detection System")

st.write("Detect unusual invoice amounts using Machine Learning")

# Sample dataset
data = np.array([[1000], [1500], [2000], [2500], [3000], [50000]])
df = pd.DataFrame(data, columns=["Amount"])

st.subheader("Sample Invoice Data")
st.dataframe(df)

# Model
model = IsolationForest(contamination=0.1, random_state=42)
df["Anomaly"] = model.fit_predict(df)

df["Result"] = df["Anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

st.subheader("Detection Result")
st.dataframe(df[["Amount", "Result"]])

# Manual input
st.subheader("Check New Invoice")

amount = st.number_input("Enter Invoice Amount", min_value=0)

if st.button("Check"):
    prediction = model.predict([[amount]])

    if prediction[0] == -1:
        st.error("⚠️ Anomaly Detected")
    else:
        st.success("✅ Normal Transaction")
