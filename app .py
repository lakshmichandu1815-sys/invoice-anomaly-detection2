import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Invoice Anomaly Detection", layout="centered")

st.title("📄 Automated Invoice Processing & Anomaly Detection")

st.write("Upload an invoice image or PDF to detect anomalies.")

# File upload
uploaded_file = st.file_uploader("Upload Invoice (Image only)", type=["png", "jpg", "jpeg"])

# Function to extract numbers from text
def extract_amounts(text):
    import re
    numbers = re.findall(r'\d+', text)
    return [int(num) for num in numbers if int(num) > 100]

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Invoice")
    st.image(image, use_column_width=True)

    # OCR extraction
    text = pytesseract.image_to_string(image)
    
    st.subheader("Extracted Text")
    st.text(text)

    # Extract numeric values
    amounts = extract_amounts(text)

    if len(amounts) > 3:
        df = pd.DataFrame(amounts, columns=["Amount"])

        st.subheader("Extracted Amounts")
        st.dataframe(df)

        # ML Model
        model = IsolationForest(contamination=0.1, random_state=42)
        df["Anomaly"] = model.fit_predict(df[["Amount"]])

        df["Result"] = df["Anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

        st.subheader("Anomaly Detection Result")
        st.dataframe(df[["Amount", "Result"]])

    else:
        st.warning("Not enough data detected in invoice.")

# Manual input section
st.subheader("Manual Invoice Check")

amount = st.number_input("Enter Invoice Amount", min_value=0)

if st.button("Check Anomaly"):
    data = np.array([[1000], [1500], [2000], [2500], [3000], [50000]])

    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(data)

    prediction = model.predict([[amount]])

    if prediction[0] == -1:
        st.error("⚠️ This amount looks like an ANOMALY!")
    else:
        st.success("✅ This amount looks NORMAL")