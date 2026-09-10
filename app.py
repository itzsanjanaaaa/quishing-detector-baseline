"""
app.py
-------
Streamlit demo for the quishing detector baseline. Upload a QR code image
(or paste a URL directly) and get a phishing/legitimate verdict with
feature-level explanation.

Run with:
    streamlit run app.py
"""

import streamlit as st
import sys
import os
import importlib
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
predict = importlib.import_module("05_predict")

st.set_page_config(page_title="Quishing Detector - Baseline", page_icon="🔐", layout="centered")

st.title("🔐 Quishing Detector — Phase 1 Baseline")
st.caption(
    "Centralized baseline for detecting QR-code phishing (quishing) attacks. "
    "Part of dissertation research on federated learning for privacy-preserving quishing detection."
)

tab1, tab2 = st.tabs(["📷 Upload QR Code", "🔗 Test a URL directly"])

with tab1:
    uploaded = st.file_uploader("Upload a QR code image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        st.image(uploaded, caption="Uploaded QR code", width=200)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        result = predict.predict_from_qr_image(tmp_path)

        if "error" in result:
            st.error(result["error"])
        else:
            for r in result["results"]:
                verdict = r["verdict"]
                prob = r["phishing_probability"]

                st.subheader("Decoded URL")
                st.code(r["url"])

                if "PHISHING" in verdict:
                    st.error(f"⚠️ {verdict}  —  {prob*100:.1f}% phishing probability")
                else:
                    st.success(f"✅ {verdict}  —  {prob*100:.1f}% phishing probability")

                with st.expander("See extracted features (what the model looked at)"):
                    st.json(r["features"])

with tab2:
    url_input = st.text_input("Paste a URL to classify", placeholder="https://example.com/path")
    if url_input:
        r = predict.predict_url(url_input)
        verdict = r["verdict"]
        prob = r["phishing_probability"]

        if "PHISHING" in verdict:
            st.error(f"⚠️ {verdict}  —  {prob*100:.1f}% phishing probability")
        else:
            st.success(f"✅ {verdict}  —  {prob*100:.1f}% phishing probability")

        with st.expander("See extracted features"):
            st.json(r["features"])

st.divider()
st.caption(
    "⚠️ This is a research baseline (Phase 1), not a production security tool. "
    "See README for data sources, methodology, limitations, and the ablation study."
)
