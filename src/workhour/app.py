"""Streamlit app.  Usage: streamlit run src/workhour/app.py"""
import streamlit as st

from workhour.data import load_dataset, split_xy
from workhour.model import MODEL_PATH, load_model, predict_effort
from workhour.train import main as train_main

st.set_page_config(page_title="Software Effort Estimator", layout="wide")


@st.cache_data
def get_data():
    return load_dataset()


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        train_main()
    return load_model()


st.title("Software Project Effort Estimator")
st.caption("Predicts person-hours for a software project from size and team features.")

df = get_data()
model, features = get_model()
X, _ = split_xy(df)

tab_estimate, tab_data = st.tabs(["Estimate effort", "Explore data"])

with tab_estimate:
    st.subheader("Project characteristics")
    cols = st.columns(3)
    inputs = {}
    for i, feat in enumerate(features):
        lo, hi, med = float(X[feat].min()), float(X[feat].max()), float(X[feat].median())
        with cols[i % 3]:
            inputs[feat] = st.number_input(feat, min_value=lo, max_value=hi, value=med)
    if st.button("Estimate effort", type="primary"):
        hours = predict_effort(model, features, inputs)
        st.metric("Estimated effort (person-hours)", f"{hours:,.0f}")

with tab_data:
    st.subheader("Dataset")
    st.dataframe(df)
    st.write(f"{df.shape[0]} projects, {df.shape[1]} columns")
    st.image("docs/heatmap.png", caption="Feature correlation heatmap")
