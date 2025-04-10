
import streamlit as st

st.set_page_config(page_title="ML Task Scheduler", layout="wide")

st.title("🧠 ML-Based Task Scheduler")
st.write("Welcome! Use the sidebar to navigate between task assignment and worker statistics.")

st.markdown("""
---
Built using Streamlit + scikit-learn. Uses a trained ML model to predict optimal worker assignments.
""")

