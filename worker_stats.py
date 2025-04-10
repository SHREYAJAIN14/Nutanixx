# pages/worker_stats.py

import streamlit as st
import pandas as pd
import numpy as np

st.title("📈 Worker Stats")

# Dummy Data
data = pd.DataFrame({
    "Worker ID": range(1, 6),
    "Current Load (%)": np.random.randint(20, 100, size=5),
    "Success Rate (%)": np.random.randint(70, 100, size=5),
    "Tasks Completed": np.random.randint(10, 50, size=5),
})

st.dataframe(data)
st.bar_chart(data.set_index("Worker ID")[["Current Load (%)", "Success Rate (%)"]])
