import streamlit as st
import joblib
from utils import predict_duration, assign_worker

st.title("📋 Job Input")

st.sidebar.markdown("### 🛠 Job Configuration")

job_id = st.text_input("Job ID")
task_type = st.selectbox("Task Type", ["One-time", "Periodic"])
priority = st.slider("Priority", 1, 5, 3)
skill_required = st.slider("Skill Required", 1, 5, 3)
time_of_day = st.slider("Time of Day (0–23)", 0, 23, 12)

submit = st.button("Predict & Assign")

if submit:
    with st.spinner("Loading models and making predictions..."):
        duration_model = joblib.load("models/duration_model.pkl")
        worker_model = joblib.load("models/worker_model.pkl")

        input_data = {
            "priority": priority,
            "skill_required": skill_required,
            "time_of_day": time_of_day,
        }

        duration = predict_duration(duration_model, input_data)
        assigned_worker = assign_worker(worker_model, input_data)

        st.success(f"✅ Job `{job_id}` predicted duration: **{duration:.2f}** mins")
        st.info(f"👷 Assigned to Worker **{assigned_worker}**")


