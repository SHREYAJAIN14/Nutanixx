import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# Generate synthetic dataset
# -----------------------------

# For duration prediction
np.random.seed(42)
n_samples = 500

duration_data = pd.DataFrame({
    'priority': np.random.randint(1, 5, n_samples),
    'skill_required': np.random.randint(1, 5, n_samples),
    'time_of_day': np.random.randint(0, 24, n_samples),
    'task_type': np.random.randint(0, 2, n_samples),  # 0 = periodic, 1 = one-time
    'estimated_duration': np.random.randint(5, 100, n_samples)
})

X_duration = duration_data.drop('estimated_duration', axis=1)
y_duration = duration_data['estimated_duration']

# Train model
duration_model = LinearRegression()
duration_model.fit(X_duration, y_duration)

# Save duration model
with open('models/duration_model.pkl', 'wb') as f:
    pickle.dump(duration_model, f)


# -----------------------------
# For worker assignment (classification)
# -----------------------------

worker_data = pd.DataFrame({
    'task_type': np.random.randint(0, 2, n_samples),
    'priority': np.random.randint(1, 5, n_samples),
    'duration': np.random.randint(5, 100, n_samples),
    'skill_required': np.random.randint(1, 5, n_samples),
    'time_of_day': np.random.randint(0, 24, n_samples),
    'worker_skill': np.random.randint(1, 6, n_samples),
    'current_load': np.random.randint(0, 100, n_samples),
    'past_success_rate': np.round(np.random.uniform(0.4, 1.0, n_samples), 2),
    'success': np.random.randint(0, 2, n_samples)
})

X_worker = worker_data.drop('success', axis=1)
y_worker = worker_data['success']

# Train model
worker_model = LogisticRegression()
worker_model.fit(X_worker, y_worker)

# Save worker model
with open('models/worker_model.pkl', 'wb') as f:
    pickle.dump(worker_model, f)

print("✅ Models trained and saved in 'models/' folder")

