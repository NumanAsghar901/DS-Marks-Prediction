# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# Page Configuration (Responsive Layout)
st.set_page_config(
    page_title="Student Marks Prediction Dashboard",
    layout="wide"
)

# -------------------------------
# Load Data
df_rq1_results = pd.read_csv("df_rq1_results.csv", index_col=0)
df_rq2_results = pd.read_csv("df_rq2_results.csv", index_col=0)
df_rq3_results = pd.read_csv("df_rq3_results.csv", index_col=0)

rqs = {
    "RQ1: Predict Midterm I": df_rq1_results,
    "RQ2: Predict Midterm II": df_rq2_results,
    "RQ3: Predict Final Exam": df_rq3_results
}

# -------------------------------
# Sidebar
st.sidebar.title("Student Marks Prediction")
st.sidebar.markdown("Made by **Numan Asghar**")
st.sidebar.markdown("---")

selected_rq = st.sidebar.selectbox(
    "Select Research Question (RQ):",
    list(rqs.keys())
)

# -------------------------------
# Title / Description
st.title("Student Marks Prediction Dashboard")

st.markdown("""
This dashboard presents the **analysis, evaluation, and model results**
for predicting student marks using machine learning models.
""")

# -------------------------------
# Workflow Diagram (Small & Centered)
st.header("Workflow / Pipeline")

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image(
        "workflow_diagram.png",
        caption="Prediction Workflow",
        width=400
    )

st.markdown("---")

# -------------------------------
# Display Selected RQ Results
df = rqs[selected_rq]

st.header(f"{selected_rq} - Results Table")

st.dataframe(df, use_container_width=True)

best_model_name = df['R2_test'].idxmax()

st.success(f"Best Model: {best_model_name}")

# -------------------------------
# Train vs Test R² Plot
st.subheader("Train vs Test R² Comparison")

r2_plot = df[['R2_train', 'R2_test']].copy()
r2_plot.index = df.index

fig_r2 = px.bar(
    r2_plot,
    barmode='group',
    title="Train vs Test R²",
    labels={"value": "R²", "index": "Model"},
    text=r2_plot.index
)

# -------------------------------
# Bootstrap MAE 95% CI Plot
st.subheader("Bootstrapped MAE 95% Confidence Interval")

mae_df = pd.DataFrame({
    "MAE": df['MAE_boot_mean'],
    "lower": df['MAE_boot_mean'] - df['MAE_boot_2.5pct'],
    "upper": df['MAE_boot_97.5pct'] - df['MAE_boot_mean']
}, index=df.index)

fig_mae = px.bar(
    mae_df,
    y="MAE",
    error_y="upper",
    error_y_minus="lower",
    text=mae_df.index,
    title="Bootstrapped MAE 95% CI"
)

# -------------------------------
# Display Charts (Responsive)
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_r2, use_container_width=True)

with col2:
    st.plotly_chart(fig_mae, use_container_width=True)

# -------------------------------
# Download CSV
st.subheader("Download Results")

csv = df.to_csv().encode('utf-8')

st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"{selected_rq.replace(' ','_')}_results.csv",
    mime="text/csv"
)

# -------------------------------
# Footer
st.markdown("---")
st.markdown("Developed by **Numan Asghar** | Data Science Project")
