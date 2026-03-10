# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

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
selected_rq = st.sidebar.selectbox("Select Research Question (RQ):", list(rqs.keys()))

# -------------------------------
# Workflow / About
st.title("Student Marks Prediction Dashboard")
st.markdown("""
This dashboard presents the analysis, evaluation, and model results for predicting student marks.
""")
st.header("Workflow / Pipeline")
st.image("workflow_diagram.png", caption='Prediction Workflow', width=700)
st.markdown("---")

# -------------------------------
# Display Selected RQ Results
df = rqs[selected_rq]
st.header(f"{selected_rq} - Results Table")
st.dataframe(df)

best_model_name = df['R2_test'].idxmax()
st.success(f"Best Model: {best_model_name}")

# -------------------------------
# Train vs Test R² Plot (Interactive)
st.subheader("Train vs Test R² Comparison")
r2_plot = df[['R2_train','R2_test']].copy()
r2_plot.index = df.index
fig_r2 = px.bar(
    r2_plot, 
    barmode='group', 
    title="Train vs Test R²", 
    labels={"value":"R²", "index":"Model"},
    text=r2_plot.index
)
st.plotly_chart(fig_r2)

# -------------------------------
# Bootstrap MAE 95% CI Plot
st.subheader("Bootstrapped MAE 95% CI")
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
st.plotly_chart(fig_mae)

# -------------------------------
# Download CSV
st.subheader("Download Results")
csv = df.to_csv().encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"{selected_rq.replace(' ','_')}_results.csv",
    mime='text/csv'
)
