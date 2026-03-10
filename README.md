# Student Marks Prediction Dashboard

**Made by Numan Asghar**

Interactive Streamlit dashboard for analyzing, evaluating, and predicting student marks using ML models. Features include:

- Workflow/Pipeline visualization  
- Results tables for RQ1, RQ2, RQ3  
- Train vs Test R² interactive charts  
- Bootstrapped MAE 95% CI plots  
- Downloadable CSV results  
- Select Research Question interactively  

---

## Demo & Run Locally

```bash
# Clone repo
git clone https://github.com/yourusername/student-marks-prediction-dashboard.git
cd student-marks-prediction-dashboard

# Create & activate venv
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
