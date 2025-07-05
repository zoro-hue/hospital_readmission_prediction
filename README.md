# 🏥 Patient Readmission Prediction - Capstone AI/ML Project

This project predicts whether a hospital patient will be readmitted after discharge using Machine Learning. It includes data preprocessing, model building, evaluation, and a deployed web app using Streamlit.

 📌 Project Objective

Predict unplanned hospital readmissions using patient data such as age, medical procedures, lab tests, medications, and prior visit history. The goal is to help hospitals reduce readmission rates and improve patient care.

🧠 Technologies & Tools

- **Python 3.11+**
- **Pandas, NumPy** – Data preprocessing
- **Scikit-learn** – ML models and evaluation
- **Matplotlib, Seaborn** – EDA visualizations
- **Streamlit** – Web app deployment
- **Pickle** – Save and load models
- **Jupyter Notebook** – ML pipeline development


📂 Project Structure

readmission_project/
│
├── hospital_readmissions.csv # Dataset (25,000 records)
├── readmission_prediction.ipynb # Jupyter Notebook (data → model → save)
├── model.pkl # Trained Random Forest model
├── scaler.pkl # StandardScaler for input features
├── feature_names.pkl # Column names used during training
├── app.py # Streamlit UI for prediction
├── requirements.txt # Dependencies
└── README.md # This file



⚙️ How to Run This Project Locally

1. **Clone the repository**

git clone https://github.com/your-username/patient-readmission-predictor.git
cd patient-readmission-predictor

2. Create virtual environment (optional)

python -m venv venv
venv\Scripts\activate     # On Windows
or
source venv/bin/activate  # On Linux/Mac

3.Install dependencies

pip install -r requirements.txt

4.Run the Streamlit app

streamlit run app.py

5.Input patient features and get prediction!

📊 ML Model Summary
Models Used: Logistic Regression, Random Forest

Best Model: Random Forest (Accuracy ~87%)

Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

Important Features: n_medications, n_inpatient, age, glucose_test


🚀 Future Improvements
Deploy on Streamlit Cloud or AWS EC2

Add support for real-time hospital data

Integrate explainability tools (e.g., SHAP values)

Extend to multi-class classification (30-day vs 60-day readmission)

👨‍💻 Author
Jayanth Jagu
Capstone AI/ML Project | July 2025
📜 License
This project is licensed for educational use only.
