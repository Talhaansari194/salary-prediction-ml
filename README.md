# 💰 Salary Prediction Using Regression and KNN

## 📌 Project Overview
This project aims to predict salaries based on job-related features using machine learning techniques. It applies Linear Regression and K-Nearest Neighbors (KNN) to analyze patterns in salary data.

---

## 🎯 Objective
To build a predictive model that estimates salary using features such as:
- Experience level
- Job title
- Company size
- Location
- Employment type

---

## 📂 Dataset
The dataset contains data science job salaries with various attributes.

### Key Features:
- experience_level
- employment_type
- job_title
- salary_in_usd (target variable)
- employee_residence
- remote_ratio
- company_location
- company_size

---

## 🔍 Exploratory Data Analysis (EDA)
- Salary distribution analysis
- Experience vs salary trends
- Company size impact
- Job role comparisons
- Salary trends over time

---

## ⚙️ Methodology

### Data Preprocessing:
- Handling categorical variables using encoding
- Feature scaling using StandardScaler
- Train-test split (80/20)

### Models Used:
1. Linear Regression  
2. K-Nearest Neighbors (KNN)

---

## 📊 Results

| Model | R² Score | RMSE |
|------|------|------|
| Linear Regression | 0.235 | 64139 |
| KNN (Initial) | 0.148 | 67703 |
| KNN (Tuned) | 0.216 | 64921 |

---

## 🚀 Improvements
- Reduced high-cardinality job titles
- Converted experience level into ordinal values
- Rebuilt preprocessing pipeline
- Tuned KNN hyperparameter (K)

---

## ⚠️ Limitations
- Salary data is dynamic and changes over time
- Dataset lacks key features like skills and certifications
- Predictions are approximate and not exact

---

## 🔮 Future Work
- Use real-time salary data
- Apply Ridge/Lasso regression
- Try ensemble models
- Deploy using Streamlit

---

## 🧠 Key Takeaways
- Data preprocessing significantly impacts performance
- Linear Regression outperformed KNN for this dataset
- Model accuracy depends heavily on feature quality

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib & Seaborn

---

## 📎 How to Run

1. Clone the repository  
2. Install dependencies  
3. Run the Jupyter Notebook  
