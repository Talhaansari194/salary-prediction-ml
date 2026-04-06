import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# from google.colab import files

# uploaded = files.upload()

df = pd.read_csv("DataScience_salaries_2025.csv")

df.head()

df.info()

df.describe()

df.isnull().sum()

df.columns

sns.histplot(df['salary_in_usd'], bins=30, kde=True)
plt.title("Salary Distribution")
plt.xlabel("Salary in USD")
plt.ylabel("Frequency")
plt.show()

avg_salary_exp = df.groupby('experience_level')['salary_in_usd'].mean().reset_index()

sns.barplot(x='experience_level', y='salary_in_usd', data=avg_salary_exp)
plt.title("Average Salary by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Average Salary (USD)")
plt.show()

avg_salary_company = df.groupby('company_size')['salary_in_usd'].mean().reset_index()

sns.barplot(x='company_size', y='salary_in_usd', data=avg_salary_company)
plt.title("Average Salary by Company Size")
plt.show()

top_jobs = df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).head(10)

top_jobs.plot(kind='bar', figsize=(10,5))
plt.title("Top 10 Highest Paying Job Titles")
plt.ylabel("Average Salary")
plt.show()

numeric_df = df.select_dtypes(include=['int64','float64'])

sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

sns.scatterplot(x='remote_ratio', y='salary_in_usd', data=df)
plt.title("Remote Work Ratio vs Salary")
plt.show()

salary_year = df.groupby('work_year')['salary_in_usd'].mean()

salary_year.plot(marker='o')
plt.title("Average Salary Over Time")
plt.ylabel("Salary in USD")
plt.show()

features = [
    'experience_level',
    'employment_type',
    'job_title',
    'employee_residence',
    'remote_ratio',
    'company_location',
    'company_size'
]

X = df[features]
y = df['salary_in_usd']

X_encoded = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)


knn = KNeighborsRegressor(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

print("KNN R2:", r2_score(y_test, y_pred_knn))
print("KNN RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_knn)))

# Improvement 1: Reduce job title categories

top_jobs = df['job_title'].value_counts().nlargest(10).index

df_improved = df.copy()
df_improved['job_title'] = df_improved['job_title'].apply(
    lambda x: x if x in top_jobs else 'Other'
)

print(df_improved['job_title'].value_counts())
