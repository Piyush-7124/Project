# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Import Data
data = pd.read_csv('Admission_Predict.csv')

# Describe Data
print(data.head())
print(data.info())
print(data.describe())

# Data Visualization
sns.histplot(data['Chance of Admit '], bins=30, kde=True)
plt.title('Distribution of Admission Chances')
plt.xlabel('Chance of Admit')
plt.ylabel('Frequency')
plt.show()

sns.pairplot(data[['Chance of Admit ', 'GRE Score', 'TOEFL Score', 'University Rating']])
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Data Preprocessing
print(data.isnull().sum())
data.drop('Serial No.', axis=1, inplace=True)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop('Chance of Admit ', axis=1))

data_scaled = pd.DataFrame(scaled_features, columns=data.columns[:-1])
data_scaled['Chance of Admit '] = data['Chance of Admit ']

# Define Target Variable (y) and Feature Variables (X)
X = data_scaled.drop('Chance of Admit ', axis=1)
y = data_scaled['Chance of Admit ']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Prediction
example_data = X_test.iloc[0].values.reshape(1, -1)
predicted_chance = model.predict(example_data)
print(f'Predicted Chance of Admission: {predicted_chance[0]}')
