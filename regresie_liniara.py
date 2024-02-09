import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# data load
file_path = 'C:/Users/Iulian/Downloads/bodyfat.csv'
data = pd.read_csv(file_path)

X = data.drop(['BodyFat', 'Density'], axis=1)  #predictor variables
y = data['BodyFat'] # target variable



# data training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting values for the test set
y_pred = model.predict(X_test)

# mean squared error and coefficient of determination (R^2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Graph of predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted BodyFat')
plt.show()

print(f'MSE: {mse}')
print(f'R^2: {r2}')
