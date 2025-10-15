import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('big_houses.csv')
X = df[['size', 'nb_rooms', 'price']]
y = df['garden']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Train Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")

print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

joblib.dump(model, 'logistic_model.joblib')
