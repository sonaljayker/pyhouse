import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("housing.csv")
X = df[["area", "bedrooms", "age"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
