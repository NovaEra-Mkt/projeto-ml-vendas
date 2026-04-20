import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("dados.csv")

X = df[["Qtd Vendas (cupons)"]]
y = df["Vlr Venda"]

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, "modelo.pkl")

print("Modelo treinado com sucesso!")