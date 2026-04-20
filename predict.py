import pandas as pd
import joblib

model = joblib.load("modelo.pkl")

novo = pd.DataFrame({
    "Qtd Vendas (cupons)": [600000]
})

pred = model.predict(novo)

print(f"Previsão: R$ {pred[0]:,.2f}")