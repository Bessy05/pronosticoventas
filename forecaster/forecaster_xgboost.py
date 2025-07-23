from xgboost import XGBRegressor
import pandas as pd
import numpy as np


class ForecasterXGBoost:
    def __init__(self, fecha_col="Fecha", valor_col="Total_Venta", horizonte=90):
        self.fecha_col = fecha_col
        self.valor_col = valor_col
        self.horizonte = horizonte

    def entrenar_y_predecir(self, df):
        df = df[[self.fecha_col, self.valor_col]].copy()
        df[self.fecha_col] = pd.to_datetime(df[self.fecha_col])
        df = df.sort_values(self.fecha_col)

        df["día"] = (df[self.fecha_col] - df[self.fecha_col].min()).dt.days
        df["mes"] = df[self.fecha_col].dt.month
        df["año"] = df[self.fecha_col].dt.year
        df["dia_semana"] = df[self.fecha_col].dt.weekday

        X = df[["día", "mes", "año", "dia_semana"]]
        y = df[self.valor_col]

        modelo = XGBRegressor(n_estimators=100, random_state=42)
        modelo.fit(X, y)

        df["Predicción_XGBoost"] = modelo.predict(X)

        última_fecha = df[self.fecha_col].max()
        día_final = df["día"].max()

        fechas_futuras = pd.date_range(última_fecha + pd.Timedelta(days=1), periods=self.horizonte)

        df_futuro = pd.DataFrame({self.fecha_col: fechas_futuras})
        df_futuro["día"] = np.arange(día_final + 1, día_final + self.horizonte + 1)
        df_futuro["mes"] = df_futuro[self.fecha_col].dt.month
        df_futuro["año"] = df_futuro[self.fecha_col].dt.year
        df_futuro["dia_semana"] = df_futuro[self.fecha_col].dt.weekday

        df_futuro["Predicción_XGBoost"] = modelo.predict(df_futuro[["día", "mes", "año", "dia_semana"]])

       
        df_total = pd.concat([df[[self.fecha_col, "Predicción_XGBoost"]], df_futuro], ignore_index=True)
        return df_total
