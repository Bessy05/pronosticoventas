import pandas as pd
import numpy as np
from prophet import Prophet


class ForecasterProphet:
    def __init__(self, fecha_col="Fecha", valor_col="Total_Venta", horizonte=90):
        self.fecha_col = fecha_col
        self.valor_col = valor_col
        self.horizonte = horizonte

    def entrenar_y_predecir(self, df):
        try:
            
            df = df[[self.fecha_col, self.valor_col]].copy()
            df[self.fecha_col] = pd.to_datetime(df[self.fecha_col])
            df = df.sort_values(self.fecha_col)

            
            q80 = df[self.valor_col].quantile(0.80)
            df[self.valor_col] = np.clip(df[self.valor_col], 0, q80)

           
            df["mes"] = df[self.fecha_col].dt.month
            df["dia_semana"] = df[self.fecha_col].dt.weekday
            df["fin_de_mes"] = df[self.fecha_col].dt.is_month_end.astype(int)

            df_prophet = df.rename(columns={self.fecha_col: "ds", self.valor_col: "y"})

          
            seasonality_mode = "multiplicative" if df_prophet["y"].min() > 0 and df_prophet["y"].max() > 1000 else "additive"
            changepoint_prior_scale = 0.5 if df_prophet["y"].std() > 500 else 0.2

            corr_mes = df["mes"].corr(df[self.valor_col])
            corr_dia = df["dia_semana"].corr(df[self.valor_col])
            corr_fin = df["fin_de_mes"].corr(df[self.valor_col])
            usar_regresores = any(abs(c) > 0.1 for c in [corr_mes, corr_dia, corr_fin])

          
            modelo = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale
            )
            modelo.add_seasonality(name="mensual", period=30.5, fourier_order=5)

            if usar_regresores:
                for col in ["mes", "dia_semana", "fin_de_mes"]:
                    modelo.add_regressor(col)
                modelo.fit(df_prophet[["ds", "y", "mes", "dia_semana", "fin_de_mes"]])
            else:
                modelo.fit(df_prophet[["ds", "y"]])

          
            df_futuro = modelo.make_future_dataframe(periods=self.horizonte)
            if usar_regresores:
                df_futuro["mes"] = df_futuro["ds"].dt.month
                df_futuro["dia_semana"] = df_futuro["ds"].dt.weekday
                df_futuro["fin_de_mes"] = df_futuro["ds"].dt.is_month_end.astype(int)

            forecast = modelo.predict(df_futuro)

      
            resultado = forecast[["ds", "yhat"]].copy()
            resultado.rename(columns={"ds": self.fecha_col, "yhat": "Predicción_Prophet"}, inplace=True)
            return resultado

        except Exception as e:
            print(f"Prophet falló: {e}")
            return pd.DataFrame(columns=[self.fecha_col, "Predicción_Prophet"])