import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ForecasterARIMA:
    def __init__(self, fecha_col, valor_col, horizonte):
        self.fecha_col = fecha_col
        self.valor_col = valor_col
        self.horizonte = horizonte

    def entrenar_y_predecir(self, df_serie):
        df_serie = df_serie[[self.fecha_col, self.valor_col]].copy()
        df_serie[self.fecha_col] = pd.to_datetime(df_serie[self.fecha_col], errors="coerce")
        df_serie[self.valor_col] = pd.to_numeric(df_serie[self.valor_col], errors="coerce").ffill()
        df_serie = df_serie.dropna().sort_values(self.fecha_col)
        df_serie = df_serie.groupby(self.fecha_col).sum().asfreq("D").ffill()
        y = df_serie[self.valor_col]

        mejor_aic = float("inf")
        mejores_params = (1, 1, 1)
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        modelo = SARIMAX(y, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
                        resultado = modelo.fit(disp=False)
                        if resultado.aic < mejor_aic:
                            mejor_aic = resultado.aic
                            mejores_params = (p, d, q)
                    except:
                        continue

        p, d, q = mejores_params
        modelo_final = SARIMAX(y, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
        modelo_fit = modelo_final.fit(disp=False)

    
        fechas_futuras = pd.date_range(df_serie.index[-1] + pd.Timedelta(days=1), periods=self.horizonte, freq="D")
        pred_futuro = modelo_fit.forecast(steps=self.horizonte)

        df_pred_futuro = pd.DataFrame({
            self.fecha_col: fechas_futuras,
            "Predicción_ARIMA": pred_futuro
        })

      
        fitted = modelo_fit.fittedvalues
        fitted.index = df_serie.index[-len(fitted):]

        valores = fitted.values.copy()

        if len(valores) > 1 and valores[0] == 0:
            valores = np.roll(valores, -1)       
            valores[-1] = valores[-2]           
            valores = valores[:len(fitted.index)]  

        df_pred_hist = pd.DataFrame({
            self.fecha_col: fitted.index,
            "Predicción_ARIMA": valores
        })

        df_total = pd.concat([df_pred_hist, df_pred_futuro], ignore_index=True)
        return df_total