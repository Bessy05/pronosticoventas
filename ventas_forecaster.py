import os
import pandas as pd
import matplotlib.pyplot as plt

from forecaster.forecaster_prophet import ForecasterProphet
from forecaster.forecaster_arima import ForecasterARIMA
from forecaster.forecaster_xgboost import ForecasterXGBoost
from utils import calcular_metricas


class VentasForecaster:
    def __init__(self, path_excel, fecha_col="Fecha", valor_col="Total_Venta",
                 region_col="Region", producto_col="Producto", horizonte=90):
        self.path_excel = path_excel
        self.fecha_col = fecha_col
        self.valor_col = valor_col
        self.region_col = region_col
        self.producto_col = producto_col
        self.horizonte = horizonte

        self.df = pd.read_excel(path_excel)
        self.df[fecha_col] = pd.to_datetime(self.df[fecha_col])
        self.segmentos = self.df[[region_col, producto_col]].drop_duplicates().values.tolist()

        self.modelos = {
            "Prophet": ForecasterProphet(fecha_col, valor_col, horizonte),
            "ARIMA": ForecasterARIMA(fecha_col, valor_col, horizonte),
            "XGBoost": ForecasterXGBoost(fecha_col, valor_col, horizonte)
        }

    def preparar_serie(self, region, producto):
        df_seg = self.df[
            (self.df[self.region_col] == region) &
            (self.df[self.producto_col] == producto)
        ]
        df_agg = df_seg.groupby(self.fecha_col)[self.valor_col].sum().reset_index()
        return df_agg.sort_values(self.fecha_col)

    def graficar_comparacion(self, df_serie, region, producto):
        df_real = df_serie.rename(columns={self.valor_col: "Real"})
        df_real[self.fecha_col] = pd.to_datetime(df_real[self.fecha_col])
        fecha_ultima_real = df_real[self.fecha_col].max()

        predicciones = {}
        metricas = {}

        for nombre, modelo in self.modelos.items():
            pred = modelo.entrenar_y_predecir(df_serie)
            pred[self.fecha_col] = pd.to_datetime(pred[self.fecha_col])
            predicciones[nombre] = pred

            col_pred = f"Predicción_{nombre}"
            df_comb = df_real.copy().merge(pred, on=self.fecha_col, how="left")
            merged = df_comb[df_comb[col_pred].notna() & df_comb["Real"].notna()]
            if not merged.empty:
                metricas[nombre] = calcular_metricas(merged["Real"], merged[col_pred])
                print(f"{nombre}: {metricas[nombre]}")
            else:
                metricas[nombre] = {"MSE": None, "MAE": None, "SMAPE": None}
                print(f"{nombre}:  Sin datos válidos para métricas")

        print("")

        colores = {
            "Prophet": "#33C15B",
            "ARIMA": "#ff7f0e",
            "XGBoost": "#E6F51B"
        }

        fig, axs = plt.subplots(len(self.modelos), 1, figsize=(12, 10), sharex=True)

        for i, (nombre, pred) in enumerate(predicciones.items()):
            col_pred = f"Predicción_{nombre}"
            color = colores.get(nombre, None)

            tramo_hist = pred[pred[self.fecha_col] <= fecha_ultima_real]
            tramo_fut = pred[pred[self.fecha_col] > fecha_ultima_real]

            axs[i].plot(df_real[self.fecha_col], df_real["Real"], label="Real", linewidth=2.5, color="#1420A3")
            axs[i].plot(tramo_hist[self.fecha_col], tramo_hist[col_pred], label=f"{nombre} (histórico)", linestyle="--", color=color)
            axs[i].plot(tramo_fut[self.fecha_col], tramo_fut[col_pred], label=f"{nombre} (futuro)", linestyle="--", color=color)

            axs[i].set_title(f"{nombre}", fontsize=12)
            axs[i].legend(loc="upper left", fontsize=9)
            axs[i].grid(True, linestyle="--", alpha=0.4)

        plt.xlabel("Fecha")
        plt.suptitle(f"Pronóstico de Ventas: {producto} en {region}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

       
        carpeta = "resultados_forecaster"
        os.makedirs(carpeta, exist_ok=True)
        base = f"{region}_{producto}".replace(" ", "_")

        df_entrenamiento = df_real[[self.fecha_col, "Real"]].copy()
        for nombre, pred in predicciones.items():
            col_pred = f"Predicción_{nombre}"
            tramo = pred[pred[self.fecha_col] <= fecha_ultima_real]
            df_entrenamiento = df_entrenamiento.merge(
                tramo[[self.fecha_col, col_pred]], on=self.fecha_col, how="left"
            )

        df_futuro = pd.DataFrame()
        for nombre, pred in predicciones.items():
            col_pred = f"Predicción_{nombre}"
            tramo_fut = pred[pred[self.fecha_col] > fecha_ultima_real][[self.fecha_col, col_pred]]
            if df_futuro.empty:
                df_futuro = tramo_fut
            else:
                df_futuro = df_futuro.merge(tramo_fut, on=self.fecha_col, how="outer")

        ruta_excel = os.path.join(carpeta, f"{base}.xlsx")
        with pd.ExcelWriter(ruta_excel) as writer:
            df_entrenamiento.to_excel(writer, sheet_name="Entrenamiento", index=False)
            df_futuro.sort_values(self.fecha_col).to_excel(writer, sheet_name="Futuro", index=False)
            pd.DataFrame(metricas).T.to_excel(writer, sheet_name="Metricas", index=True)

        ruta_pdf = os.path.join(carpeta, f"{base}_grafico.pdf")
        plt.savefig(ruta_pdf)
        plt.close()

        print(f"Exportado: {base}.xlsx + {base}_grafico.pdf\n")

    def procesar_todos_segmentos(self, max_segmentos=None):
        print("Procesando pronósticos por segmento...\n")
        segmentos = self.segmentos[:max_segmentos] if max_segmentos else self.segmentos
        for i, (region, producto) in enumerate(segmentos, 1):
            print(f"🔹 Segmento {i}/{len(segmentos)}: {region} - {producto}")
            df_serie = self.preparar_serie(region, producto)
            if len(df_serie) < 30:
                print("Serie muy corta. Se omite.\n")
                continue
            try:
                self.graficar_comparacion(df_serie, region, producto)
            except Exception as e:
                print(f"Error en {region}-{producto}: {e}\n")