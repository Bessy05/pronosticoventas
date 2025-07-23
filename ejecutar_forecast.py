from ventas_forecaster import VentasForecaster


archivo = "data/datos_ventas_2020_2023.xlsx"

# Instanciar y ejecutar
forecaster = VentasForecaster(archivo)
forecaster.procesar_todos_segmentos()


