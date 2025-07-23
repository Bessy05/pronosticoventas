Pronóstico de Ventas

Este proyecto realiza pronósticos de ventas por región y producto utilizando tres modelos: **Prophet**, **ARIMA** y **XGBoost**. Los resultados se exportan en formato Excel y PDF con métricas de evaluación y gráficos comparativos.

Características

- Agrupación de ventas por fecha, región y producto
- Predicción futura con horizonte configurable
- Comparación entre modelos con métricas (MSE, MAE, SMAPE)
- Exportación automática de resultados
- Visualización clara con subgráficos por modelo

Modelos utilizados

- Prophet: Captura tendencias y estacionalidades suaves
- ARIMA: Modela dependencias temporales lineales
- XGBoost: Aprende patrones complejos con regresores y memoria


Estructura del proyecto:

forecast_ventas/ │ ├── forecaster/              
Clases de cada modelo ├── forecaster_prophet.py │   ├── forecaster_arima.py │   └── forecaster_xgboost.py │├── resultados_forecaster/  
# Archivos generados ├── data/                   
# Datos de entrada ├── ejecutar_forecast.py    
# Script principal ├── ventas_forecaster.py    
# Clase central del flujo ├── utils.py                 
# Funciones auxiliares └── README.md

Cómo ejecutar

1. Coloca tu archivo Excel en la carpeta `data/` o use el que le doy son datos ficticios,datos reales podrian tener otro tipo de comportamiento en los modelos.
2. Asegúrate de tener Python 3.11 y las librerías necesarias:

`bash`
`pip install pandas matplotlib prophet xgboost statsmodels openpyxl`

- Ejecuta el script desde terminal con:
python ejecutar_forecast.py

salida:
- Region_Producto.xlsx: contiene entrenamiento, futuro y métricas
- Region_Producto_grafico.pdf: gráfico con subplots por model

Nota:Prophet tiene un error muy alto no tomar en cuenta para decisiones modelos con errores altos 

