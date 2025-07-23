import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error



def calcular_metricas(real, pred):
    """Calcula MSE, MAE y SMAPE entre dos series."""
    real = np.array(real)
    pred = np.array(pred)

    # Filtrar valores válidos 
    mask = ~np.isnan(real) & ~np.isnan(pred)
    real = real[mask]
    pred = pred[mask]

    mse = mean_squared_error(real, pred)
    mae = mean_absolute_error(real, pred)
    smape = 100 * np.mean(2 * np.abs(real - pred) / (np.abs(real) + np.abs(pred) + 1e-8))

    return {
        "MSE": round(mse, 2),
        "MAE": round(mae, 2),
        "SMAPE": round(smape, 2)
    }
