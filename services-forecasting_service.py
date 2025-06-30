# Time Series Forecasting

from pyspark.sql import DataFrame, functions as F
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd


class ForecastingService:
    def forecast_sales_trends(self, sales_df: DataFrame, time_col: str = "date", value_col: str = "sales") -> Dict:
        """Forecast sales using ARIMA models per product"""
        results = {}
        for product in sales_df.select("product_id").distinct().collect():
            product_id = product["product_id"]
            pdf = sales_df.filter(F.col("product_id") == product_id) \
                .orderBy(time_col) \
                .select(time_col, value_col) \
                .toPandas()
            
            model = ARIMA(pdf[value_col], order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=7)  # 7-day forecast
            
            results[product_id] = {
                "forecast": forecast.tolist(),
                "aic": model_fit.aic
            }
        return results