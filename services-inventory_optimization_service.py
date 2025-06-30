# Inventory Optimization Service

from pyspark.sql import DataFrame, functions as F
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from typing import Dict


class InventoryOptimizationService:
    def forecast_demand(self, sales_df: DataFrame, features: List[str]) -> DataFrame:
        """Predict product demand using Random Forest regressor"""
        assembler = VectorAssembler(
            inputCols=features,
            outputCol="features")
        
        rf = RandomForestRegressor(
            featuresCol="features",
            labelCol="quantity",
            numTrees=100,
            maxDepth=5)
        
        pipeline = Pipeline(stages=[assembler, rf])
        model = pipeline.fit(sales_df)
        
        return model.transform(sales_df)

    def calculate_reorder_points(self, inventory_df: DataFrame, demand_forecast_df: DataFrame) -> DataFrame:
        """Calculate optimal reorder points based on demand forecasts"""
        return inventory_df.join(
            demand_forecast_df.select(
                "product_id",
                F.col("prediction").alias("forecasted_demand")),
            "product_id"
        ).withColumn(
            "reorder_point",
            F.ceil(F.col("forecasted_demand") * 1.2)  # 20% safety stock
        )

    def simulate_stockouts(self, inventory_df: DataFrame, sales_df: DataFrame) -> Dict:
        """Simulate potential stockout scenarios"""
        stockout_risk = inventory_df.join(
            sales_df.groupBy("product_id").agg(
                F.avg("quantity").alias("avg_daily_sales"),
                F.stddev("quantity").alias("sales_stddev")
            ),
            "product_id"
        ).withColumn(
            "stockout_probability",
            F.expr("1 - CDF('normal', current_stock, avg_daily_sales, sales_stddev)")
        )
        
        return {
            "critical": stockout_risk.filter(F.col("stockout_probability") > 0.7),
            "warning": stockout_risk.filter((F.col("stockout_probability") > 0.3) & (F.col("stockout_probability") <= 0.7)),
            "safe": stockout_risk.filter(F.col("stockout_probability") <= 0.3)
        }