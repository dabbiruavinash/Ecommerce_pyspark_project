# Pricing Analytics Service 

from pyspark.sql import DataFrame, functions as F
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.feature import StringIndexer, VectorAssembler
from typing import List, Dict
import numpy as np


class PricingAnalyticsService:
    def estimate_price_elasticity(self, sales_df: DataFrame) -> Dict:
        """Calculate price elasticity of demand by product"""
        glr = GeneralizedLinearRegression(
            family="gaussian",
            link="identity",
            featuresCol="features",
            labelCol="quantity_sold")
        
        results = {}
        for product_id in sales_df.select("product_id").distinct().collect():
            product_data = sales_df.filter(F.col("product_id") == product_id.row.product_id)
            
            assembler = VectorAssembler(
                inputCols=["price"],
                outputCol="features")
            
            pipeline = Pipeline(stages=[assembler, glr])
            model = pipeline.fit(product_data)
            
            beta = model.stages[-1].coefficients[0]
            avg_price = product_data.agg(F.avg("price")).collect()[0][0]
            avg_quantity = product_data.agg(F.avg("quantity_sold")).collect()[0][0]
            
            elasticity = (beta * avg_price) / avg_quantity
            results[product_id.row.product_id] = elasticity
            
        return results

    def optimize_pricing(self, products_df: DataFrame, elasticity: Dict, margin_constraint: float = 0.3) -> DataFrame:
        """Calculate optimal prices based on elasticity and margin constraints"""
        return products_df.withColumn(
            "optimal_price",
            F.when(
                F.col("cost").isNotNull(),
                F.col("cost") / (1 - margin_constraint) * 
                (1 + 1/F.lit(elasticity.get(F.col("product_id"), -2.0)))
            ).otherwise(F.col("price"))
        )