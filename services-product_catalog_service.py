# Product Catalog Service

from pyspark.sql import DataFrame, functions as F
from typing import List, Dict, Optional
from models.product_models import ProductModel
from utils.decorators import validate_schema
from utils.logger import CustomLogger
import json


class ProductCatalogService:
    def __init__(self, spark):
        self.spark = spark
        self.logger = CustomLogger.get_logger(__name__)

    @validate_schema(required_columns=["product_id", "name", "price", "category_id"])
    def process_product_hierarchy(self, products_df: DataFrame, categories_df: DataFrame) -> ProductModel:
        """Build complete product catalog with category hierarchy"""
        return ProductModel(
            self.spark,
            products_df.join(
                categories_df,
                "category_id",
                "left"
            ).withColumn(
                "full_category_path",
                F.concat_ws(" > ", 
                    F.col("category_level1"),
                    F.col("category_level2"),
                    F.col("category_level3"))
            )
        )

    def calculate_inventory_turnover(self, inventory_df: DataFrame, sales_df: DataFrame) -> DataFrame:
        """Calculate inventory turnover metrics by product"""
        return inventory_df.join(
            sales_df.groupBy("product_id").agg(
                F.sum("quantity").alias("total_sold"),
                F.countDistinct("order_id").alias("order_count")
            ),
            "product_id",
            "left"
        ).withColumn(
            "turnover_ratio",
            F.when(F.col("current_stock") > 0, 
                F.col("total_sold") / F.col("current_stock"))
            .otherwise(F.lit(0.0))
        )

    def apply_dynamic_pricing(self, product_model: ProductModel, pricing_rules: Dict) -> ProductModel:
        """Apply dynamic pricing rules to products"""
        df = product_model.get_df()
        for rule in pricing_rules:
            df = df.withColumn(
                "price",
                F.when(
                    F.expr(rule["condition"]),
                    F.col("price") * rule["multiplier"]
                ).otherwise(F.col("price"))
            )
        return ProductModel(self.spark, df)