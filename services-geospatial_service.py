# Geospatial Analytics

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import FloatType
from geopy.distance import great_circle
import pandas as pd


class GeospatialService:
    def __init__(self, spark):
        self.spark = spark

    @F.udf(FloatType())
    def _calculate_distance(lat1, lon1, lat2, lon2):
        return float(great_circle((lat1, lon1), (lat2, lon2)).km)

    def analyze_location_patterns(self, customers_df: DataFrame, stores_df: DataFrame) -> DataFrame:
        """Calculate nearest store for each customer"""
        cross_join = customers_df.crossJoin(
            stores_df.select(
                F.col("store_id"),
                F.col("store_lat").alias("s_lat"),
                F.col("store_lon").alias("s_lon"))
        )
        
        return cross_join.withColumn(
            "distance_km",
            self._calculate_distance(
                F.col("customer_lat"),
                F.col("customer_lon"),
                F.col("s_lat"),
                F.col("s_lon"))
        ).groupBy("customer_id").agg(
            F.min("distance_km").alias("nearest_store_distance"),
            F.first("store_id").alias("nearest_store_id")
        )