# Customer 360 View Module

from pyspark.sql import DataFrame, functions as F
from typing import Dict, List


class Customer360Service:
    def build_customer_profiles(self, customers_df: DataFrame, 
                              orders_df: DataFrame, 
                              interactions_df: DataFrame) -> DataFrame:
        """Create comprehensive customer 360-degree view"""
        order_stats = orders_df.groupBy("customer_id").agg(
            F.count("*").alias("order_count"),
            F.sum("total_amount").alias("lifetime_value"),
            F.avg("total_amount").alias("avg_order_value"),
            F.max("order_date").alias("last_order_date")
        )
        
        interaction_stats = interactions_df.groupBy("customer_id").agg(
            F.count("*").alias("interaction_count"),
            F.countDistinct("channel").alias("channels_used"),
            F.max("interaction_date").alias("last_interaction_date")
        )
        
        return customers_df.join(
            order_stats,
            "customer_id",
            "left"
        ).join(
            interaction_stats,
            "customer_id",
            "left"
        ).withColumn(
            "days_since_last_order",
            F.datediff(F.current_date(), "last_order_date")
        ).withColumn(
            "customer_segment",
            F.when(F.col("lifetime_value") > 1000, "VIP")
            .when(F.col("lifetime_value") > 500, "Loyal")
            .otherwise("Standard")
        )