# Order Processing Service

from pyspark.sql import DataFrame, functions as F, Window
from typing import List, Dict, Tuple
from models.order_models import OrderModel
from utils.decorators import timeit
from datetime import datetime, timedelta
import pandas as pd


class OrderProcessingService:
    def __init__(self, spark):
        self.spark = spark

    @timeit
    def process_order_timelines(self, orders_df: DataFrame) -> DataFrame:
        """Calculate order processing timelines and SLA compliance"""
        return orders_df.withColumn(
            "processing_time_hours",
            (F.unix_timestamp("shipped_date") - F.unix_timestamp("order_date")) / 3600
        ).withColumn(
            "sla_compliant",
            F.when(
                F.col("processing_time_hours") <= F.col("promised_sla_hours"),
                True
            ).otherwise(False)
        )

    def detect_order_anomalies(self, order_model: OrderModel) -> Tuple[DataFrame, DataFrame]:
        """Identify anomalous orders using statistical methods"""
        df = order_model.get_df()
        stats = df.agg(
            F.avg("total_amount").alias("avg_order"),
            F.stddev("total_amount").alias("stddev_order")
        ).collect()[0]
        
        threshold = stats["avg_order"] + 3 * stats["stddev_order"]
        
        anomalies = df.filter(F.col("total_amount") > threshold)
        normal = df.subtract(anomalies)
        
        return normal, anomalies

    def calculate_customer_order_sequences(self, orders_df: DataFrame) -> DataFrame:
        """Calculate order sequences and intervals for customers"""
        window_spec = Window.partitionBy("customer_id").orderBy("order_date")
        
        return orders_df.withColumn(
            "order_sequence",
            F.row_number().over(window_spec)
        ).withColumn(
            "days_since_last_order",
            F.datediff(
                "order_date",
                F.lag("order_date", 1).over(window_spec)
            )
        )