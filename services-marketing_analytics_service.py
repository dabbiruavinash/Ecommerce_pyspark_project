# Marketing Analytics Service

from pyspark.sql import DataFrame, functions as F
from pyspark.ml.feature import RFM
from typing import List, Dict
from datetime import datetime


class MarketingAnalyticsService:
    def calculate_rfm_scores(self, customers_df: DataFrame, orders_df: DataFrame) -> DataFrame:
        """Calculate RFM (Recency, Frequency, Monetary) scores"""
        rfm = RFM(
            customerCol="customer_id",
            orderDateCol="order_date",
            amountCol="total_amount")
        
        return rfm.transform(
            orders_df.join(
                customers_df.select("customer_id", "email"),
                "customer_id"
            )
        )

    def analyze_campaign_performance(self, campaigns_df: DataFrame, orders_df: DataFrame) -> DataFrame:
        """Calculate marketing campaign ROI and conversion rates"""
        return campaigns_df.join(
            orders_df.groupBy("campaign_id").agg(
                F.sum("total_amount").alias("revenue"),
                F.countDistinct("customer_id").alias("conversions")
            ),
            "campaign_id",
            "left"
        ).withColumn(
            "roi",
            (F.col("revenue") - F.col("cost")) / F.col("cost") * 100
        ).withColumn(
            "cpa",
            F.col("cost") / F.col("conversions")
        )

    def segment_customers(self, customers_df: DataFrame, features: List[str]) -> DataFrame:
        """Perform customer segmentation using K-means clustering"""
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        from pyspark.ml.clustering import KMeans
        
        assembler = VectorAssembler(
            inputCols=features,
            outputCol="features")
        
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True)
        
        kmeans = KMeans(
            k=5,
            featuresCol="scaled_features",
            predictionCol="segment")
        
        pipeline = Pipeline(stages=[assembler, scaler, kmeans])
        model = pipeline.fit(customers_df)
        
        return model.transform(customers_df)