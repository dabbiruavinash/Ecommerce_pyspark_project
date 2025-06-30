# Real-time Analytics Module

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.streaming import DataStreamWriter
from typing import Optional, Dict


class RealtimeAnalyticsService:
    def process_clickstream(self, stream_df: DataFrame) -> DataStreamWriter:
        """Process real-time clickstream data"""
        return stream_df.withColumn(
            "event_time",
            F.to_timestamp(F.col("event_timestamp"))
        ).withWatermark(
            "event_time",
            "5 minutes"
        ).groupBy(
            F.window("event_time", "1 minute", "30 seconds"),
            "user_id",
            "page_category"
        ).agg(
            F.count("*").alias("page_views"),
            F.countDistinct("product_id").alias("unique_products_viewed")
        )

    def create_kafka_sink(self, df: DataFrame, config: Dict) -> DataStreamWriter:
        """Create Kafka sink for processed streaming data"""
        return df.select(
            F.to_json(F.struct("*")).alias("value")
        ).writeStream.format("kafka") \
            .option("kafka.bootstrap.servers", config["bootstrap_servers"]) \
            .option("topic", config["topic"]) \
            .option("checkpointLocation", config["checkpoint_location"]) \
            .outputMode(config.get("output_mode", "update"))