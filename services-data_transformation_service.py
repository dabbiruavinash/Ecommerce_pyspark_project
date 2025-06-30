#Data Transformation Service 

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, expr, udf, pandas_udf, window, date_format, to_date, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType, ArrayType
from typing import Dict, List, Optional, Tuple, Callable, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import wraps
from utils.logger import CustomLogger
from utils.decorators import timeit, log_input_output
from models.base_model import BaseDataModel


class DataTransformationService:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = CustomLogger.get_logger(__name__)

    @timeit
    def apply_complex_transformations(
        self,
        df: DataFrame,
        transformations: List[Dict[str, Any]],
        join_dfs: Optional[Dict[str, DataFrame]] = None
    ) -> DataFrame:
        """
        Apply a series of complex transformations defined in a declarative way
        :param df: Input DataFrame
        :param transformations: List of transformation definitions
        :param join_dfs: Additional DataFrames that might be needed for joins
        :return: Transformed DataFrame
        """
        transformed_df = df
        
        for transform in transformations:
            transform_type = transform.get("type")
            
            try:
                if transform_type == "add_column":
                    transformed_df = self._add_column(transformed_df, transform)
                elif transform_type == "filter":
                    transformed_df = self._apply_filter(transformed_df, transform)
                elif transform_type == "join":
                    transformed_df = self._apply_join(transformed_df, transform, join_dfs)
                elif transform_type == "aggregate":
                    transformed_df = self._apply_aggregation(transformed_df, transform)
                elif transform_type == "pivot":
                    transformed_df = self._apply_pivot(transformed_df, transform)
                elif transform_type == "window":
                    transformed_df = self._apply_window_function(transformed_df, transform)
                elif transform_type == "udf":
                    transformed_df = self._apply_udf(transformed_df, transform)
                elif transform_type == "pandas_udf":
                    transformed_df = self._apply_pandas_udf(transformed_df, transform)
                elif transform_type == "schema_change":
                    transformed_df = self._apply_schema_change(transformed_df, transform)
                else:
                    self.logger.warning(f"Unknown transformation type: {transform_type}")
                    
            except Exception as e:
                self.logger.error(f"Failed to apply transformation {transform}: {str(e)}")
                raise
                
        return transformed_df

    def _add_column(self, df: DataFrame, transform: Dict) -> DataFrame:
        column_name = transform["column_name"]
        expression = transform["expression"]
        return df.withColumn(column_name, expr(expression))

    def _apply_filter(self, df: DataFrame, transform: Dict) -> DataFrame:
        condition = transform["condition"]
        return df.filter(expr(condition))

    def _apply_join(self, df: DataFrame, transform: Dict, join_dfs: Optional[Dict[str, DataFrame]]) -> DataFrame:
        join_with = transform["join_with"]
        join_type = transform.get("join_type", "inner")
        join_condition = transform["condition"]
        
        if join_dfs and join_with in join_dfs:
            return df.join(join_dfs[join_with], on=expr(join_condition), how=join_type)
        else:
            raise ValueError(f"Join DataFrame '{join_with}' not found in provided join_dfs")

    def _apply_aggregation(self, df: DataFrame, transform: Dict) -> DataFrame:
        group_by = transform.get("group_by", [])
        aggregations = transform["aggregations"]
        
        if not isinstance(group_by, list):
            group_by = [group_by]
            
        agg_exprs = [expr(agg) for agg in aggregations]
        return df.groupBy(*group_by).agg(*agg_exprs)

    def _apply_pivot(self, df: DataFrame, transform: Dict) -> DataFrame:
        pivot_col = transform["pivot_col"]
        values = transform.get("values")
        group_by = transform.get("group_by", [])
        aggregations = transform["aggregations"]
        
        pivot_df = df.groupBy(*group_by).pivot(pivot_col, values)
        
        if isinstance(aggregations, str):
            return pivot_df.agg(expr(aggregations))
        else:
            return pivot_df.agg(*[expr(agg) for agg in aggregations])

    def _apply_window_function(self, df: DataFrame, transform: Dict) -> DataFrame:
        window_spec = transform["window_spec"]
        partition_by = window_spec.get("partition_by", [])
        order_by = window_spec.get("order_by", [])
        frame = window_spec.get("frame")
        
        if not isinstance(partition_by, list):
            partition_by = [partition_by]
            
        if not isinstance(order_by, list):
            order_by = [order_by]
            
        window_fn = window(*partition_by, *order_by)
        if frame:
            window_fn = window_fn.rowsBetween(frame["start"], frame["end"])
            
        column_name = transform["column_name"]
        function = transform["function"]
        
        return df.withColumn(column_name, function.over(window_fn))

    def _apply_udf(self, df: DataFrame, transform: Dict) -> DataFrame:
        udf_name = transform["udf_name"]
        input_cols = transform["input_cols"]
        output_col = transform["output_col"]
        return_type = transform.get("return_type", StringType())
        
        if not isinstance(input_cols, list):
            input_cols = [input_cols]
            
        udf_function = udf(lambda *args: transform["function"](*args), return_type)
        self.spark.udf.register(udf_name, udf_function)
        
        return df.withColumn(output_col, udf_function(*[col(c) for c in input_cols]))

    def _apply_pandas_udf(self, df: DataFrame, transform: Dict) -> DataFrame:
        pandas_udf_name = transform["pandas_udf_name"]
        input_cols = transform["input_cols"]
        output_col = transform["output_col"]
        return_type = transform["return_type"]
        function_type = transform.get("function_type", "SCALAR")
        
        if not isinstance(input_cols, list):
            input_cols = [input_cols]
            
        pandas_udf_function = pandas_udf(transform["function"], return_type, function_type)
        self.spark.udf.register(pandas_udf_name, pandas_udf_function)
        
        return df.withColumn(output_col, pandas_udf_function(*[col(c) for c in input_cols]))

    def _apply_schema_change(self, df: DataFrame, transform: Dict) -> DataFrame:
        schema_changes = transform["changes"]
        select_exprs = []
        
        for col_name in df.columns:
            if col_name in schema_changes:
                change = schema_changes[col_name]
                if change.get("drop", False):
                    continue
                new_name = change.get("new_name")
                new_type = change.get("new_type")
                
                if new_name and new_type:
                    select_exprs.append(col(col_name).cast(new_type).alias(new_name))
                elif new_name:
                    select_exprs.append(col(col_name).alias(new_name))
                elif new_type:
                    select_exprs.append(col(col_name).cast(new_type))
                else:
                    select_exprs.append(col(col_name))
            else:
                select_exprs.append(col(col_name))
                
        return df.select(*select_exprs)

    @log_input_output
    def process_customer_data(self, customer_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Process raw customer data into clean and enriched datasets"""
        # Data cleaning
        clean_df = customer_df.dropDuplicates(["customer_id"]) \
            .na.fill({"loyalty_tier": "standard"}) \
            .withColumn("full_name", concat_ws(" ", col("first_name"), col("last_name")))
            
        # Data validation
        valid_df = clean_df.filter(
            (col("customer_id").isNotNull()) &
            (col("email").rlike("^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$")) &
            (col("registration_date") <= current_timestamp())
        )
        
        invalid_df = clean_df.subtract(valid_df)
        
        return valid_df, invalid_df

    @log_input_output
    def transform_product_catalog(self, product_df: DataFrame, category_df: DataFrame) -> DataFrame:
        """Transform product catalog data with category hierarchy"""
        return product_df.join(
            category_df,
            on="category_id",
            how="left"
        ).groupBy("category_id", "category_name").agg(
            expr("count(*) as product_count"),
            expr("avg(price) as avg_price"),
            expr("sum(case when in_stock then 1 else 0 end) as in_stock_count")
        )

    @log_input_output
    def calculate_customer_lifetime_value(self, orders_df: DataFrame, customers_df: DataFrame) -> DataFrame:
        """Calculate customer lifetime value metrics"""
        customer_spend = orders_df.groupBy("customer_id").agg(
            expr("sum(total_amount) as total_spend"),
            expr("count(distinct order_id) as order_count"),
            expr("avg(total_amount) as avg_order_value"),
            expr("datediff(max(order_date), min(order_date)) as customer_tenure_days")
        )
        
        return customers_df.join(
            customer_spend,
            on="customer_id",
            how="left"
        ).withColumn(
            "clv_segment",
            when(col("total_spend") > 1000, "high")
            .when(col("total_spend") > 500, "medium")
            .otherwise("low")
        ).withColumn(
            "purchase_frequency",
            col("order_count") / (col("customer_tenure_days") / 30.0)
        )