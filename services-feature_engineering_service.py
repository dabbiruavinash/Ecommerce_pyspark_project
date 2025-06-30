# ML Feature Engineering

from pyspark.sql import DataFrame, functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from typing import List, Dict


class FeatureEngineeringService:
    def create_feature_pipeline(self, categorical_cols: List[str], numeric_cols: List[str]) -> Pipeline:
        """Create ML feature engineering pipeline"""
        indexers = [
            StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep")
            for c in categorical_cols
        ]
        
        encoders = [
            OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_encoded")
            for c in categorical_cols
        ]
        
        assembler = VectorAssembler(
            inputCols=[f"{c}_encoded" for c in categorical_cols] + numeric_cols,
            outputCol="features")
        
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features")
        
        return Pipeline(stages=indexers + encoders + [assembler, scaler]))

    def calculate_time_based_features(self, df: DataFrame, time_col: str) -> DataFrame:
        """Extract temporal features from timestamp column"""
        return df.withColumn("hour_of_day", F.hour(time_col)) \
            .withColumn("day_of_week", F.dayofweek(time_col)) \
            .withColumn("is_weekend", F.dayofweek(time_col).isin([1,7])) \
            .withColumn("month", F.month(time_col)) \
            .withColumn("quarter", F.quarter(time_col))