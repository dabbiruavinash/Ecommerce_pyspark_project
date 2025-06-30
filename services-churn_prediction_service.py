# Customer Churn Prediction

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from typing import List


class ChurnPredictionService:
    def train_churn_model(self, features_df: DataFrame, label_col: str = "is_churned") -> Pipeline:
        """Train gradient boosted trees churn prediction model"""
        feature_cols = [c for c in features_df.columns if c != label_col]
        
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features")
        
        indexer = StringIndexer(
            inputCol=label_col,
            outputCol="label")
        
        gbt = GBTClassifier(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            maxDepth=5)
        
        pipeline = Pipeline(stages=[assembler, indexer, gbt])
        return pipeline.fit(features_df)

    def calculate_churn_risk(self, model: Pipeline, customers_df: DataFrame) -> DataFrame:
        """Predict churn risk for customers"""
        return model.transform(customers_df).withColumn(
            "churn_risk",
            F.col("probability")[1]  # Probability of positive class (churned)
        )