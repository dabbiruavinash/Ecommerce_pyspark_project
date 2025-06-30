# Recommendation Engine

from pyspark.sql import DataFrame, functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from typing import Dict, Optional
import numpy as np


class RecommendationService:
    def __init__(self, spark):
        self.spark = spark

    def train_collaborative_filtering(self, ratings_df: DataFrame, params: Dict) -> ALS:
        """Train ALS collaborative filtering model"""
        als = ALS(
            userCol="customer_id",
            itemCol="product_id",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True
        )
        
        param_grid = ParamGridBuilder() \
            .addGrid(als.rank, params.get("ranks", [10, 20])) \
            .addGrid(als.maxIter, params.get("max_iters", [5, 10])) \
            .addGrid(als.regParam, params.get("reg_params", [0.01, 0.1])) \
            .build()
        
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction")
        
        cv = CrossValidator(
            estimator=als,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3)
        
        model = cv.fit(ratings_df)
        return model.bestModel

    def generate_recommendations(self, model: ALS, customers: DataFrame, n: int = 5) -> DataFrame:
        """Generate top-N recommendations for customers"""
        return model.recommendForUserSubset(customers, n)

    def calculate_association_rules(self, orders_df: DataFrame, min_support: float = 0.01) -> DataFrame:
        """Calculate product association rules using FP-Growth"""
        from pyspark.ml.fpm import FPGrowth
        
        transaction_df = orders_df.groupBy("order_id") \
            .agg(F.collect_list("product_id").alias("items"))
        
        fp_growth = FPGrowth(
            itemsCol="items",
            minSupport=min_support,
            minConfidence=0.5)
        
        model = fp_growth.fit(transaction_df)
        return model.associationRules