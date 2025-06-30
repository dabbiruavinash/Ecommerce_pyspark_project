# A/B Testing Module

from pyspark.sql import DataFrame, functions as F
from pyspark.ml.stat import ChiSquareTest
import numpy as np
from scipy import stats


class ABTestingService:
    def analyze_experiment(self, experiment_df: DataFrame, variant_col: str = "variant", success_col: str = "converted") -> Dict:
        """Calculate statistical significance of A/B test results"""
        contingency_table = experiment_df.groupBy(variant_col) \
            .agg(
                F.sum(F.when(F.col(success_col), 1).otherwise(0)).alias("success"),
                F.count("*").alias("total")
            ).orderBy(variant_col) \
            .collect()
        
        observed = np.array([
            [r["success"], r["total"] - r["success"]] 
            for r in contingency_table
        ])
        
        chi2, p, dof, expected = stats.chi2_contingency(observed)
        
        return {
            "chi_square": chi2,
            "p_value": p,
            "is_significant": p < 0.05,
            "conversion_rates": {
                r[variant_col]: r["success"] / r["total"] 
                for r in contingency_table
            }
        }