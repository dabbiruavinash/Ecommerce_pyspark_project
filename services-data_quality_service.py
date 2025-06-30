# Data Quality Service

from pyspark.sql import DataFrame, functions as F
from typing import Dict, List, Tuple
import json


class DataQualityService:
    def __init__(self, spark):
        self.spark = spark

    def run_data_quality_checks(self, df: DataFrame, rules: Dict) -> Dict:
        """Execute data quality checks against DataFrame"""
        results = {}
        for check_name, rule in rules.items():
            if rule["type"] == "completeness":
                results[check_name] = self._check_completeness(df, rule)
            elif rule["type"] == "uniqueness":
                results[check_name] = self._check_uniqueness(df, rule)
            elif rule["type"] == "consistency":
                results[check_name] = self._check_consistency(df, rule)
        
        return {
            "summary": {
                "passed_checks": sum(1 for r in results.values() if r["passed"])),
                "failed_checks": sum(1 for r in results.values() if not r["passed"])),
                "total_checks": len(results)
            },
            "details": results
        }

    def _check_completeness(self, df: DataFrame, rule: Dict) -> Dict:
        """Check for null/missing values"""
        null_counts = df.select([
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) 
            for c in rule["columns"]
        ]).collect()[0]
        
        total_count = df.count()
        results = {
            c: {"null_count": null_counts[c], "null_percentage": null_counts[c]/total_count*100}
            for c in rule["columns"]
        }
        
        passed = all(null_counts[c]/total_count <= rule["threshold"] for c in rule["columns"])
        return {"passed": passed, "results": results}

    def _check_uniqueness(self, df: DataFrame, rule: Dict) -> Dict:
        """Check for duplicate values"""
        duplicate_counts = df.groupBy(rule["columns"]).count() \
            .filter(F.col("count") > 1) \
            .agg(F.sum("count")).collect()[0][0] or 0
        
        total_count = df.count()
        duplicate_percentage = duplicate_counts/total_count*100
        
        passed = duplicate_percentage <= rule["threshold"]
        return {
            "passed": passed,
            "results": {
                "duplicate_count": duplicate_counts,
                "duplicate_percentage": duplicate_percentage
            }
        }