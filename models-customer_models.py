# Customer Model

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, when, sha2, concat_ws, current_timestamp, date_format
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from models.base_model import BaseDataModel
from utils.validators import validate_email, validate_phone
import hashlib


@dataclass
class CustomerProfile:
    customer_id: str
    first_name: str
    last_name: str
    email: str
    phone: str
    address: Dict[str, str]
    registration_date: datetime
    loyalty_tier: str
    demographics: Dict[str, str]
    preferences: Dict[str, List[str]]]
    last_activity: datetime


class CustomerModel(BaseDataModel):
    def __init__(self, spark, df: DataFrame):
        super().__init__(spark, df)
        self.required_columns = [
            "customer_id", "first_name", "last_name", "email", 
            "phone", "registration_date", "loyalty_tier"
        ]
        self.primary_key = "customer_id"
        self._validate_schema()

    def _validate_schema(self):
        missing_cols = [col for col in self.required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CustomerModel: {missing_cols}")

    def anonymize_pii(self, salt: str) -> 'CustomerModel':
        """Anonymize personally identifiable information using salted hashing"""
        self.df = self.df.withColumn(
            "email_hash",
            sha2(concat_ws("|", col("email"), lit(salt)), 256)
        ).withColumn(
            "phone_hash",
            sha2(concat_ws("|", col("phone"), lit(salt)), 256)
        ).drop("email", "phone")
        return self

    def validate_contact_info(self) -> Tuple[DataFrame, DataFrame]:
        """Validate email and phone formats, return valid and invalid records"""
        valid_df = self.df.filter(
            validate_email(col("email")) & validate_phone(col("phone"))
        )
        invalid_df = self.df.subtract(valid_df)
        return valid_df, invalid_df

    def enrich_with_external_data(self, external_df: DataFrame, join_cols: List[str]) -> 'CustomerModel':
        """Enrich customer data with external data sources"""
        self.df = self.df.join(
            external_df,
            on=join_cols,
            how="left"
        )
        return self

    def calculate_activity_segments(self, activity_rules: Dict[str, Tuple[int, int]]]) -> 'CustomerModel':
        """Segment customers based on activity rules"""
        for segment, (min_days, max_days) in activity_rules.items():
            self.df = self.df.withColumn(
                f"is_{segment}",
                when(
                    (col("days_since_last_activity") >= min_days) & 
                    (col("days_since_last_activity") <= max_days),
                    lit(True)
                ).otherwise(lit(False))
        return self

    def get_merge_condition(self) -> str:
        """Get merge condition for Delta Lake merge operations"""
        return "target.customer_id = source.customer_id"

    def to_delta(self, delta_service, table_path: str, mode: str = "overwrite"):
        """Write customer data to Delta Lake"""
        delta_service.write_delta_table(
            df=self.df,
            table_path=table_path,
            mode=mode,
            partition_by=["loyalty_tier"],
            delta_properties={
                "delta.autoOptimize.optimizeWrite": "true",
                "delta.autoOptimize.autoCompact": "true"
            }
        )

    @classmethod
    def from_delta(cls, spark, delta_service, table_path: str, version: Optional[int] = None):
        """Create CustomerModel from Delta table"""
        df = delta_service.read_delta_table(table_path, version=version)
        return cls(spark, df)