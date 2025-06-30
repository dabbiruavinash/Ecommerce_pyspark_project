#Delta Lake Service

from pyspark.sql import SparkSession, DataFrame
from delta.tables import DeltaTable
from typing import Union, Optional, Dict, List
from utils.logger import CustomLogger
from models.base_model import BaseDataModel
from datetime import datetime
import json


class DeltaLakeService:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = CustomLogger.get_logger(__name__)

    def write_delta_table(
        self,
        df: DataFrame,
        table_path: str,
        mode: str = "overwrite",
        partition_by: Optional[List[str]] = None,
        merge_schema: bool = True,
        overwrite_schema: bool = False,
        optimize_write: bool = True,
        delta_properties: Optional[Dict[str, str]] = None
    ) -> None:
        
        try:
            writer = df.write.format("delta")
            
            if partition_by:
                writer = writer.partitionBy(partition_by)
                
            if mode == "overwrite":
                if overwrite_schema:
                    writer = writer.option("overwriteSchema", "true")
                writer.mode("overwrite").save(table_path)
            elif mode == "append":
                writer.mode("append").save(table_path)
            elif mode == "merge":
                if not isinstance(df, BaseDataModel):
                    raise ValueError("For merge operations, the DataFrame must be an instance of BaseDataModel")
                self._merge_delta_table(df, table_path)
            else:
                raise ValueError(f"Unsupported write mode: {mode}")
                
            if optimize_write:
                self.spark.sql(f"OPTIMIZE delta.`{table_path}`")
                
            if delta_properties:
                self._set_delta_properties(table_path, delta_properties)
                
            self.logger.info(f"Successfully wrote data to Delta table at {table_path} in {mode} mode")
            
        except Exception as e:
            self.logger.error(f"Failed to write Delta table: {str(e)}")
            raise

    def _merge_delta_table(self, data_model: BaseDataModel, table_path: str) -> None:
        delta_table = DeltaTable.forPath(self.spark, table_path)
        merge_condition = data_model.get_merge_condition()
        
        delta_table.alias("target").merge(
            data_model.get_df().alias("source"),
            merge_condition
        ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

    def _set_delta_properties(self, table_path: str, properties: Dict[str, str]) -> None:
        for key, value in properties.items():
            self.spark.sql(f"ALTER TABLE delta.`{table_path}` SET TBLPROPERTIES ('{key}' = '{value}')")

    def read_delta_table(
        self,
        table_path: str,
        version: Optional[Union[int, str]] = None,
        timestamp: Optional[Union[str, datetime]] = None,
        schema_evolution: bool = True
    ) -> DataFrame:
        
        try:
            reader = self.spark.read.format("delta")
            
            if version is not None:
                reader = reader.option("versionAsOf", version)
            elif timestamp is not None:
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                reader = reader.option("timestampAsOf", timestamp)
                
            if not schema_evolution:
                reader = reader.option("mergeSchema", "false")
                
            df = reader.load(table_path)
            self.logger.info(f"Successfully read Delta table from {table_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to read Delta table: {str(e)}")
            raise

    def vacuum_delta_table(
        self,
        table_path: str,
        retention_hours: int = 168,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        
        try:
            vacuum_command = f"VACUUM delta.`{table_path}` RETAIN {retention_hours} HOURS"
            if dry_run:
                vacuum_command += " DRY RUN"
                
            result = self.spark.sql(vacuum_command).collect()
            result_dict = [row.asDict() for row in result]
            
            self.logger.info(f"Vacuum operation completed for Delta table at {table_path}")
            return {"status": "success", "results": result_dict}
            
        except Exception as e:
            self.logger.error(f"Failed to vacuum Delta table: {str(e)}")
            return {"status": "failed", "error": str(e)}

    def create_delta_table(
        self,
        df: DataFrame,
        table_name: str,
        database: str = "default",
        path: Optional[str] = None,
        partition_by: Optional[List[str]] = None,
        properties: Optional[Dict[str, str]] = None,
        comment: Optional[str] = None,
        overwrite: bool = False
    ) -> None:
        
        try:
            create_stmt = f"CREATE {'OR REPLACE ' if overwrite else ''}TABLE {database}.{table_name} "
            
            if path:
                create_stmt += f"USING DELTA LOCATION '{path}' "
            else:
                create_stmt += "USING DELTA "
                
            if partition_by:
                create_stmt += f"PARTITIONED BY ({', '.join(partition_by)}) "
                
            if comment:
                create_stmt += f"COMMENT '{comment}' "
                
            if properties:
                props_str = ", ".join([f"'{k}' = '{v}'" for k, v in properties.items()])
                create_stmt += f"TBLPROPERTIES ({props_str}) "
                
            self.spark.sql(create_stmt)
            df.write.insertInto(f"{database}.{table_name}", overwrite=overwrite)
            
            self.logger.info(f"Successfully created Delta table {database}.{table_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create Delta table: {str(e)}")
            raise