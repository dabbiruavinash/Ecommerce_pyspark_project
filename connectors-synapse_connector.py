#Synapse Connector

from pyspark.sql import SparkSession, DataFrame
from typing import Optional, Dict, List
from dataclasses import dataclass
import logging
from utils.logger import CustomLogger
from configs.synapse_config import SynapseConfig
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


@dataclass
class SynapseConnectionConfig:
    synapse_server: str
    synapse_database: str
    synapse_user: Optional[str] = None
    synapse_password: Optional[str] = None
    synapse_port: int = 1433
    storage_account: Optional[str] = None
    storage_container: Optional[str] = None
    storage_path: Optional[str] = None
    use_managed_identity: bool = True


class SynapseConnector:
    def __init__(self, spark: SparkSession, config: SynapseConnectionConfig):
        self.spark = spark
        self.config = config
        self.logger = CustomLogger.get_logger(__name__)
        self._validate_config()
        self._setup_storage_credentials()

    def _validate_config(self):
        if not self.config.synapse_server:
            raise ValueError("Synapse server name must be provided")
        if not self.config.synapse_database:
            raise ValueError("Synapse database name must be provided")
        if not self.config.use_managed_identity and (not self.config.synapse_user or not self.config.synapse_password):
            raise ValueError("Either use managed identity or provide username/password")

    def _setup_storage_credentials(self):
        if self.config.use_managed_identity and self.config.storage_account:
            try:
                credential = DefaultAzureCredential()
                blob_service_client = BlobServiceClient(
                    account_url=f"https://{self.config.storage_account}.blob.core.windows.net",
                    credential=credential
                )
                self.logger.info("Successfully authenticated with Azure Storage using Managed Identity")
            except Exception as e:
                self.logger.error(f"Failed to authenticate with Azure Storage: {str(e)}")
                raise

    def write_to_synapse(
        self,
        df: DataFrame,
        table_name: str,
        write_mode: str = "overwrite",
        staging_dir: Optional[str] = None,
        options: Optional[Dict[str, str]] = None
    ) -> None:
        
        try:
            write_options = {
                "url": f"jdbc:sqlserver://{self.config.synapse_server}:{self.config.synapse_port};database={self.config.synapse_database}",
                "dbtable": table_name,
                "forwardSparkAzureStorageCredentials": "true",
                "tempDir": staging_dir or f"abfss://{self.config.storage_container}@{self.config.storage_account}.dfs.core.windows.net/{self.config.storage_path}/temp",
            }
            
            if self.config.use_managed_identity:
                write_options["authentication"] = "ActiveDirectoryMSI"
            else:
                write_options["user"] = self.config.synapse_user
                write_options["password"] = self.config.synapse_password
                
            if options:
                write_options.update(options)
                
            df.write \
                .format("com.databricks.spark.sqldw") \
                .options(**write_options) \
                .mode(write_mode) \
                .save()
                
            self.logger.info(f"Successfully wrote data to Synapse table {table_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to write to Synapse: {str(e)}")
            raise

    def read_from_synapse(
        self,
        table_name: str,
        query: Optional[str] = None,
        options: Optional[Dict[str, str]] = None
    ) -> DataFrame:
        
        try:
            read_options = {
                "url": f"jdbc:sqlserver://{self.config.synapse_server}:{self.config.synapse_port};database={self.config.synapse_database}",
                "forwardSparkAzureStorageCredentials": "true",
                "tempDir": f"abfss://{self.config.storage_container}@{self.config.storage_account}.dfs.core.windows.net/{self.config.storage_path}/temp",
            }
            
            if query:
                read_options["query"] = query
            else:
                read_options["dbtable"] = table_name
                
            if self.config.use_managed_identity:
                read_options["authentication"] = "ActiveDirectoryMSI"
            else:
                read_options["user"] = self.config.synapse_user
                read_options["password"] = self.config.synapse_password
                
            if options:
                read_options.update(options)
                
            df = self.spark.read \
                .format("com.databricks.spark.sqldw") \
                .options(**read_options) \
                .load()
                
            self.logger.info(f"Successfully read data from Synapse {'query' if query else 'table'}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to read from Synapse: {str(e)}")
            raise

    def export_to_parquet(
        self,
        df: DataFrame,
        output_path: str,
        partition_by: Optional[List[str]] = None,
        mode: str = "overwrite",
        compression: str = "snappy"
    ) -> None:
        
        try:
            full_path = f"abfss://{self.config.storage_container}@{self.config.storage_account}.dfs.core.windows.net/{output_path}"
            
            writer = df.write \
                .format("parquet") \
                .option("compression", compression) \
                .mode(mode)
                
            if partition_by:
                writer = writer.partitionBy(partition_by)
                
            writer.save(full_path)
            self.logger.info(f"Successfully exported data to Parquet at {full_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export to Parquet: {str(e)}")
            raise