#Databricks Connector

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from typing import Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from utils.logger import CustomLogger
from configs.databricks_config import DatabricksConfig


@dataclass
class SparkConfig:
    app_name: str
    master: str = "local[*]"
    config: Optional[Dict[str, Any]] = None
    enable_hive_support: bool = False
    enable_delta: bool = True


class BaseSparkConnector(ABC):
    def __init__(self, config: SparkConfig):
        self.config = config
        self.logger = CustomLogger.get_logger(__name__)

    @abstractmethod
    def create_session(self) -> SparkSession:
        pass


class DatabricksSparkConnector(BaseSparkConnector):
    def __init__(self, config: SparkConfig):
        super().__init__(config)
        self._validate_config()

    def _validate_config(self):
        if not self.config.app_name:
            raise ValueError("Spark application name must be provided")
        if not isinstance(self.config.enable_delta, bool):
            raise TypeError("enable_delta must be a boolean value")

    def create_session(self) -> SparkSession:
        try:
            builder = SparkSession.builder.appName(self.config.app_name)
            
            if self.config.master:
                builder = builder.master(self.config.master)
                
            if self.config.config:
                for key, value in self.config.config.items():
                    builder = builder.config(key, value)
                    
            if self.config.enable_hive_support:
                builder = builder.enableHiveSupport()
                
            if self.config.enable_delta:
                builder = configure_spark_with_delta_pip(builder)
                
            spark = builder.getOrCreate()
            self.logger.info(f"Spark session created successfully with app name: {self.config.app_name}")
            return spark
            
        except Exception as e:
            self.logger.error(f"Failed to create Spark session: {str(e)}")
            raise