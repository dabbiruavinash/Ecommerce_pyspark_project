# Orchestration Service

from pyspark.sql import SparkSession
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from utils.logger import CustomLogger
from connectors.databricks_connector import DatabricksSparkConnector, SparkConfig
from services.delta_service import DeltaLakeService
from services.data_transformation_service import DataTransformationService
from connectors.synapse_connector import SynapseConnector, SynapseConnectionConfig
from models.customer_models import CustomerModel
from models.product_models import ProductModel
from models.order_models import OrderModel
import json
import os


class OrchestrationService:
    def __init__(self):
        self.logger = CustomLogger.get_logger(__name__)
        self.spark = self._initialize_spark()
        self.delta_service = DeltaLakeService(self.spark)
        self.transformation_service = DataTransformationService(self.spark)
        self.synapse_connector = self._initialize_synapse_connector()

    def _initialize_spark(self) -> SparkSession:
        config = SparkConfig(
            app_name="EcomRetailPipeline",
            config={
                "spark.sql.shuffle.partitions": "200",
                "spark.databricks.delta.optimizeWrite.enabled": "true",
                "spark.databricks.delta.autoCompact.enabled": "true",
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.sources.partitionOverwriteMode": "dynamic"
            },
            enable_hive_support=True,
            enable_delta=True
        )
        return DatabricksSparkConnector(config).create_session()

    def _initialize_synapse_connector(self) -> SynapseConnector:
        config = SynapseConnectionConfig(
            synapse_server=os.getenv("SYNAPSE_SERVER"),
            synapse_database=os.getenv("SYNAPSE_DATABASE"),
            storage_account=os.getenv("STORAGE_ACCOUNT"),
            storage_container=os.getenv("STORAGE_CONTAINER"),
            storage_path="synapse-exports",
            use_managed_identity=True
        )
        return SynapseConnector(self.spark, config)

    def run_pipeline(self, execution_date: datetime = datetime.now()) -> None:
        """Main pipeline orchestration method"""
        try:
            self.logger.info(f"Starting pipeline execution for {execution_date}")
            
            # Extract phase
            raw_customers = self._extract_customer_data(execution_date)
            raw_products = self._extract_product_data(execution_date)
            raw_orders = self._extract_order_data(execution_date)
            
            # Transform phase
            processed_customers = self._transform_customer_data(raw_customers)
            processed_products = self._transform_product_data(raw_products)
            processed_orders = self._transform_order_data(raw_orders)
            
            # Enrich and join data
            enriched_data = self._enrich_data(
                processed_customers, 
                processed_products, 
                processed_orders
            )
            
            # Load to Delta Lake
            self._load_to_delta(enriched_data)
            
            # Export to Synapse
            self._export_to_synapse(enriched_data)
            
            # Run DBT models
            self._run_dbt_models()
            
            self.logger.info("Pipeline execution completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise

    def _extract_customer_data(self, execution_date: datetime) -> DataFrame:
        """Extract customer data from source systems"""
        self.logger.info("Extracting customer data")
        
        # In a real implementation, this would connect to actual data sources
        # For demo purposes, we'll read from a sample Delta table
        customer_path = "dbfs:/mnt/ecom-retail/source/customers"
        return self.delta_service.read_delta_table(customer_path)

    def _transform_customer_data(self, raw_customers: DataFrame) -> CustomerModel:
        """Transform raw customer data into clean model"""
        self.logger.info("Transforming customer data")
        
        valid_customers, invalid_customers = self.transformation_service.process_customer_data(raw_customers)
        
        # Save invalid records for data quality monitoring
        self.delta_service.write_delta_table(
            df=invalid_customers,
            table_path="dbfs:/mnt/ecom-retail/staging/invalid_customers",
            mode="append"
        )
        
        return CustomerModel(self.spark, valid_customers)

    def _enrich_data(
        self, 
        customers: CustomerModel, 
        products: ProductModel, 
        orders: OrderModel
    ) -> Dict[str, DataFrame]:
        """Enrich and join all datasets"""
        self.logger.info("Enriching datasets")
        
        # Customer Lifetime Value calculation
        clv_df = self.transformation_service.calculate_customer_lifetime_value(
            orders.get_df(),
            customers.get_df()
        )
        
        # Product performance analysis
        product_performance_df = self.transformation_service.analyze_product_performance(
            orders.get_df(),
            products.get_df()
        )
        
        # Customer segmentation
        segmented_customers = customers.calculate_activity_segments({
            "active": (0, 30),
            "lapsing": (31, 90),
            "churned": (91, 365),
            "inactive": (366, None)
        })
        
        return {
            "customers": segmented_customers.get_df(),
            "products": product_performance_df,
            "orders": orders.get_df(),
            "customer_lifetime_value": clv_df
        }

    def _load_to_delta(self, data: Dict[str, DataFrame]) -> None:
        """Load transformed data to Delta Lake"""
        self.logger.info("Loading data to Delta Lake")
        
        # Write customers data
        self.delta_service.write_delta_table(
            df=data["customers"],
            table_path="dbfs:/mnt/ecom-retail/processed/customers",
            mode="overwrite",
            partition_by=["loyalty_tier", "value_segment"]
        )
        
        # Write products data
        self.delta_service.write_delta_table(
            df=data["products"],
            table_path="dbfs:/mnt/ecom-retail/processed/products",
            mode="overwrite",
            partition_by=["category"]
        )
        
        # Write orders data
        self.delta_service.create_delta_table(
            df=data["orders"],
            table_name="orders",
            database="ecom_retail",
            path="dbfs:/mnt/ecom-retail/processed/orders",
            partition_by=["year", "month"],
            properties={
                "delta.enableChangeDataFeed": "true",
                "delta.autoOptimize.optimizeWrite": "true"
            }
        )

    def _export_to_synapse(self, data: Dict[str, DataFrame]) -> None:
        """Export data to Azure Synapse in Parquet format"""
        self.logger.info("Exporting data to Azure Synapse")
        
        # Export customers data
        self.synapse_connector.export_to_parquet(
            df=data["customers"],
            output_path="processed/customers",
            partition_by=["loyalty_tier"],
            compression="snappy"
        )
        
        # Export products data
        self.synapse_connector.write_to_synapse(
            df=data["products"],
            table_name="analytics.dim_products",
            write_mode="overwrite",
            options={
                "maxStrLength": "4000",
                "batchSize": "100000"
            }
        )
        
        # Export orders data
        self.synapse_connector.write_to_synapse(
            df=data["orders"],
            table_name="analytics.fact_orders",
            write_mode="append",
            options={
                "maxStrLength": "4000",
                "batchSize": "100000"
            }
        )

    def _run_dbt_models(self) -> None:
        """Execute DBT models for further transformations"""
        self.logger.info("Running DBT models")
        
        try:
            from dbt.cli.main import dbtRunner
            
            dbt = dbtRunner()
            cli_args = [
                "run",
                "--project-dir", "./dbt_models",
                "--profiles-dir", "./dbt_models",
                "--target", "prod"
            ]
            
            res = dbt.invoke(cli_args)
            if not res.success:
                raise RuntimeError("DBT run failed")
                
            self.logger.info("DBT models executed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to run DBT models: {str(e)}")
            raise


if __name__ == "__main__":
    pipeline = OrchestrationService()
    pipeline.run_pipeline()