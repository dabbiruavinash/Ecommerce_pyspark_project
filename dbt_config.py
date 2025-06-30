# DBT Configuration and Models

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DBTConfig:
    project_dir: str
    profiles_dir: str
    target: str = "dev"
    full_refresh: bool = False
    vars: Optional[Dict[str, Any]] = None
    threads: int = 4
    schema: str = "dbt_analytics"


@dataclass
class DBTModelConfig:
    materialized: str = "table"
    schema: Optional[str] = None
    tags: Optional[List[str]] = None
    pre_hook: Optional[List[str]] = None
    post_hook: Optional[List[str]] = None
    grants: Optional[Dict[str, List[str]]] = None
    persist_docs: Optional[Dict[str, str]] = None
    full_refresh: Optional[bool] = None
    incremental_strategy: Optional[str] = None
    unique_key: Optional[Union[str, List[str]]] = None