"""Data Import Step - Load from GCP BigQuery or local CSV"""

import pandas as pd
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

try:
    from google.cloud import bigquery
    from google.oauth2.credentials import Credentials
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataImportStep:
    """Load data from GCP or local source"""
    
    def __init__(self, config: Dict):
        """
        Initialize data import step
        
        Args:
            config: Configuration dict with:
                - data_source: 'gcp' or 'local'
                - data_path: Path for local source
                - gcp_project: GCP project ID
                - gcp_dataset: GCP dataset name
        """
        self.config = config
        self.df = None
    
    def execute(self) -> pd.DataFrame:
        """Execute data import"""
        logger.info("[1/5] Loading data...")
        
        if self.config.get('data_source') == 'gcp':
            self._load_from_gcp()
        else:
            self._load_from_local()
        
        logger.info(f"✓ Loaded {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        return self.df
    
    def _load_from_local(self):
        """Load from local CSV file"""
        data_path = self.config.get('path', self.config.get('data_path', 'data/raw/marketing_data.csv'))
        logger.info(f"Loading from {data_path}...")
        self.df = pd.read_csv(data_path)
    
    def _load_from_gcp(self):
        """Load from GCP BigQuery using gcloud authentication"""
        if not GCP_AVAILABLE:
            raise ImportError("google.cloud.bigquery required. Install with: pip install google-cloud-bigquery")
        
        try:
            logger.info("Loading from GCP BigQuery...")
            
            # Get access token from gcloud CLI
            result = subprocess.run(
                ['gcloud', 'auth', 'print-access-token'],
                capture_output=True,
                text=True,
                check=True
            )
            token = result.stdout.strip()
            
            # Create BigQuery client
            gcp_project = self.config.get('gcp_project', 'alterdata-rekrutacja-20')
            gcp_dataset = self.config.get('gcp_dataset', 'marketing_data')
            credentials = Credentials(token=token)
            client = bigquery.Client(
                project=gcp_project,
                credentials=credentials
            )
            
            logger.info(f"Connected to GCP project: {gcp_project}")
            
            # Download cost_revenue table
            logger.info(f"Downloading {gcp_dataset}.cost_revenue...")
            query = f"""
            SELECT * 
            FROM `{gcp_project}.{gcp_dataset}.cost_revenue`
            """
            df = client.query(query).result().to_dataframe()
            logger.info(f"  ✓ cost_revenue: {len(df):,} rows, {len(df.columns)} columns")
            
            # Try to download promotions table
            try:
                logger.info(f"Downloading {gcp_dataset}.promotions...")
                query_promo = f"""
                SELECT * 
                FROM `{gcp_project}.{gcp_dataset}.promotions`
                """
                df_promotions = client.query(query_promo).result().to_dataframe()
                logger.info(f"  ✓ promotions: {len(df_promotions):,} rows, {len(df_promotions.columns)} columns")
                
                # Merge on common columns
                common_cols = set(df.columns) & set(df_promotions.columns)
                if 'week' in common_cols and 'geo' in common_cols:
                    logger.info("Merging on: week, geo")
                    df = df.merge(df_promotions, on=['week', 'geo'], how='left')
                elif len(common_cols) > 0:
                    merge_cols = list(common_cols)
                    logger.info(f"Merging on: {merge_cols}")
                    df = df.merge(df_promotions, on=merge_cols, how='left')
            except Exception as e:
                logger.warning(f"Promotions table not accessible: {type(e).__name__}")
            
            # Convert object columns to numeric (BigQuery returns strings)
            numeric_cols = [c for c in df.columns if 'cost' in c.lower() or 'revenue' in c.lower() or 'conversion' in c.lower()]
            for col in numeric_cols:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            self.df = df
            logger.info(f"✓ Final data: {len(df):,} rows, {len(df.columns)} columns")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get gcloud auth token: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"GCP BigQuery error: {type(e).__name__}: {e}")
            raise
    
    def get_metadata(self) -> Dict:
        """Get data metadata"""
        return {
            'shape': list(self.df.shape),
            'date_range': {
                'start': str(self.df['week'].min()),
                'end': str(self.df['week'].max())
            }
        }
