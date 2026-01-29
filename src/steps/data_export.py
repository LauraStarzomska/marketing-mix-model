"""Data Export Step - Save results to local files and/or GCS"""

import pandas as pd
import json
import logging
import pickle
from pathlib import Path
from typing import Dict
from io import BytesIO, StringIO

from src.mmm_model import MMModel

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataExportStep:
    """Export pipeline results to files"""
    
    def __init__(self, config: Dict):
        """
        Initialize data export step
        
        Args:
            config: Configuration dict with:
                - output_dir: Directory for output files
                - output_prefix: Prefix for output files
                - save_results: Whether to save (default True)
        """
        self.config = config
    
    def execute(self, 
                df_processed: pd.DataFrame,
                model: MMModel,
                results: Dict) -> Dict:
        """
        Execute data export
        
        Args:
            df_processed: Processed data
            model: Trained model
            results: All results (metrics, elasticity, contribution)
            
        Returns:
            Dictionary with output file paths
        """
        if not self.config.get('save_results', True):
            logger.info("Skipping results save")
            return {}
        
        logger.info("[5/5] Saving results...")
        
        output_files = {}
        
        # Save to local filesystem
        if self.config.get('local', {}).get('enabled', True):
            local_files = self._save_local(df_processed, model, results)
            output_files['local'] = local_files
        
        # Save to GCS
        if self.config.get('gcs', {}).get('enabled', False):
            gcs_files = self._save_to_gcs(df_processed, model, results)
            output_files['gcs'] = gcs_files
        
        return output_files
    
    def _save_local(self, df_processed: pd.DataFrame, 
                   model: MMModel, results: Dict) -> Dict:
        """Save results to local filesystem"""
        output_dir = Path(self.config.get('local', {}).get('dir', 
                         self.config.get('dir', self.config.get('output_dir', 'models/results/'))))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = self.config.get('output_prefix', 'mmm_results')
        output_files = {}
        
        # Save JSON results
        json_path = output_dir / f"{prefix}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✓ Saved results to {json_path}")
        output_files['results_json'] = str(json_path)
        
        # Save processed data
        csv_path = output_dir / f"{prefix}_processed_data.csv"
        df_processed.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved processed data to {csv_path}")
        output_files['processed_data_csv'] = str(csv_path)
        
        # Save model object
        model_path = output_dir / f"{prefix}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✓ Saved model to {model_path}")
        output_files['model_pickle'] = str(model_path)
        
        return output_files
    
    def _save_to_gcs(self, df_processed: pd.DataFrame, 
                    model: MMModel, results: Dict) -> Dict:
        """Save results to Google Cloud Storage"""
        if not GCS_AVAILABLE:
            logger.warning("google-cloud-storage not installed. Skipping GCS export.")
            logger.warning("Install with: pip install google-cloud-storage")
            return {}
        
        gcs_config = self.config.get('gcs', {})
        bucket_name = gcs_config.get('bucket')
        
        if not bucket_name:
            logger.warning("GCS bucket not configured. Skipping GCS export.")
            return {}
        
        try:
            storage_client = storage.Client(project=gcs_config.get('project'))
            bucket = storage_client.bucket(bucket_name)
            
            prefix = self.config.get('output_prefix', 'mmm_results')
            gcs_path = gcs_config.get('path', 'mmm_pipeline_outputs/')
            
            output_files = {}
            
            # Save JSON results
            json_name = f"{gcs_path}{prefix}_results.json"
            json_blob = bucket.blob(json_name)
            json_blob.upload_from_string(
                json.dumps(results, indent=2),
                content_type='application/json'
            )
            logger.info(f"✓ Uploaded results to gs://{bucket_name}/{json_name}")
            output_files['results_json'] = f"gs://{bucket_name}/{json_name}"
            
            # Save processed data
            csv_name = f"{gcs_path}{prefix}_processed_data.csv"
            csv_blob = bucket.blob(csv_name)
            csv_buffer = StringIO()
            df_processed.to_csv(csv_buffer, index=False)
            csv_blob.upload_from_string(
                csv_buffer.getvalue(),
                content_type='text/csv'
            )
            logger.info(f"✓ Uploaded processed data to gs://{bucket_name}/{csv_name}")
            output_files['processed_data_csv'] = f"gs://{bucket_name}/{csv_name}"
            
            # Save model object
            model_name = f"{gcs_path}{prefix}_model.pkl"
            model_blob = bucket.blob(model_name)
            model_buffer = BytesIO()
            pickle.dump(model, model_buffer)
            model_blob.upload_from_string(
                model_buffer.getvalue(),
                content_type='application/octet-stream'
            )
            logger.info(f"✓ Uploaded model to gs://{bucket_name}/{model_name}")
            output_files['model_pickle'] = f"gs://{bucket_name}/{model_name}"
            
            return output_files
            
        except Exception as e:
            logger.error(f"GCS upload failed: {type(e).__name__}: {e}")
            return {}
