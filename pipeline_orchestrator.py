"""
Pipeline Orchestrator - Coordinates execution of pipeline steps

Reads configuration and executes steps sequentially:
1. Data Import
2. Model Training (Preprocessing + Model Building)
3. Model Evaluation
4. Data Export
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict
from datetime import datetime

from src.steps import (
    DataImportStep,
    ModelTrainingStep,
    ModelEvaluationStep,
    DataExportStep
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrate MMM pipeline execution"""
    
    def __init__(self, config_path: str):
        """
        Initialize orchestrator with config file
        
        Args:
            config_path: Path to pipeline config YAML
        """
        self.config = self._load_config(config_path)
        self.results = {
            'pipeline': {
                'start_time': datetime.now().isoformat(),
                'config_file': config_path
            }
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    
    def run(self) -> Dict:
        """Execute pipeline with configured steps"""
        logger.info("="*80)
        logger.info("STARTING MMM PIPELINE")
        logger.info("="*80)
        
        try:
            # Step 1: Data Import
            if self.config.get('steps', {}).get('data_import', {}).get('enabled', True):
                self._execute_data_import()
            
            # Step 2: Model Training
            if self.config.get('steps', {}).get('model_training', {}).get('enabled', True):
                self._execute_model_training()
            
            # Step 3: Model Evaluation
            if self.config.get('steps', {}).get('model_evaluation', {}).get('enabled', True):
                self._execute_model_evaluation()
            
            # Step 4: Data Export
            if self.config.get('steps', {}).get('data_export', {}).get('enabled', True):
                self._execute_data_export()
            
            self.results['pipeline']['end_time'] = datetime.now().isoformat()
            self.results['pipeline']['status'] = 'SUCCESS'
            
            logger.info("\n" + "="*80)
            logger.info("✓ PIPELINE COMPLETE")
            logger.info("="*80)
            
            return self.results
            
        except Exception as e:
            self.results['pipeline']['status'] = 'FAILED'
            self.results['pipeline']['error'] = str(e)
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _execute_data_import(self):
        """Execute data import step"""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA IMPORT")
        logger.info("="*80)
        
        step_config = self.config.get('steps', {}).get('data_import', {}).get('config', {})
        
        # Merge with global data config
        if 'data' in self.config:
            step_config.update(self.config['data'])
        
        step = DataImportStep(step_config)
        self.df_raw = step.execute()
        
        # Store metadata
        self.results['data_import'] = {
            'status': 'SUCCESS',
            **step.get_metadata()
        }
    
    def _execute_model_training(self):
        """Execute model training step"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: MODEL TRAINING (Preprocessing + Model Building)")
        logger.info("="*80)
        
        step_config = self.config.get('steps', {}).get('model_training', {}).get('config', {})
        
        # Merge with global model config
        if 'model' in self.config:
            step_config.update(self.config['model'])
        
        step = ModelTrainingStep(step_config)
        self.model, self.df_processed, self.features = step.execute(self.df_raw)
        
        self.results['model_training'] = {
            'status': 'SUCCESS',
            'model_type': step_config.get('type', 'ridge'),
            'model_alpha': step_config.get('alpha', 1.0)
        }
    
    def _execute_model_evaluation(self):
        """Execute model evaluation step"""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: MODEL EVALUATION")
        logger.info("="*80)
        
        step = ModelEvaluationStep()
        eval_results = step.execute(self.model, self.df_processed, self.features)
        
        self.results['model_evaluation'] = {
            'status': 'SUCCESS',
            **eval_results
        }
    
    def _execute_data_export(self):
        """Execute data export step"""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: DATA EXPORT")
        logger.info("="*80)
        
        step_config = self.config.get('steps', {}).get('data_export', {}).get('config', {})
        
        # Merge with global output config
        if 'output' in self.config:
            step_config.update(self.config['output'])
        
        # Add output prefix if not set
        if 'output_prefix' not in step_config:
            step_config['output_prefix'] = f"mmm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        step = DataExportStep(step_config)
        output_files = step.execute(self.df_processed, self.model, self.results)
        
        self.results['data_export'] = {
            'status': 'SUCCESS',
            'output_files': output_files
        }
    
    def get_summary(self) -> str:
        """Generate summary report"""
        summary = "\n" + "="*80 + "\n"
        summary += "MMM PIPELINE EXECUTION SUMMARY\n"
        summary += "="*80 + "\n\n"
        
        summary += f"Status: {self.results['pipeline'].get('status', 'UNKNOWN')}\n"
        summary += f"Start: {self.results['pipeline'].get('start_time')}\n"
        summary += f"End: {self.results['pipeline'].get('end_time', 'N/A')}\n\n"
        
        # Data import summary
        if 'data_import' in self.results:
            data = self.results['data_import']
            summary += f"DATA:\n"
            summary += f"  Shape: {data['shape'][0]:,} × {data['shape'][1]}\n"
            if 'date_range' in data:
                summary += f"  Range: {data['date_range']['start']} to {data['date_range']['end']}\n"
            summary += "\n"
        
        # Model training summary
        if 'model_training' in self.results:
            model = self.results['model_training']
            summary += f"MODEL:\n"
            summary += f"  Type: {model.get('model_type', 'unknown').upper()}\n"
            summary += f"  Alpha: {model.get('model_alpha', 'N/A')}\n\n"
        
        # Model evaluation summary
        if 'model_evaluation' in self.results:
            eval_data = self.results['model_evaluation']
            
            if 'model_metrics' in eval_data:
                metrics = eval_data['model_metrics']
                summary += f"PERFORMANCE:\n"
                summary += f"  R² Score: {metrics['r2_score']:.4f}\n"
                summary += f"  RMSE: ${metrics['rmse']:,.0f}\n"
                summary += f"  MAE: ${metrics['mae']:,.0f}\n\n"
            
            if 'elasticity' in eval_data:
                summary += f"ELASTICITY (top 5):\n"
                elasticity = eval_data['elasticity']
                for i, (channel, value) in enumerate(sorted(elasticity.items(), key=lambda x: abs(x[1]), reverse=True)[:5]):
                    summary += f"  {channel}: {value:.4f}\n"
                summary += "\n"
            
            if 'contribution' in eval_data:
                summary += f"CONTRIBUTION (top 5):\n"
                contribution = eval_data['contribution']
                for i, (channel, value) in enumerate(sorted(contribution.items(), key=lambda x: x[1], reverse=True)[:5]):
                    summary += f"  {channel}: {value:.2f}%\n"
                summary += "\n"
        
        summary += "="*80 + "\n"
        return summary


def main(config_path: str = 'config/pipeline_config.yaml'):
    """Run pipeline with configuration file"""
    orchestrator = PipelineOrchestrator(config_path)
    results = orchestrator.run()
    
    # Print summary
    print(orchestrator.get_summary())
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MMM Pipeline with Config')
    parser.add_argument('--config', default='config/pipeline_config.yaml',
                       help='Path to pipeline config YAML (default: config/pipeline_config.yaml)')
    
    args = parser.parse_args()
    results = main(args.config)
