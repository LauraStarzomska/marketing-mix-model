"""Model Evaluation Step - Calculate metrics, elasticity, and contribution"""

import pandas as pd
import logging
from typing import Dict

from src.mmm_model import MMModel

logger = logging.getLogger(__name__)


class ModelEvaluationStep:
    """Evaluate model performance and calculate elasticity"""
    
    def __init__(self):
        """Initialize model evaluation step"""
        self.results = {}
    
    def execute(self, 
                model: MMModel, 
                df_processed: pd.DataFrame, 
                features: Dict) -> Dict:
        """
        Execute model evaluation
        
        Args:
            model: Trained MMModel
            df_processed: Processed data
            features: Feature sets dictionary
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("[4/5] Calculating elasticity and contribution...")
        
        # Prepare data
        X = df_processed[features['spend_features']]
        y = df_processed[features['target'][0]]
        
        # Get metrics
        metrics = model.evaluate(X, y)
        self._store_metrics(metrics)
        
        # Calculate elasticity
        elasticity = self._calculate_elasticity(model, X, y, features['spend_features'])
        self._store_elasticity(elasticity)
        
        # Calculate contribution
        contribution = self._calculate_contribution(model, X, y)
        self._store_contribution(contribution)
        
        logger.info("✓ Evaluation complete")
        return self.results
    
    def _store_metrics(self, metrics: Dict):
        """Store model metrics"""
        logger.info(f"✓ Model R² Score: {metrics['R²']:.4f}")
        logger.info(f"✓ RMSE: ${metrics['RMSE']:,.0f}")
        logger.info(f"✓ MAE: ${metrics['MAE']:,.0f}")
        
        self.results['model_metrics'] = {
            'r2_score': float(metrics['R²']),
            'rmse': float(metrics['RMSE']),
            'mae': float(metrics['MAE']),
            'mse': float(metrics['MSE'])
        }
    
    def _calculate_elasticity(self, model: MMModel, X: pd.DataFrame, 
                             y: pd.Series, features: list) -> Dict:
        """Calculate elasticity for each channel"""
        elasticity = {}
        logger.info("\nElasticity by Channel:")
        for channel in features:
            elasticity[channel] = model.calculate_elasticity(X, y, channel)
            logger.info(f"  {channel}: {elasticity[channel]:.4f}")
        return elasticity
    
    def _store_elasticity(self, elasticity: Dict):
        """Store elasticity results"""
        self.results['elasticity'] = {
            channel: float(value) for channel, value in elasticity.items()
        }
    
    def _calculate_contribution(self, model: MMModel, X: pd.DataFrame, 
                               y: pd.Series) -> pd.Series:
        """Calculate revenue contribution by channel"""
        contribution = model.calculate_contribution(X, y)
        logger.info("\nRevenue Contribution by Channel (%):")
        for channel, contrib_value in contribution.items():
            logger.info(f"  {channel}: {contrib_value:.2f}%")
        return contribution
    
    def _store_contribution(self, contribution: pd.Series):
        """Store contribution results"""
        self.results['contribution'] = {
            channel: float(value) for channel, value in contribution.items()
        }
