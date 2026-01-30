"""Model Training Step"""

import pandas as pd
import logging
from typing import Dict, Tuple

from src.preprocessing import MarketingDataPreprocessor
from src.mmm_model import MMModel

logger = logging.getLogger(__name__)


class ModelTrainingStep:
    """Preprocess data and train MMM model"""
    
    def __init__(self, config: Dict):
        """
        Initialize model training step
        
        Args:
            config: Configuration dict with:
                - model_type: 'ols', 'ridge', or 'lasso'
                - model_alpha: Regularization parameter
        """
        self.config = config
        self.df_raw = None
        self.df_processed = None
        self.model = None
        self.features = None
    
    def execute(self, df_raw: pd.DataFrame) -> Tuple[MMModel, pd.DataFrame, Dict]:
        """
        Execute preprocessing and model training
        
        Args:
            df_raw: Raw data from import step
            
        Returns:
            Tuple of (trained_model, processed_data, features)
        """
        self.df_raw = df_raw
        
        # Step 2: Preprocess
        logger.info("[2/5] Preprocessing data...")
        self._preprocess_data()
        
        # Step 3: Build model
        logger.info("[3/5] Building MMM model...")
        self._build_model()
        
        logger.info("✓ Model training complete")
        return self.model, self.df_processed, self.features
    
    def _preprocess_data(self):
        """Preprocess data using MarketingDataPreprocessor"""
        preprocessor = MarketingDataPreprocessor(self.df_raw)
        preprocessor.run()
        
        self.df_processed = preprocessor.get_preprocessed_data()
        self.features = preprocessor.get_feature_sets()
        
        summary = preprocessor.get_summary()
        logger.info(f"✓ Data shape: {summary['shape_before']}")
        logger.info(f"✓ Channels: {summary['num_channels']} - {', '.join(summary['channels'])}")
    
    def _build_model(self):
        """Build MMM model"""
        # Prepare features - keep as DataFrame
        X = self.df_processed[self.features['spend_features']]
        y = self.df_processed[self.features['target'][0]]  # Keep as Series
        
        logger.info(f"Features: {X.shape}")
        logger.info(f"Target: {y.shape}")
        
        # Build and train model
        model_type = self.config.get('type', 'ridge')
        model_alpha = self.config.get('alpha', 1.0)
        
        self.model = MMModel(
            model_type=model_type,
            alpha=model_alpha
        )
        self.model.fit(X, y)
        
        logger.info(f"✓ Model type: {model_type.upper()}")
