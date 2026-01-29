"""
Data Preprocessing Module for Marketing Mix Model

Handles:
- Missing value imputation
- Feature engineering
- Outlier handling
- Scaling and transformations
- Channel selection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class MarketingDataPreprocessor:
    """Preprocessing pipeline for marketing mix model data"""
    
    def __init__(self, df, config=None):
        """
        Initialize preprocessor
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw marketing data
        config : dict, optional
            Configuration parameters
        """
        self.df = df.copy()
        self.config = config or self._default_config()
        self.scaler = StandardScaler()
        self.preprocessing_log = []
        
    @staticmethod
    def _default_config():
        """Default preprocessing configuration"""
        return {
            'fill_tv_impressions': True,
            'remove_outdoor2': True,
            'log_transform': True,
            'handle_promotions': True,
            'fillna_method': 'median',  # 'median', 'ffill', 'bfill'
        }
    
    def run(self):
        """Execute full preprocessing pipeline"""
        logger.info("Starting data preprocessing...")
        
        self._handle_missing_values()
        self._remove_unusable_channels()
        self._create_channel_lists()
        self._engineer_features()
        self._handle_promotional_data()
        
        logger.info("✓ Preprocessing complete")
        return self
    
    def _handle_missing_values(self):
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        # TV Impressions: 5.3% missing
        if self.config['fill_tv_impressions'] and 'impression_tv' in self.df.columns:
            missing_before = self.df['impression_tv'].isna().sum()
            
            if self.config['fillna_method'] == 'median':
                self.df['impression_tv'].fillna(
                    self.df['impression_tv'].median(), 
                    inplace=True
                )
            elif self.config['fillna_method'] == 'ffill':
                # Forward fill for time series
                self.df['impression_tv'].fillna(method='ffill', inplace=True)
                self.df['impression_tv'].fillna(method='bfill', inplace=True)
            
            missing_after = self.df['impression_tv'].isna().sum()
            logger.info(f"  • TV Impressions: {missing_before} → {missing_after} missing")
            self.preprocessing_log.append(f"Filled TV impressions using {self.config['fillna_method']}")
        
        # Promotional data: 46.8% missing is expected (not all weeks have promos)
        # Keep as-is for now - will create indicator variable
        logger.info(f"  • Promotional Data: {self.df['promo_description'].isna().sum()} missing (expected)")
    
    def _remove_unusable_channels(self):
        """Remove channels with insufficient data"""
        logger.info("Removing unusable channels...")
        
        if self.config['remove_outdoor2']:
            # Outdoor2: 100% missing impressions
            cols_to_drop = [c for c in self.df.columns if 'outdoor2' in c.lower()]
            if cols_to_drop:
                self.df = self.df.drop(columns=cols_to_drop)
                logger.info(f"  • Removed Outdoor2 channel ({len(cols_to_drop)} columns)")
                self.preprocessing_log.append("Removed Outdoor2 channel (100% missing)")
    
    def _create_channel_lists(self):
        """Identify and store channel columns"""
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        self.cost_cols = [c for c in numeric_cols if 'cost' in c.lower()]
        self.impression_cols = [c for c in numeric_cols if 'impression' in c.lower()]
        self.channels = [c.replace('cost_', '') for c in self.cost_cols]
        
        logger.info(f"  • Identified {len(self.channels)} channels: {', '.join(self.channels) if self.channels else 'None'}")
        self.preprocessing_log.append(f"Identified {len(self.channels)} marketing channels")
    
    def _engineer_features(self):
        """Create derived features"""
        logger.info("Engineering features...")
        
        # CPM: Cost Per Mille (per 1000 impressions)
        for channel in self.channels:
            cost_col = f'cost_{channel}'
            impression_col = f'impression_{channel}'
            
            if cost_col in self.df.columns and impression_col in self.df.columns:
                # Avoid division by zero
                self.df[f'cpm_{channel}'] = (
                    self.df[cost_col] / (self.df[impression_col] / 1000 + 1)
                )
        
        logger.info(f"  • Created {len(self.channels)} CPM features")
        self.preprocessing_log.append(f"Created CPM (Cost Per Mille) features")
        
        # Log transformations for elasticity interpretation
        if self.config['log_transform']:
            for col in self.cost_cols + self.impression_cols:
                if col in self.df.columns:
                    # Add small constant to avoid log(0)
                    self.df[f'{col}_log'] = np.log(self.df[col] + 1)
            
            logger.info(f"  • Created log-transformed features")
            self.preprocessing_log.append("Applied log transformations for elasticity")
    
    def _handle_promotional_data(self):
        """Process promotional information"""
        logger.info("Processing promotional data...")
        
        if 'promo_description' in self.df.columns:
            # Create binary promotion indicator
            self.df['has_promotion'] = (~self.df['promo_description'].isna()).astype(int)
            
            # Count of unique promotions per week-geo
            promo_counts = self.df.groupby(['week', 'geo'])['promo_description'].nunique()
            
            logger.info(f"  • Created promotion indicator (has_promotion)")
            logger.info(f"  • {self.df['has_promotion'].sum()} records with promotions")
            self.preprocessing_log.append("Created promotional indicator variable")
    
    def get_preprocessed_data(self):
        """Return preprocessed dataframe"""
        return self.df
    
    def get_feature_sets(self):
        """Return organized feature sets for modeling"""
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        return {
            'target': ['revenue'],
            'spend_features': self.cost_cols,
            'impression_features': self.impression_cols,
            'control_vars': ['population', 'market_share'],
            'interaction_features': [c for c in self.df.columns if '_log' in c],
            'promotional_features': ['has_promotion'],
            'all_features': [c for c in numeric_cols if c not in ['revenue'] and 'promo_description' not in c]
        }
    
    def get_summary(self):
        """Return preprocessing summary"""
        return {
            'shape_before': f"{self.df.shape}",
            'channels': self.channels,
            'num_channels': len(self.channels),
            'cost_columns': len(self.cost_cols),
            'impression_columns': len(self.impression_cols),
            'missing_values': self.df.isnull().sum().sum(),
            'completeness_%': round((1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100, 1),
            'preprocessing_steps': self.preprocessing_log
        }
    
    def print_summary(self):
        """Print preprocessing summary to console"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("PREPROCESSING SUMMARY")
        print("="*80)
        
        print(f"\nData Shape: {summary['shape_before']}")
        print(f"Data Completeness: {summary['completeness_%']}%")
        
        print(f"\nChannels ({summary['num_channels']}): {', '.join(summary['channels'])}")
        print(f"  • Cost columns: {summary['cost_columns']}")
        print(f"  • Impression columns: {summary['impression_columns']}")
        
        print(f"\nPreprocessing Steps:")
        for i, step in enumerate(summary['preprocessing_steps'], 1):
            print(f"  {i}. {step}")
        
        print("\n✓ Data ready for modeling")
        print("="*80 + "\n")


def preprocess_marketing_data(df, config=None):
    """
    Convenience function to preprocess marketing data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw marketing data
    config : dict, optional
        Configuration parameters
        
    Returns:
    --------
    df_processed : pd.DataFrame
        Preprocessed data
    feature_sets : dict
        Organized feature sets for modeling
    """
    preprocessor = MarketingDataPreprocessor(df, config)
    preprocessor.run()
    preprocessor.print_summary()
    
    return preprocessor.get_preprocessed_data(), preprocessor.get_feature_sets()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    from data_loader import load_from_file
    
    df = load_from_file('../data/raw/marketing_data.csv')
    
    # Preprocess
    df_processed, features = preprocess_marketing_data(df)
    
    # Save processed data
    df_processed.to_csv('../data/processed/marketing_data_processed.csv', index=False)
    df_processed.to_parquet('../data/processed/marketing_data_processed.parquet', index=False)
    
    print("✓ Processed data saved to data/processed/")
