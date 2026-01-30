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
import re

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
            'create_lags': True,  # Create lag variables for media spend
            'lag_periods': [1, 2],  # Create lag 1 and lag 2
            'create_fixed_effects': True,  # Create geo & week fixed effects
            'create_seasonality': True,  # Create seasonal controls
        }
    
    def run(self):
        """Execute full preprocessing pipeline"""
        logger.info("Starting data preprocessing...")
        
        self._handle_missing_values()
        self._remove_unusable_channels()
        self._create_channel_lists()
        self._engineer_features()
        self._handle_promotional_data()
        self._create_lagged_features()
        self._create_fixed_effects()
        self._create_seasonality_features()
        self._isolate_organic_control()
        
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
        """Process promotional information and extract promo features"""
        logger.info("Processing promotional data...")
        
        if 'promo_description' not in self.df.columns:
            logger.info("  ⚠ No promotional data found")
            return
        
        # 1. Binary indicator: any promotion present
        self.df['has_promotion'] = (~self.df['promo_description'].isna()).astype(int)
        
        # 2. Extract promotional feature types using regex patterns
        promo_text = self.df['promo_description'].fillna('').str.lower()
        
        # Define promotional patterns
        promo_patterns = {
            'free_shipping_promo': [r'darmowa dostawa', r'bezpłatna wysyłka', r'free shipping', r'za darmo\s+wysy'],
            'discount_percent_promo': [r'rabat', r'-\d+%', r'discount', r'taniej', r'tańsza'],
            'free_item_promo': [r'gratis', r'bezpłatnie', r'free item', r'za darmo\s+produkt'],
            'bundle_multibuy_promo': [r'druga sztuka', r'kupujesz.*drugi', r'2[+\w]*\s+za', r'bundle', r'komplet'],
            'min_purchase_promo': [r'zamówień powyżej', r'minimum', r'od\s+\d+', r'powyżej\s+\d+', r'za\s+\d+\s+zł'],
        }
        
        # Create binary features for each promo type
        promo_feature_cols = []
        for promo_type, patterns in promo_patterns.items():
            pattern_str = '|'.join(patterns)
            self.df[promo_type] = promo_text.str.contains(pattern_str, regex=True, case=False).astype(int)
            promo_feature_cols.append(promo_type)
            count = self.df[promo_type].sum()
            if count > 0:
                logger.info(f"  • {promo_type}: {count:,} records ({100*count/len(self.df):.1f}%)")
        
        # 3. Extract discount percentage if mentioned
        self.df['discount_percent'] = 0
        discount_pattern = r'-(\d+)%'
        for idx, promo in self.df['promo_description'].items():
            if pd.notna(promo):
                match = re.search(discount_pattern, promo)
                if match:
                    self.df.loc[idx, 'discount_percent'] = int(match.group(1))
        
        if self.df['discount_percent'].max() > 0:
            logger.info(f"  • discount_percent: {(self.df['discount_percent'] > 0).sum():,} records with extracted %")
            promo_feature_cols.append('discount_percent')
        
        # 4. Promotion intensity: count active promos per week-geo
        self.df['promo_intensity'] = self.df.groupby(['week', 'geo'])['promo_description'].transform(
            lambda x: (~x.isna()).sum()
        )
        logger.info(f"  • promo_intensity: promotional count per week-geo")
        promo_feature_cols.append('promo_intensity')
        
        # Store promotional feature columns
        self.promo_feature_cols = promo_feature_cols
        
        logger.info(f"  • Created {len(promo_feature_cols)} promotional features")
        self.preprocessing_log.append(f"Created {len(promo_feature_cols)} promotional features")
        self.preprocessing_log.append(f"  - has_promotion (binary), free_shipping, discount_%, free_item, bundle, min_purchase")
        self.preprocessing_log.append(f"  - discount_percent (extracted %), promo_intensity (count per week-geo)")
    
    def _create_lagged_features(self):
        """Create lagged variables for media spend (capture carryover effects)"""
        if not self.config['create_lags']:
            return
        
        logger.info("Creating lagged media spend variables...")
        lag_periods = self.config['lag_periods']
        
        # Ensure data is sorted by week for proper lags
        if 'week' in self.df.columns:
            self.df['week'] = pd.to_datetime(self.df['week'])
            self.df = self.df.sort_values('week').reset_index(drop=True)
        
        lag_cols = []
        for cost_col in self.cost_cols:
            for lag in lag_periods:
                lag_col = f'{cost_col}_lag{lag}'
                # Group by geo to avoid spillover between geos
                self.df[lag_col] = self.df.groupby('geo')[cost_col].shift(lag)
                lag_cols.append(lag_col)
        
        logger.info(f"  • Created {len(lag_cols)} lagged spend variables (lag {lag_periods})")
        self.preprocessing_log.append(f"Created lagged spend variables (lag {lag_periods})")
        
        # Fill missing lags with 0 (start of series)
        for lag_col in lag_cols:
            self.df[lag_col].fillna(0, inplace=True)
    
    def _create_fixed_effects(self):
        """Create fixed effects dummies for geo and week"""
        if not self.config['create_fixed_effects']:
            return
        
        logger.info("Creating fixed effects...")
        
        # Geo fixed effects (one-hot encode, drop first for multicollinearity)
        if 'geo' in self.df.columns:
            geo_dummies = pd.get_dummies(self.df['geo'], prefix='geo_fe', drop_first=True)
            n_geo = geo_dummies.shape[1]
            self.df = pd.concat([self.df, geo_dummies], axis=1)
            logger.info(f"  • Created {n_geo} geo fixed effects dummies")
            self.preprocessing_log.append(f"Created {n_geo} geo fixed effects")
        
        # Week fixed effects (one-hot encode, drop first)
        if 'week' in self.df.columns:
            week_dummies = pd.get_dummies(self.df['week'], prefix='week_fe', drop_first=True)
            n_week = week_dummies.shape[1]
            self.df = pd.concat([self.df, week_dummies], axis=1)
            logger.info(f"  • Created {n_week} week fixed effects dummies")
            self.preprocessing_log.append(f"Created {n_week} week fixed effects")
    
    def _create_seasonality_features(self):
        """Create seasonal control variables (month, quarter, day of week)"""
        if not self.config['create_seasonality']:
            return
        
        logger.info("Creating seasonality features...")
        
        if 'week' not in self.df.columns:
            return
        
        # Ensure week is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['week']):
            self.df['week'] = pd.to_datetime(self.df['week'])
        
        # Month of year (12 months, drop first)
        month_dummies = pd.get_dummies(self.df['week'].dt.month, prefix='month', drop_first=True)
        self.df = pd.concat([self.df, month_dummies], axis=1)
        
        # Quarter (4 quarters, drop first)
        quarter_dummies = pd.get_dummies(self.df['week'].dt.quarter, prefix='quarter', drop_first=True)
        self.df = pd.concat([self.df, quarter_dummies], axis=1)
        
        # Week of year (52 weeks, drop first to avoid collinearity with month)
        week_of_year_dummies = pd.get_dummies(self.df['week'].dt.isocalendar().week, prefix='week_of_year', drop_first=True)
        self.df = pd.concat([self.df, week_of_year_dummies], axis=1)
        
        logger.info(f"  • Created month, quarter, and week-of-year seasonality controls")
        self.preprocessing_log.append("Created seasonality features (month, quarter, week_of_year)")
    
    def _isolate_organic_control(self):
        """Isolate organic traffic as dedicated control variable"""
        logger.info("Processing organic traffic control...")
        
        if 'impression_organic' in self.df.columns:
            # Create log-transformed organic traffic
            self.df['log_organic'] = np.log(self.df['impression_organic'] + 1)
            logger.info(f"  • Created log_organic control variable")
            self.preprocessing_log.append("Created log_organic as control variable")
            
            # Optional: create lagged organic for promo halo effects
            if self.config['create_lags'] and 'geo' in self.df.columns:
                self.df['log_organic_lag1'] = self.df.groupby('geo')['log_organic'].shift(1).fillna(0)
                logger.info(f"  • Created log_organic_lag1 for halo effects")
                self.preprocessing_log.append("Created lagged organic traffic (halo effects)")
    
    def get_preprocessed_data(self):
        """Return preprocessed dataframe"""
        return self.df
    
    def get_feature_sets(self):
        """Return organized feature sets for modeling"""
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Promotional features created during preprocessing
        promo_features = getattr(self, 'promo_feature_cols', ['has_promotion'])
        
        # Lagged features
        lagged_features = [c for c in self.df.columns if '_lag' in c]
        
        # Fixed effects
        geo_fe = [c for c in self.df.columns if c.startswith('geo_fe')]
        week_fe = [c for c in self.df.columns if c.startswith('week_fe')]
        
        # Seasonal features
        seasonal_features = [c for c in self.df.columns if any(x in c for x in ['month_', 'quarter_', 'week_of_year_'])]
        
        # Organic control
        organic_features = [c for c in self.df.columns if 'organic' in c and c.startswith('log_')]
        
        return {
            'target': ['revenue'],
            'spend_features': self.cost_cols,
            'spend_features_log': [f'{c}_log' for c in self.cost_cols if f'{c}_log' in self.df.columns],
            'lagged_spend_features': lagged_features,
            'impression_features': self.impression_cols,
            'control_vars': ['population', 'market_share'] + organic_features,
            'promotional_features': promo_features,
            'geo_fixed_effects': geo_fe,
            'week_fixed_effects': week_fe,
            'seasonal_features': seasonal_features,
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
