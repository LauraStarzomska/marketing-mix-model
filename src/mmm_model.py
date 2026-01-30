"""
Marketing Mix Model (MMM) - Core Modeling Module
Handles regression models and analysis for MMM
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import logging

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.stats as stats

logger = logging.getLogger(__name__)


class AdstockTransformer:
    """Apply adstock transformation to model delayed advertising effects"""

    @staticmethod
    def geometric_adstock(spend_series: pd.Series, decay_rate: float = 0.5) -> pd.Series:
        """
        Apply geometric adstock to model carryover effects.

        Adstock formula: adstocked[t] = spend[t] + decay * spend[t-1] + decay^2 * spend[t-2] + ...

        Args:
            spend_series: Original spend data (should be sorted by time)
            decay_rate: Decay rate (0-1). Higher = longer lasting effect

        Returns:
            Adstocked spend series
        """
        if not 0 <= decay_rate <= 1:
            raise ValueError("decay_rate must be between 0 and 1")

        adstocked = np.zeros(len(spend_series))

        for i in range(len(spend_series)):
            for lag in range(i + 1):
                adstocked[i] += decay_rate ** lag * spend_series.iloc[i - lag]

        return pd.Series(adstocked, index=spend_series.index)

    @staticmethod
    def apply_adstock_to_channels(
        df: pd.DataFrame, spend_columns: List[str], decay_rates: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Apply adstock transformation to all spending channels.

        Args:
            df: DataFrame with spend columns
            spend_columns: List of spending column names
            decay_rates: Dict mapping column names to decay rates (default 0.5 for all)

        Returns:
            DataFrame with adstocked columns
        """
        df_adstocked = df.copy()

        if decay_rates is None:
            decay_rates = {col: 0.5 for col in spend_columns}

        for col in spend_columns:
            if col in df.columns:
                decay = decay_rates.get(col, 0.5)
                df_adstocked[f"{col}_adstocked"] = AdstockTransformer.geometric_adstock(
                    df[col], decay_rate=decay
                )

        return df_adstocked


class MMModel:
    """Marketing Mix Model using regression"""

    def __init__(self, model_type: str = "ols", **kwargs):
        """
        Initialize MMM model.

        Args:
            model_type: 'ols' (OLS), 'ridge' (Ridge), or 'lasso' (Lasso)
            **kwargs: Additional arguments for model (e.g., alpha for Ridge/Lasso)
        """
        self.model_type = model_type
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        if model_type == "ols":
            self.model = LinearRegression()
        elif model_type == "ridge":
            alpha = kwargs.get("alpha", 1.0)
            self.model = Ridge(alpha=alpha)
        elif model_type == "lasso":
            alpha = kwargs.get("alpha", 0.01)
            self.model = Lasso(alpha=alpha, max_iter=10000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.is_fitted = False
        self.X_features = None
        self.y_name = None
        self.metrics = {}

        logger.info(f"Initialized {model_type.upper()} model")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        standardize: bool = True,
    ) -> "MMModel":
        """
        Fit the MMM model.

        Args:
            X: Feature matrix
            y: Target variable (sales)
            standardize: Whether to standardize features

        Returns:
            self
        """
        self.X_features = X.columns.tolist()
        self.y_name = y.name if hasattr(y, "name") else "target"

        # Prepare data
        X_train = X.values
        y_train = y.values.reshape(-1, 1)

        # Standardize if requested
        if standardize:
            X_train = self.scaler_X.fit_transform(X_train)
            y_train = self.scaler_y.fit_transform(y_train).ravel()
        else:
            y_train = y_train.ravel()

        # Fit model
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        logger.info(f"Model fitted with {X.shape[0]} observations and {X.shape[1]} features")
        return self

    def predict(self, X: pd.DataFrame, standardize: bool = True) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix
            standardize: Whether to use same standardization as training

        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_pred = X.values

        if standardize:
            X_pred = self.scaler_X.transform(X_pred)

        y_pred = self.model.predict(X_pred)

        return y_pred

    def evaluate(self, X: pd.DataFrame, y: pd.Series, standardize: bool = True) -> Dict:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: Target variable
            standardize: Whether to use standardization

        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X, standardize=standardize)

        # Calculate metrics
        mse = mean_squared_error(y.values, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y.values, y_pred)
        r2 = r2_score(y.values, y_pred)

        self.metrics = {
            "R²": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MSE": mse,
        }

        logger.info(f"Model Performance - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        return self.metrics

    def get_coefficients(self, standardized: bool = False) -> pd.Series:
        """
        Get model coefficients with feature names.

        Args:
            standardized: Whether coefficients are in standardized scale

        Returns:
            Series with feature names as index and coefficients as values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        coefs = self.model.coef_
        intercept = self.model.intercept_

        coef_dict = {name: coef for name, coef in zip(self.X_features, coefs)}
        coef_dict["intercept"] = intercept

        return pd.Series(coef_dict)

    def calculate_elasticity(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_name: str,
    ) -> float:
        """
        Calculate elasticity for a feature.

        Elasticity = (% change in output) / (% change in input)
                  = coefficient * (mean_input / mean_output)

        Args:
            X: Feature matrix
            y: Target variable
            feature_name: Name of feature to calculate elasticity for

        Returns:
            Elasticity value
        """
        if feature_name not in X.columns:
            raise ValueError(f"Feature {feature_name} not found in X")

        coefs = self.get_coefficients()
        coef = coefs.get(feature_name, 0)

        mean_feature = X[feature_name].mean()
        mean_target = y.mean()

        if mean_target == 0 or mean_feature == 0:
            return 0

        elasticity = coef * (mean_feature / mean_target)
        return elasticity

    def calculate_contribution(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.Series:
        """
        Calculate contribution of each feature to total sales.

        Contribution = Feature_Effect / Total_Sales

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Series with contribution for each feature
        """
        coefs = self.get_coefficients()
        contributions = {}

        for feature in self.X_features:
            if feature in coefs.index:
                # Calculate feature effect
                feature_effect = coefs[feature] * X[feature].mean()
                contribution = feature_effect / y.mean() if y.mean() != 0 else 0
                contributions[feature] = contribution * 100  # As percentage

        return pd.Series(contributions).sort_values(ascending=False)

    def calculate_marginal_roas(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        spend_features: List[str],
        raw_spend: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculate Marginal Return on Ad Spend (MROAS) for each channel.

        MROAS = β_k * (Total_Sales / Total_Spend_k)
        
        Interpretation: For every additional $1 spent on channel k at current spend levels,
        we expect revenue to increase by MROAS_k dollars.

        Args:
            X: Feature matrix (should contain log-transformed spend)
            y: Target variable (sales/revenue)
            spend_features: List of spend feature names (e.g., ['log_spend_google'])
            raw_spend: Optional DataFrame with raw (non-log) spend values for scale calculation

        Returns:
            DataFrame with MROAS for each channel
        """
        coefs = self.get_coefficients()
        total_sales = y.sum()
        
        mroas_results = {}
        
        for spend_feat in spend_features:
            if spend_feat not in coefs.index:
                logger.warning(f"Feature {spend_feat} not in model coefficients")
                continue
            
            # Get coefficient
            beta_k = coefs[spend_feat]
            
            # Determine spend column name
            # Try to find corresponding raw spend column
            spend_col = None
            if raw_spend is not None:
                # Try exact match or remove '_log' suffix
                if spend_feat in raw_spend.columns:
                    spend_col = spend_feat
                elif spend_feat.endswith('_log'):
                    base_name = spend_feat.replace('_log', '')
                    if base_name in raw_spend.columns:
                        spend_col = base_name
            
            if spend_col and raw_spend is not None:
                total_spend = raw_spend[spend_col].sum()
            else:
                logger.warning(f"Could not find raw spend for {spend_feat}, using mean approximation")
                total_spend = np.exp(X[spend_feat].mean()) - 1 if spend_feat.endswith('_log') else X[spend_feat].mean()
            
            # Avoid division by zero
            if total_spend > 0:
                mroas = beta_k * (total_sales / total_spend)
                mroas_results[spend_feat] = {
                    'coefficient': beta_k,
                    'total_sales': total_sales,
                    'total_spend': total_spend,
                    'mroas': mroas,
                    'mroas_interpretation': f'${mroas:.2f} revenue per $1 spend'
                }
        
        mroas_df = pd.DataFrame(mroas_results).T
        mroas_df = mroas_df.sort_values('mroas', ascending=False)
        
        logger.info(f"\nMarginal ROAS by Channel:")
        for idx, row in mroas_df.iterrows():
            logger.info(f"  {idx}: {row['mroas_interpretation']}")
        
        return mroas_df

    def summary(self) -> str:
        """Get model summary"""
        if not self.is_fitted:
            return "Model not fitted yet"

        summary = f"\n{'='*60}\n"
        summary += f"Marketing Mix Model Summary\n"
        summary += f"{'='*60}\n"
        summary += f"Model Type: {self.model_type.upper()}\n"
        summary += f"Features: {len(self.X_features)}\n"
        summary += f"Target: {self.y_name}\n"
        summary += f"\nModel Coefficients:\n"

        coefs = self.get_coefficients()
        for name, coef in coefs.items():
            summary += f"  {name:20s}: {coef:10.6f}\n"

        summary += f"\nPerformance Metrics:\n"
        for metric, value in self.metrics.items():
            summary += f"  {metric:10s}: {value:10.4f}\n"

        summary += f"{'='*60}\n"
        return summary


def build_mmm_dataset(
    df: pd.DataFrame,
    sales_column: str,
    spend_columns: List[str],
    additional_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for MMM modeling.

    Args:
        df: Raw data
        sales_column: Name of sales target column
        spend_columns: List of spending columns to include
        additional_features: Additional feature columns (organic, seasonality, etc.)

    Returns:
        X (features), y (target)
    """
    feature_cols = spend_columns.copy()

    if additional_features:
        feature_cols.extend(additional_features)

    # Validate columns exist
    for col in feature_cols + [sales_column]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    X = df[feature_cols].copy()
    y = df[sales_column].copy()

    # Remove rows with NaN
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    logger.info(f"Created MMM dataset: {X.shape[0]} observations, {X.shape[1]} features")

    return X, y
