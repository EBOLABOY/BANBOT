"""
Feature Selector Module

Provides feature selection capabilities based on model importance and SHAP values,
helping identify the most important features, reduce overfitting and improve model performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import logging
from typing import List, Dict, Optional, Union, Any, Tuple
from datetime import datetime
import shap
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FeatureSelector')


class FeatureSelector:
    """
    Feature selector class for selecting important features from a pre-trained model
    """
    
    def __init__(
        self,
        model_path: str,
        method: str = 'importance',
        n_features: int = 20,
        output_dir: str = 'data/results/feature_selection',
        verbose: bool = False
    ):
        """
        Initialize feature selector
        
        Args:
            model_path: Path to pre-trained model
            method: Feature selection method, 'importance' or 'shap'
            n_features: Number of features to select
            output_dir: Output directory
            verbose: Whether to print detailed logs
        """
        self.model_path = model_path
        self.method = method
        self.n_features = n_features
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        
        # Detect model type
        self.model_type = self._detect_model_type()
        
        # Store selected features
        self.selected_features = None
        self.feature_importances = None
    
    def _load_model(self) -> Any:
        """
        Load pre-trained model
        
        Returns:
            Trained model
        """
        logger.info(f"Loading model from {self.model_path}")
        return joblib.load(self.model_path)
    
    def _detect_model_type(self) -> str:
        """
        Detect model type
        
        Returns:
            str: Model type, e.g. 'xgboost', 'lightgbm', 'sklearn'
        """
        model_str = str(type(self.model))
        if 'xgboost' in model_str.lower():
            return 'xgboost'
        elif 'lightgbm' in model_str.lower():
            return 'lightgbm'
        elif 'sklearn' in model_str.lower() or 'randomforest' in model_str.lower():
            return 'sklearn'
        else:
            return 'unknown'
    
    def select_features(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> List[str]:
        """
        Select features
        
        Args:
            X: Feature DataFrame
            y: Target variable (for SHAP value calculation)
            
        Returns:
            List[str]: List of selected features
        """
        if self.method == 'importance':
            selected_features = self._select_by_importance(X)
        elif self.method == 'shap':
            if y is None:
                logger.warning("Target variable y required for SHAP method, falling back to importance method")
                selected_features = self._select_by_importance(X)
            else:
                selected_features = self._select_by_shap(X, y)
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")
        
        self.selected_features = selected_features
        return selected_features
    
    def _select_by_importance(self, X: pd.DataFrame) -> List[str]:
        """
        Select features based on model feature importance
        
        Args:
            X: Feature DataFrame
            
        Returns:
            List[str]: List of selected features
        """
        logger.info("Selecting features based on model importance")
        
        # Check if model has feature_importances_
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        # Get feature importance
        importances = self.model.feature_importances_
        
        # Create feature importance DataFrame
        feature_names = X.columns.tolist()
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Save feature importance result
        self.feature_importances = importance_df
        
        # Select top N features
        selected_features = importance_df.head(self.n_features)['feature'].tolist()
        
        logger.info(f"Selected {len(selected_features)} features")
        if self.verbose:
            for i, feat in enumerate(selected_features[:10]):
                importance = importance_df[importance_df['feature'] == feat]['importance'].values[0]
                logger.info(f"  {i+1}. {feat}: {importance:.6f}")
            if len(selected_features) > 10:
                logger.info(f"  ... and {len(selected_features) - 10} more features")
        
        return selected_features
    
    def _select_by_shap(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Select features based on SHAP values
        
        Args:
            X: Feature DataFrame
            y: Target variable
            
        Returns:
            List[str]: List of selected features
        """
        logger.info("Selecting features based on SHAP values")
        
        # Calculate SHAP values
        # For large datasets, use random sampling to reduce computation time
        if len(X) > 1000:
            logger.info(f"Dataset is large, randomly sampling 1000 records for SHAP calculation")
            sample_indices = np.random.choice(len(X), 1000, replace=False)
            X_sample = X.iloc[sample_indices]
        else:
            X_sample = X
        
        # Choose SHAP explainer based on model type
        if self.model_type == 'xgboost':
            explainer = shap.Explainer(self.model)
            shap_values = explainer(X_sample)
            shap_values_mean = np.abs(shap_values.values).mean(0)
        else:
            # For other models, use TreeExplainer
            try:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):  # For multi-output models
                    shap_values_mean = np.abs(shap_values[0]).mean(0)
                else:
                    shap_values_mean = np.abs(shap_values).mean(0)
            except Exception as e:
                logger.warning(f"SHAP value calculation failed: {e}, falling back to importance method")
                return self._select_by_importance(X)
        
        # Create SHAP value DataFrame
        feature_names = X.columns.tolist()
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'importance': shap_values_mean
        })
        
        # Sort by importance
        shap_df = shap_df.sort_values('importance', ascending=False)
        
        # Save feature importance result
        self.feature_importances = shap_df
        
        # Select top N features
        selected_features = shap_df.head(self.n_features)['feature'].tolist()
        
        logger.info(f"Selected {len(selected_features)} features")
        if self.verbose:
            for i, feat in enumerate(selected_features[:10]):
                importance = shap_df[shap_df['feature'] == feat]['importance'].values[0]
                logger.info(f"  {i+1}. {feat}: {importance:.6f}")
            if len(selected_features) > 10:
                logger.info(f"  ... and {len(selected_features) - 10} more features")
        
        return selected_features
    
    def plot_feature_importance(
        self,
        save_path: Optional[str] = None,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot feature importance chart

        Args:
            save_path: Path to save the plot, if None, the plot is not saved
            top_n: Number of top features to display
            figsize: Figure size
        """
        if self.feature_importances is None:
            logger.warning("Feature importance not calculated yet, cannot plot.")
            return

        # Get top N features
        if top_n is not None and top_n < len(self.feature_importances):
            plot_df = self.feature_importances.head(top_n)
        else:
            plot_df = self.feature_importances

        # Create figure
        plt.figure(figsize=figsize)

        # Plot bar chart
        plt.barh(
            plot_df['feature'][::-1],  # Reverse order to show most important at top
            plot_df['importance'][::-1]
        )

        # Add title and labels
        method_name = "Importance" if self.method == 'importance' else "SHAP Value"
        plt.title(f"Top {len(plot_df)} Features (Method: {method_name})")
        plt.xlabel('Importance Score')
        plt.ylabel('Feature Name')
        plt.tight_layout()

        # Save figure
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to: {save_path}")

        plt.close()
    
    def save_selected_features(self, save_path: Optional[str] = None) -> str:
        """
        Save selected features to a JSON file
        
        Args:
            save_path: Path to save, if None, use default path
            
        Returns:
            str: Path to saved file
        """
        if self.selected_features is None:
            logger.warning("No features selected yet, cannot save")
            return ""
        
        # Use default path
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(
                self.output_dir, 
                f"selected_features_{self.method}_{self.n_features}_{timestamp}.json"
            )
        
        # Create save data
        data = {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'selection_method': self.method,
            'n_features': self.n_features,
            'selected_features': self.selected_features,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # If feature importance is available, also save it
        if self.feature_importances is not None:
            # Convert to dictionary format
            importance_dict = {}
            for _, row in self.feature_importances.iterrows():
                importance_dict[row['feature']] = float(row['importance'])
            data['feature_importances'] = importance_dict
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        logger.info(f"Selected features saved to: {save_path}")
        
        return save_path


def retrain_with_selected_features(
    model_path: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: List[str],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    early_stopping_rounds: int = 50,
    output_dir: str = 'data/results/models',
    verbose: bool = False
) -> str:
    """
    使用选择的特征重新训练模型
    
    Args:
        model_path: 原始模型路径
        X_train: 训练特征
        y_train: 训练目标
        selected_features: 选择的特征列表
        X_val: 验证特征(可选)
        y_val: 验证目标(可选)
        early_stopping_rounds: 早停轮数
        output_dir: 输出目录
        verbose: 是否打印详细日志
        
    Returns:
        str: 新模型保存路径
    """
    logger.info(f"使用选择的 {len(selected_features)} 个特征重新训练模型")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载原始模型
    original_model = joblib.load(model_path)
    
    # 提取模型参数(对于XGBoost模型)
    if isinstance(original_model, XGBRegressor):
        # 获取模型参数
        params = original_model.get_params()
        
        # 移除不需要的参数
        remove_keys = ['base_score', 'callbacks', '_estimator_type', 'feature_names_in_', 'feature_types_in_']
        for key in remove_keys:
            if key in params:
                del params[key]
        
        # 创建新模型，将early_stopping_rounds移到这里
        params['early_stopping_rounds'] = early_stopping_rounds # 添加早停参数
        model = XGBRegressor(**params)
        
        # 筛选特征
        X_train_selected = X_train[selected_features]
        X_val_selected = X_val[selected_features] if X_val is not None else None
        
        # 训练模型 - fit方法只接收eval_set
        if X_val_selected is not None and y_val is not None:
            model.fit(
                X_train_selected, y_train,
                eval_set=[(X_val_selected, y_val)],
                verbose=verbose
            )
        else:
            model.fit(X_train_selected, y_train)
    else:
        # 对于其他类型的模型，暂时不支持
        logger.warning(f"不支持的模型类型: {type(original_model)}，无法重新训练")
        return ""
    
    # 评估模型
    if X_val_selected is not None and y_val is not None:
        y_pred = model.predict(X_val_selected)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        logger.info(f"模型评估指标:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  R²: {r2:.6f}")
    
    # 保存新模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = os.path.basename(model_path).split('.')[0]
    save_path = os.path.join(output_dir, f"{model_name}_selected_{len(selected_features)}_{timestamp}.joblib")
    
    joblib.dump(model, save_path)
    logger.info(f"重新训练的模型已保存至: {save_path}")
    
    # 保存所用特征
    features_path = os.path.join(output_dir, f"{model_name}_selected_features_{timestamp}.json")
    with open(features_path, 'w') as f:
        json.dump({
            'model_path': save_path,
            'original_model_path': model_path,
            'selected_features': selected_features,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=4)
    
    logger.info(f"特征列表已保存至: {features_path}")
    
    return save_path


def main():
    """
    主函数，用于命令行调用
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="特征选择工具")
    
    # 基本参数
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--X_file", type=str, required=True, help="特征文件路径")
    parser.add_argument("--y_file", type=str, required=True, help="目标文件路径")
    parser.add_argument("--method", type=str, default="importance", 
                        choices=["importance", "shap"], help="特征选择方法")
    parser.add_argument("--n_features", type=int, default=20, help="选择的特征数量")
    parser.add_argument("--output_dir", type=str, default="data/results/feature_selection", 
                        help="输出目录")
    parser.add_argument("--verbose", action="store_true", help="是否打印详细日志")
    
    # 重新训练参数
    parser.add_argument("--retrain", action="store_true", help="是否重新训练模型")
    parser.add_argument("--X_val_file", type=str, help="验证特征文件路径")
    parser.add_argument("--y_val_file", type=str, help="验证目标文件路径")
    parser.add_argument("--early_stopping_rounds", type=int, default=50, help="早停轮数")
    
    args = parser.parse_args()
    
    # 加载数据
    X = pd.read_csv(args.X_file, index_col=0)
    y = pd.read_csv(args.y_file, index_col=0).iloc[:, 0]
    
    # 加载验证集(如果有)
    X_val, y_val = None, None
    if args.retrain and args.X_val_file and args.y_val_file:
        X_val = pd.read_csv(args.X_val_file, index_col=0)
        y_val = pd.read_csv(args.y_val_file, index_col=0).iloc[:, 0]
    
    # 创建特征选择器
    selector = FeatureSelector(
        model_path=args.model_path,
        method=args.method,
        n_features=args.n_features,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    # 选择特征
    selected_features = selector.select_features(X, y)
    
    # 保存所选特征
    selector.save_selected_features()
    
    # 绘制特征重要性
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(
        args.output_dir, 
        f"feature_importance_{args.method}_{args.n_features}_{timestamp}.png"
    )
    selector.plot_feature_importance(plot_path)
    
    # 如果需要重新训练
    if args.retrain:
        retrain_with_selected_features(
            model_path=args.model_path,
            X_train=X,
            y_train=y,
            selected_features=selected_features,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=args.early_stopping_rounds,
            output_dir=os.path.join(args.output_dir, "models"),
            verbose=args.verbose
        )


if __name__ == "__main__":
    main() 