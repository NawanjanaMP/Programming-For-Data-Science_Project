"""
Predictive Analysis Module
Statistical modeling and prediction for e-commerce data
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictiveAnalyzer:
    """
    Comprehensive predictive analysis for e-commerce data
    """
    
    def __init__(self, output_dir: str = 'models'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.results = {}
        
    def preprocess_for_modeling(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Preprocess data for machine learning modeling
        
        Args:
            df (pd.DataFrame): Input data
            target_col (str): Target column name
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Features, target, feature names
        """
        df_model = df.copy()
        
        # Remove target column and non-predictive columns
        non_predictive_cols = ['url', 'source', 'scraped_at', target_col]
        feature_cols = [col for col in df_model.columns if col not in non_predictive_cols]
        
        # Handle categorical variables
        categorical_cols = df_model[feature_cols].select_dtypes(include=['object']).columns
        numerical_cols = df_model[feature_cols].select_dtypes(include=[np.number]).columns
        
        processed_features = []
        feature_names = []
        
        # Process numerical features
        if len(numerical_cols) > 0:
            numerical_data = df_model[numerical_cols].fillna(df_model[numerical_cols].median())
            
            # Scale numerical features
            scaler_key = f"{target_col}_numerical"
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                numerical_scaled = self.scalers[scaler_key].fit_transform(numerical_data)
            else:
                numerical_scaled = self.scalers[scaler_key].transform(numerical_data)
            
            processed_features.append(numerical_scaled)
            feature_names.extend(numerical_cols.tolist())
        
        # Process categorical features
        for col in categorical_cols:
            encoder_key = f"{target_col}_{col}"
            if encoder_key not in self.encoders:
                self.encoders[encoder_key] = LabelEncoder()
                # Handle missing values
                col_data = df_model[col].fillna('Unknown')
                encoded_data = self.encoders[encoder_key].fit_transform(col_data)
            else:
                col_data = df_model[col].fillna('Unknown')
                # Handle unseen categories
                known_categories = self.encoders[encoder_key].classes_
                col_data = col_data.apply(lambda x: x if x in known_categories else 'Unknown')
                encoded_data = self.encoders[encoder_key].transform(col_data)
            
            processed_features.append(encoded_data.reshape(-1, 1))
            feature_names.append(f"{col}_encoded")
        
        # Combine all features
        if processed_features:
            X = np.hstack(processed_features)
        else:
            raise ValueError("No valid features found for modeling")
        
        # Prepare target variable
        y = df_model[target_col].fillna(df_model[target_col].median())
        
        return X, y, feature_names
    
    def price_prediction_model(self, df: pd.DataFrame, target_col: str = 'price') -> Dict[str, Any]:
        """
        Build models to predict book prices based on ratings and other features
        
        Args:
            df (pd.DataFrame): Input data
            target_col (str): Target column name (price)
            
        Returns:
            Dict[str, Any]: Model results and metrics
        """
        logger.info(f"Building price prediction models for {target_col}")
        
        try:
            # Preprocess data
            X, y, feature_names = self.preprocess_for_modeling(df, target_col)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Initialize models
            models = {
                'linear_regression': LinearRegression(),
                'ridge_regression': Ridge(alpha=1.0),
                'lasso_regression': Lasso(alpha=1.0),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            results = {
                'feature_names': feature_names,
                'data_shape': {
                    'total_samples': len(X),
                    'features': len(feature_names),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                },
                'model_performance': {}
            }
            
            # Train and evaluate models
            for model_name, model in models.items():
                logger.info(f"Training {model_name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                          scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
                
                model_results = {
                    'train_mse': float(train_mse),
                    'test_mse': float(test_mse),
                    'train_rmse': float(np.sqrt(train_mse)),
                    'test_rmse': float(np.sqrt(test_mse)),
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'test_mae': float(test_mae),
                    'cv_rmse_mean': float(cv_rmse),
                    'cv_rmse_std': float(np.sqrt(-cv_scores).std())
                }
                
                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(feature_names, model.feature_importances_))
                    model_results['feature_importance'] = feature_importance
                elif hasattr(model, 'coef_'):
                    # For linear models, use absolute coefficients as importance
                    feature_importance = dict(zip(feature_names, np.abs(model.coef_)))
                    model_results['feature_importance'] = feature_importance
                
                results['model_performance'][model_name] = model_results
                
                # Save model
                model_path = os.path.join(self.output_dir, f'{model_name}_price_model.joblib')
                joblib.dump(model, model_path)
                self.models[f'{model_name}_price'] = model
            
            # Find best model
            best_model_name = min(results['model_performance'].keys(), 
                                key=lambda k: results['model_performance'][k]['test_rmse'])
            results['best_model'] = best_model_name
            
            # Generate predictions for visualization
            best_model = self.models[f'{best_model_name}_price']
            results['predictions'] = {
                'y_test_actual': y_test.tolist(),
                'y_test_predicted': best_model.predict(X_test).tolist()
            }
            
            self.results['price_prediction'] = results
            logger.info(f"Price prediction modeling completed. Best model: {best_model_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in price prediction modeling: {str(e)}")
            return {'error': str(e)}
    
    def rating_prediction_model(self, df: pd.DataFrame, target_col: str = 'rating') -> Dict[str, Any]:
        """
        Build models to predict ratings based on price and other features
        
        Args:
            df (pd.DataFrame): Input data
            target_col (str): Target column name (rating)
            
        Returns:
            Dict[str, Any]: Model results and metrics
        """
        logger.info(f"Building rating prediction models for {target_col}")
        
        try:
            # Preprocess data
            X, y, feature_names = self.preprocess_for_modeling(df, target_col)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Initialize models (same as price prediction but could be different)
            models = {
                'linear_regression': LinearRegression(),
                'ridge_regression': Ridge(alpha=1.0),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            results = {
                'feature_names': feature_names,
                'data_shape': {
                    'total_samples': len(X),
                    'features': len(feature_names),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                },
                'model_performance': {}
            }
            
            # Train and evaluate models
            for model_name, model in models.items():
                logger.info(f"Training {model_name} for rating prediction...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                model_results = {
                    'train_mse': float(train_mse),
                    'test_mse': float(test_mse),
                    'train_rmse': float(np.sqrt(train_mse)),
                    'test_rmse': float(np.sqrt(test_mse)),
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'test_mae': float(test_mae)
                }
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(feature_names, model.feature_importances_))
                    model_results['feature_importance'] = feature_importance
                elif hasattr(model, 'coef_'):
                    feature_importance = dict(zip(feature_names, np.abs(model.coef_)))
                    model_results['feature_importance'] = feature_importance
                
                results['model_performance'][model_name] = model_results
                
                # Save model
                model_path = os.path.join(self.output_dir, f'{model_name}_rating_model.joblib')
                joblib.dump(model, model_path)
                self.models[f'{model_name}_rating'] = model
            
            # Find best model
            best_model_name = min(results['model_performance'].keys(), 
                                key=lambda k: results['model_performance'][k]['test_rmse'])
            results['best_model'] = best_model_name
            
            self.results['rating_prediction'] = results
            logger.info(f"Rating prediction modeling completed. Best model: {best_model_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in rating prediction modeling: {str(e)}")
            return {'error': str(e)}
    
    def category_pricing_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze pricing patterns across categories
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Any]: Category pricing analysis results
        """
        logger.info("Analyzing category pricing patterns...")
        
        if 'category' not in df.columns or 'price' not in df.columns:
            return {'error': 'Required columns (category, price) not found'}
        
        results = {
            'category_statistics': {},
            'price_trends': {},
            'category_clusters': {},
            'pricing_insights': []
        }
        
        # Category statistics
        category_stats = df.groupby('category').agg({
            'price': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'rating': ['mean', 'std'] if 'rating' in df.columns else []
        }).round(2)
        
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
        results['category_statistics'] = category_stats.to_dict('index')
        
        # Price trends analysis
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]['price']
            
            if len(cat_data) > 1:
                # Statistical tests for normality and outliers
                _, normality_p = stats.normaltest(cat_data)
                
                # Price percentiles
                percentiles = np.percentile(cat_data, [25, 50, 75, 90, 95])
                
                results['price_trends'][category] = {
                    'is_normal': normality_p > 0.05,
                    'normality_p_value': float(normality_p),
                    'percentiles': {
                        '25th': float(percentiles[0]),
                        '50th': float(percentiles[1]),
                        '75th': float(percentiles[2]),
                        '90th': float(percentiles[3]),
                        '95th': float(percentiles[4])
                    },
                    'coefficient_variation': float(cat_data.std() / cat_data.mean()) if cat_data.mean() > 0 else None
                }
        
        # Clustering categories based on pricing characteristics
        try:
            # Prepare data for clustering
            cluster_features = []
            category_names = []
            
            for category in df['category'].unique():
                cat_data = df[df['category'] == category]
                if len(cat_data) >= 3:  # Minimum samples for meaningful statistics
                    features = [
                        cat_data['price'].mean(),
                        cat_data['price'].std(),
                        cat_data['price'].median(),
                        len(cat_data)
                    ]
                    if 'rating' in df.columns:
                        features.append(cat_data['rating'].mean())
                    
                    cluster_features.append(features)
                    category_names.append(category)
            
            if len(cluster_features) >= 3:
                cluster_features = np.array(cluster_features)
                
                # Standardize features
                scaler = StandardScaler()
                cluster_features_scaled = scaler.fit_transform(cluster_features)
                
                # Perform clustering
                n_clusters = min(3, len(cluster_features))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(cluster_features_scaled)
                
                # Store clustering results
                for i, category in enumerate(category_names):
                    results['category_clusters'][category] = {
                        'cluster': int(cluster_labels[i]),
                        'features': cluster_features[i].tolist()
                    }
                
                # Cluster characteristics
                cluster_characteristics = {}
                for cluster_id in range(n_clusters):
                    cluster_categories = [cat for cat, info in results['category_clusters'].items() 
                                        if info['cluster'] == cluster_id]
                    cluster_data = df[df['category'].isin(cluster_categories)]
                    
                    cluster_characteristics[f'cluster_{cluster_id}'] = {
                        'categories': cluster_categories,
                        'avg_price': float(cluster_data['price'].mean()),
                        'avg_rating': float(cluster_data['rating'].mean()) if 'rating' in df.columns else None,
                        'total_items': len(cluster_data)
                    }
                
                results['cluster_characteristics'] = cluster_characteristics
        
        except Exception as e:
            logger.warning(f"Clustering analysis failed: {str(e)}")
            results['clustering_error'] = str(e)
        
        # Generate insights
        insights = []
        
        # Find most and least expensive categories
        if results['category_statistics']:
            price_means = {cat: stats['price_mean'] for cat, stats in results['category_statistics'].items()}
            most_expensive = max(price_means.keys(), key=lambda k: price_means[k])
            least_expensive = min(price_means.keys(), key=lambda k: price_means[k])
            
            insights.append(f"Most expensive category: {most_expensive} (avg: ${price_means[most_expensive]:.2f})")
            insights.append(f"Least expensive category: {least_expensive} (avg: ${price_means[least_expensive]:.2f})")
        
        # Find categories with high price variability
        if results['price_trends']:
            high_var_categories = [cat for cat, trends in results['price_trends'].items() 
                                 if trends.get('coefficient_variation', 0) > 0.5]
            if high_var_categories:
                insights.append(f"High price variability categories: {', '.join(high_var_categories)}")
        
        results['pricing_insights'] = insights
        self.results['category_pricing'] = results
        
        logger.info("Category pricing analysis completed")
        return results
    
    def recommendation_system(self, df: pd.DataFrame, item_features: List[str] = None) -> Dict[str, Any]:
        """
        Create a basic recommendation system using statistical similarity
        
        Args:
            df (pd.DataFrame): Input data
            item_features (List[str]): Features to use for similarity
            
        Returns:
            Dict[str, Any]: Recommendation system results
        """
        logger.info("Building recommendation system...")
        
        if item_features is None:
            item_features = ['price', 'rating']
            if 'category' in df.columns:
                item_features.append('category')
        
        # Filter available features
        available_features = [feat for feat in item_features if feat in df.columns]
        
        if len(available_features) == 0:
            return {'error': 'No valid features found for recommendation system'}
        
        try:
            # Prepare feature matrix
            feature_matrix = df[available_features].copy()
            
            # Handle categorical variables
            categorical_cols = feature_matrix.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                feature_matrix[col] = le.fit_transform(feature_matrix[col].fillna('Unknown'))
            
            # Handle missing values
            feature_matrix = feature_matrix.fillna(feature_matrix.median())
            
            # Standardize features
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)
            
            # Build nearest neighbors model
            nn_model = NearestNeighbors(n_neighbors=6, metric='cosine')  # 5 + 1 (itself)
            nn_model.fit(feature_matrix_scaled)
            
            # Generate sample recommendations
            sample_recommendations = {}
            
            # Get recommendations for first 5 items
            for idx in range(min(5, len(df))):
                distances, indices = nn_model.kneighbors([feature_matrix_scaled[idx]])
                
                # Exclude the item itself (index 0)
                similar_indices = indices[0][1:]
                similar_distances = distances[0][1:]
                
                recommendations = []
                for i, (sim_idx, distance) in enumerate(zip(similar_indices, similar_distances)):
                    similarity_score = 1 - distance  # Convert distance to similarity
                    
                    rec_item = {
                        'index': int(sim_idx),
                        'similarity_score': float(similarity_score),
                        'title': df.iloc[sim_idx].get('title', f'Item {sim_idx}'),
                        'price': float(df.iloc[sim_idx].get('price', 0)),
                        'rating': float(df.iloc[sim_idx].get('rating', 0))
                    }
                    recommendations.append(rec_item)
                
                sample_recommendations[f'item_{idx}'] = {
                    'original_item': {
                        'index': idx,
                        'title': df.iloc[idx].get('title', f'Item {idx}'),
                        'price': float(df.iloc[idx].get('price', 0)),
                        'rating': float(df.iloc[idx].get('rating', 0))
                    },
                    'recommendations': recommendations
                }
            
            # Calculate system statistics
            results = {
                'system_info': {
                    'total_items': len(df),
                    'features_used': available_features,
                    'similarity_metric': 'cosine',
                    'neighbors_per_item': 5
                },
                'sample_recommendations': sample_recommendations,
                'model_saved': False
            }
            
            # Save recommendation model
            rec_model_path = os.path.join(self.output_dir, 'recommendation_model.joblib')
            rec_data = {
                'nn_model': nn_model,
                'scaler': scaler,
                'features': available_features,
                'categorical_encoders': {}
            }
            
            joblib.dump(rec_data, rec_model_path)
            results['model_saved'] = True
            results['model_path'] = rec_model_path
            
            self.results['recommendation_system'] = results
            logger.info("Recommendation system built successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Error building recommendation system: {str(e)}")
            return {'error': str(e)}
    
    def stock_availability_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trends in stock availability vs pricing
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Any]: Stock availability analysis results
        """
        logger.info("Analyzing stock availability trends...")
        
        required_cols = ['price']
        stock_col = 'in_stock' if 'in_stock' in df.columns else None
        
        if not all(col in df.columns for col in required_cols):
            return {'error': f'Required columns {required_cols} not found'}
        
        results = {
            'availability_statistics': {},
            'price_availability_relationship': {},
            'category_availability': {},
            'insights': []
        }
        
        if stock_col:
            # Basic availability statistics
            availability_stats = df[stock_col].value_counts()
            total_items = len(df)
            
            results['availability_statistics'] = {
                'total_items': total_items,
                'in_stock_count': int(availability_stats.get(1, 0)),
                'out_of_stock_count': int(availability_stats.get(0, 0)),
                'in_stock_percentage': float((availability_stats.get(1, 0) / total_items) * 100),
                'out_of_stock_percentage': float((availability_stats.get(0, 0) / total_items) * 100)
            }
            
            # Price vs availability analysis
            in_stock_prices = df[df[stock_col] == 1]['price']
            out_of_stock_prices = df[df[stock_col] == 0]['price']
            
            if len(in_stock_prices) > 0 and len(out_of_stock_prices) > 0:
                # Statistical test
                stat, p_value = stats.mannwhitneyu(in_stock_prices, out_of_stock_prices, 
                                                 alternative='two-sided')
                
                results['price_availability_relationship'] = {
                    'in_stock_price_mean': float(in_stock_prices.mean()),
                    'out_of_stock_price_mean': float(out_of_stock_prices.mean()),
                    'in_stock_price_median': float(in_stock_prices.median()),
                    'out_of_stock_price_median': float(out_of_stock_prices.median()),
                    'mann_whitney_u_stat': float(stat),
                    'mann_whitney_p_value': float(p_value),
                    'significant_difference': p_value < 0.05
                }
            
            # Category-wise availability
            if 'category' in df.columns:
                category_availability = df.groupby('category')[stock_col].agg([
                    'count', 'sum', 'mean'
                ]).round(3)
                category_availability['availability_percentage'] = category_availability['mean'] * 100
                
                results['category_availability'] = category_availability.to_dict('index')
            
            # Generate insights
            insights = []
            
            if results['availability_statistics']['in_stock_percentage'] < 80:
                insights.append("Low overall stock availability (< 80%)")
            
            if 'price_availability_relationship' in results:
                if results['price_availability_relationship']['significant_difference']:
                    in_stock_mean = results['price_availability_relationship']['in_stock_price_mean']
                    out_of_stock_mean = results['price_availability_relationship']['out_of_stock_price_mean']
                    
                    if in_stock_mean > out_of_stock_mean:
                        insights.append("In-stock items tend to be more expensive than out-of-stock items")
                    else:
                        insights.append("Out-of-stock items tend to be more expensive than in-stock items")
            
            results['insights'] = insights
        else:
            results['error'] = 'No stock availability column found'
        
        self.results['stock_availability'] = results
        logger.info("Stock availability analysis completed")
        
        return results
    
    def generate_model_comparison_plot(self) -> str:
        """
        Generate model comparison visualization
        
        Returns:
            str: Path to saved plot
        """
        if 'price_prediction' not in self.results:
            logger.warning("No price prediction results found for plotting")
            return None
        
        try:
            model_performance = self.results['price_prediction']['model_performance']
            
            # Extract metrics
            model_names = list(model_performance.keys())
            test_rmse = [model_performance[model]['test_rmse'] for model in model_names]
            test_r2 = [model_performance[model]['test_r2'] for model in model_names]
            
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # RMSE comparison
            bars1 = ax1.bar(model_names, test_rmse, alpha=0.7)
            ax1.set_title('Model Comparison: Test RMSE')
            ax1.set_ylabel('RMSE')
            ax1.tick_params(axis='x', rotation=45)
            
            # Color best performing model
            best_rmse_idx = np.argmin(test_rmse)
            bars1[best_rmse_idx].set_color('green')
            
            # R² comparison
            bars2 = ax2.bar(model_names, test_r2, alpha=0.7)
            ax2.set_title('Model Comparison: Test R²')
            ax2.set_ylabel('R² Score')
            ax2.tick_params(axis='x', rotation=45)
            
            # Color best performing model
            best_r2_idx = np.argmax(test_r2)
            bars2[best_r2_idx].set_color('green')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, 'model_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Model comparison plot saved: {plot_path}")
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating model comparison plot: {str(e)}")
            return None
    
    def generate_comprehensive_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive predictive analysis report
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Any]: Comprehensive analysis results
        """
        logger.info("Starting comprehensive predictive analysis...")
        
        comprehensive_results = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': df.columns.tolist()
            },
            'analyses_completed': [],
            'models_created': [],
            'key_findings': []
        }
        
        # Run all analyses
        try:
            # Price prediction
            if 'price' in df.columns:
                price_results = self.price_prediction_model(df)
                if 'error' not in price_results:
                    comprehensive_results['analyses_completed'].append('price_prediction')
                    comprehensive_results['models_created'].extend(list(price_results['model_performance'].keys()))
            
            # Rating prediction
            if 'rating' in df.columns:
                rating_results = self.rating_prediction_model(df)
                if 'error' not in rating_results:
                    comprehensive_results['analyses_completed'].append('rating_prediction')
            
            # Category pricing analysis
            category_results = self.category_pricing_analysis(df)
            if 'error' not in category_results:
                comprehensive_results['analyses_completed'].append('category_pricing_analysis')
            
            # Recommendation system
            rec_results = self.recommendation_system(df)
            if 'error' not in rec_results:
                comprehensive_results['analyses_completed'].append('recommendation_system')
            
            # Stock availability analysis
            stock_results = self.stock_availability_analysis(df)
            if 'error' not in stock_results:
                comprehensive_results['analyses_completed'].append('stock_availability_analysis')
            
            # Generate comparison plots
            comparison_plot = self.generate_model_comparison_plot()
            if comparison_plot:
                comprehensive_results['visualizations'] = [comparison_plot]
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            comprehensive_results['error'] = str(e)
        
        # Extract key findings
        key_findings = []
        
        if 'price_prediction' in self.results:
            best_model = self.results['price_prediction']['best_model']
            best_r2 = self.results['price_prediction']['model_performance'][best_model]['test_r2']
            key_findings.append(f"Best price prediction model: {best_model} (R² = {best_r2:.3f})")
        
        if 'category_pricing' in self.results and 'pricing_insights' in self.results['category_pricing']:
            key_findings.extend(self.results['category_pricing']['pricing_insights'])
        
        if 'stock_availability' in self.results and 'insights' in self.results['stock_availability']:
            key_findings.extend(self.results['stock_availability']['insights'])
        
        comprehensive_results['key_findings'] = key_findings
        comprehensive_results['detailed_results'] = self.results
        
        # Save comprehensive results
        results_path = os.path.join(self.output_dir, 'comprehensive_analysis_results.json')
        
        import json
        with open(results_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        logger.info(f"Comprehensive predictive analysis completed. Results saved to {results_path}")
        return comprehensive_results

if __name__ == "__main__":
    # Example usage
    try:
        # Load cleaned data
        df = pd.read_csv('data/books_data_cleaned.csv')
        
        # Initialize analyzer
        analyzer = PredictiveAnalyzer()
        
        # Generate comprehensive analysis
        results = analyzer.generate_comprehensive_report(df)
        
        print("Predictive analysis completed!")
        print(f"Analyses completed: {', '.join(results['analyses_completed'])}")
        print(f"Models created: {len(results['models_created'])}")
        print("\nKey findings:")
        for finding in results['key_findings']:
            print(f"- {finding}")
            
    except FileNotFoundError:
        print("Cleaned data file not found. Run data cleaning first.")
    except Exception as e:
        print(f"Error during predictive analysis: {str(e)}")
        logger.error(f"Predictive analysis failed: {str(e)}")
