"""
Advanced Statistical Analysis Module
Comprehensive statistical analysis for retail pricing data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for e-commerce data
    """
    
    def __init__(self):
        self.analysis_results = {}
        
    def descriptive_statistics(self, df: pd.DataFrame, numerical_columns: List[str] = None) -> Dict:
        """
        Calculate comprehensive descriptive statistics
        
        Args:
            df (pd.DataFrame): Data to analyze
            numerical_columns (List[str]): Numerical columns to analyze
            
        Returns:
            Dict: Descriptive statistics results
        """
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {}
        
        for col in numerical_columns:
            if col not in df.columns:
                continue
                
            data = df[col].dropna()
            
            if len(data) == 0:
                continue
            
            # Basic statistics
            stats_dict = {
                'count': len(data),
                'mean': float(np.mean(data)),
                'median': float(np.median(data)),
                'std': float(np.std(data)),
                'var': float(np.var(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'range': float(np.max(data) - np.min(data)),
                'q25': float(np.percentile(data, 25)),
                'q75': float(np.percentile(data, 75)),
                'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data))
            }
            
            # Mode (most frequent value)
            try:
                mode_result = stats.mode(data)
                stats_dict['mode'] = float(mode_result.mode[0])
                stats_dict['mode_count'] = int(mode_result.count[0])
            except:
                stats_dict['mode'] = None
                stats_dict['mode_count'] = None
            
            # Coefficient of variation
            if stats_dict['mean'] != 0:
                stats_dict['cv'] = stats_dict['std'] / abs(stats_dict['mean'])
            else:
                stats_dict['cv'] = None
            
            results[col] = stats_dict
            
        self.analysis_results['descriptive_stats'] = results
        logger.info(f"Calculated descriptive statistics for {len(results)} numerical columns")
        
        return results
    
    def price_distribution_analysis(self, df: pd.DataFrame, price_col: str = 'price', category_col: str = 'category') -> Dict:
        """
        Analyze price distributions across categories
        
        Args:
            df (pd.DataFrame): Data to analyze
            price_col (str): Price column name
            category_col (str): Category column name
            
        Returns:
            Dict: Price distribution analysis results
        """
        results = {
            'overall_distribution': {},
            'category_distributions': {},
            'category_comparisons': {}
        }
        
        if price_col not in df.columns:
            logger.error(f"Price column '{price_col}' not found")
            return results
        
        # Overall price distribution
        prices = df[price_col].dropna()
        results['overall_distribution'] = {
            'mean': float(np.mean(prices)),
            'median': float(np.median(prices)),
            'std': float(np.std(prices)),
            'percentiles': {
                '10th': float(np.percentile(prices, 10)),
                '25th': float(np.percentile(prices, 25)),
                '75th': float(np.percentile(prices, 75)),
                '90th': float(np.percentile(prices, 90)),
                '95th': float(np.percentile(prices, 95)),
                '99th': float(np.percentile(prices, 99))
            }
        }
        
        # Category-wise analysis
        if category_col in df.columns:
            category_stats = df.groupby(category_col)[price_col].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)
            
            results['category_distributions'] = category_stats.to_dict('index')
            
            # Statistical tests between categories
            categories = df[category_col].unique()
            if len(categories) > 1:
                # ANOVA test for price differences between categories
                category_groups = [df[df[category_col] == cat][price_col].dropna() 
                                 for cat in categories if len(df[df[category_col] == cat]) > 1]
                
                if len(category_groups) > 1:
                    try:
                        f_stat, p_value = stats.f_oneway(*category_groups)
                        results['category_comparisons']['anova'] = {
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
                    except:
                        results['category_comparisons']['anova'] = None
        
        self.analysis_results['price_distribution'] = results
        logger.info("Completed price distribution analysis")
        
        return results
    
    def rating_analysis(self, df: pd.DataFrame, rating_col: str = 'rating', price_col: str = 'price') -> Dict:
        """
        Analyze rating patterns and relationships
        
        Args:
            df (pd.DataFrame): Data to analyze
            rating_col (str): Rating column name
            price_col (str): Price column name
            
        Returns:
            Dict: Rating analysis results
        """
        results = {
            'rating_distribution': {},
            'price_rating_relationship': {},
            'rating_statistics': {}
        }
        
        if rating_col not in df.columns:
            logger.error(f"Rating column '{rating_col}' not found")
            return results
        
        ratings = df[rating_col].dropna()
        
        # Rating distribution
        rating_counts = ratings.value_counts().sort_index()
        results['rating_distribution'] = {
            'counts': rating_counts.to_dict(),
            'percentages': (rating_counts / len(ratings) * 100).round(2).to_dict()
        }
        
        # Rating statistics
        results['rating_statistics'] = {
            'mean': float(np.mean(ratings)),
            'median': float(np.median(ratings)),
            'mode': float(stats.mode(ratings)[0][0]),
            'std': float(np.std(ratings)),
            'skewness': float(stats.skew(ratings)),
            'kurtosis': float(stats.kurtosis(ratings))
        }
        
        # Price-Rating relationship
        if price_col in df.columns:
            valid_data = df[[price_col, rating_col]].dropna()
            
            if len(valid_data) > 2:
                # Correlation analysis
                pearson_corr, pearson_p = pearsonr(valid_data[price_col], valid_data[rating_col])
                spearman_corr, spearman_p = spearmanr(valid_data[price_col], valid_data[rating_col])
                
                results['price_rating_relationship'] = {
                    'pearson_correlation': {
                        'coefficient': float(pearson_corr),
                        'p_value': float(pearson_p),
                        'significant': pearson_p < 0.05
                    },
                    'spearman_correlation': {
                        'coefficient': float(spearman_corr),
                        'p_value': float(spearman_p),
                        'significant': spearman_p < 0.05
                    }
                }
                
                # Price by rating group analysis
                rating_price_stats = valid_data.groupby(rating_col)[price_col].agg([
                    'count', 'mean', 'median', 'std'
                ]).round(2)
                
                results['price_rating_relationship']['price_by_rating'] = rating_price_stats.to_dict('index')
        
        self.analysis_results['rating_analysis'] = results
        logger.info("Completed rating analysis")
        
        return results
    
    def correlation_analysis(self, df: pd.DataFrame, numerical_columns: List[str] = None) -> Dict:
        """
        Perform comprehensive correlation analysis
        
        Args:
            df (pd.DataFrame): Data to analyze
            numerical_columns (List[str]): Columns to include in correlation
            
        Returns:
            Dict: Correlation analysis results
        """
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter columns that exist in dataframe
        existing_columns = [col for col in numerical_columns if col in df.columns]
        
        if len(existing_columns) < 2:
            logger.warning("Need at least 2 numerical columns for correlation analysis")
            return {}
        
        # Calculate correlation matrices
        correlation_data = df[existing_columns].dropna()
        
        results = {
            'pearson_correlation': {},
            'spearman_correlation': {},
            'significant_correlations': []
        }
        
        # Pearson correlation
        pearson_corr = correlation_data.corr(method='pearson')
        results['pearson_correlation'] = pearson_corr.to_dict()
        
        # Spearman correlation
        spearman_corr = correlation_data.corr(method='spearman')
        results['spearman_correlation'] = spearman_corr.to_dict()
        
        # Find significant correlations
        for i, col1 in enumerate(existing_columns):
            for j, col2 in enumerate(existing_columns):
                if i < j:  # Avoid duplicate pairs
                    pearson_val = pearson_corr.loc[col1, col2]
                    spearman_val = spearman_corr.loc[col1, col2]
                    
                    # Test significance
                    try:
                        _, p_pearson = pearsonr(correlation_data[col1], correlation_data[col2])
                        _, p_spearman = spearmanr(correlation_data[col1], correlation_data[col2])
                        
                        if abs(pearson_val) > 0.3 or abs(spearman_val) > 0.3:
                            results['significant_correlations'].append({
                                'variables': [col1, col2],
                                'pearson': {
                                    'coefficient': float(pearson_val),
                                    'p_value': float(p_pearson),
                                    'significant': p_pearson < 0.05
                                },
                                'spearman': {
                                    'coefficient': float(spearman_val),
                                    'p_value': float(p_spearman),
                                    'significant': p_spearman < 0.05
                                }
                            })
                    except:
                        continue
        
        self.analysis_results['correlation_analysis'] = results
        logger.info(f"Completed correlation analysis for {len(existing_columns)} variables")
        
        return results
    
    def category_analysis(self, df: pd.DataFrame, category_col: str = 'category') -> Dict:
        """
        Analyze categorical data patterns
        
        Args:
            df (pd.DataFrame): Data to analyze
            category_col (str): Category column name
            
        Returns:
            Dict: Category analysis results
        """
        results = {
            'frequency_distribution': {},
            'category_statistics': {},
            'diversity_metrics': {}
        }
        
        if category_col not in df.columns:
            logger.error(f"Category column '{category_col}' not found")
            return results
        
        categories = df[category_col].dropna()
        
        # Frequency distribution
        category_counts = categories.value_counts()
        category_percentages = (category_counts / len(categories) * 100).round(2)
        
        results['frequency_distribution'] = {
            'counts': category_counts.to_dict(),
            'percentages': category_percentages.to_dict(),
            'total_categories': len(category_counts),
            'most_popular': category_counts.index[0] if len(category_counts) > 0 else None,
            'least_popular': category_counts.index[-1] if len(category_counts) > 0 else None
        }
        
        # Category statistics with numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_cols:
            category_stats = df.groupby(category_col)[numerical_cols].agg([
                'count', 'mean', 'median', 'std'
            ]).round(2)
            
            results['category_statistics'] = {col: category_stats[col].to_dict('index') 
                                           for col in numerical_cols}
        
        # Diversity metrics
        results['diversity_metrics'] = {
            'shannon_diversity': self._calculate_shannon_diversity(category_counts),
            'simpson_diversity': self._calculate_simpson_diversity(category_counts),
            'gini_coefficient': self._calculate_gini_coefficient(category_counts)
        }
        
        self.analysis_results['category_analysis'] = results
        logger.info("Completed category analysis")
        
        return results
    
    def hypothesis_testing(self, df: pd.DataFrame, test_configs: List[Dict]) -> Dict:
        """
        Perform various hypothesis tests
        
        Args:
            df (pd.DataFrame): Data to analyze
            test_configs (List[Dict]): Test configurations
            
        Returns:
            Dict: Hypothesis test results
        """
        results = {}
        
        for config in test_configs:
            test_name = config.get('name', 'unnamed_test')
            test_type = config.get('type')
            
            try:
                if test_type == 'ttest_independent':
                    # Independent t-test
                    group_col = config['group_column']
                    value_col = config['value_column']
                    groups = config['groups']
                    
                    group1_data = df[df[group_col] == groups[0]][value_col].dropna()
                    group2_data = df[df[group_col] == groups[1]][value_col].dropna()
                    
                    if len(group1_data) > 1 and len(group2_data) > 1:
                        stat, p_value = ttest_ind(group1_data, group2_data)
                        
                        results[test_name] = {
                            'test_type': 'independent_t_test',
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'group1_mean': float(np.mean(group1_data)),
                            'group2_mean': float(np.mean(group2_data)),
                            'group1_n': len(group1_data),
                            'group2_n': len(group2_data)
                        }
                
                elif test_type == 'mann_whitney':
                    # Mann-Whitney U test
                    group_col = config['group_column']
                    value_col = config['value_column']
                    groups = config['groups']
                    
                    group1_data = df[df[group_col] == groups[0]][value_col].dropna()
                    group2_data = df[df[group_col] == groups[1]][value_col].dropna()
                    
                    if len(group1_data) > 1 and len(group2_data) > 1:
                        stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                        
                        results[test_name] = {
                            'test_type': 'mann_whitney_u',
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'group1_median': float(np.median(group1_data)),
                            'group2_median': float(np.median(group2_data)),
                            'group1_n': len(group1_data),
                            'group2_n': len(group2_data)
                        }
                
                elif test_type == 'chi_square':
                    # Chi-square test of independence
                    col1 = config['column1']
                    col2 = config['column2']
                    
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    results[test_name] = {
                        'test_type': 'chi_square_independence',
                        'chi2_statistic': float(chi2),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof),
                        'significant': p_value < 0.05,
                        'contingency_table': contingency_table.to_dict()
                    }
                    
            except Exception as e:
                logger.error(f"Error in hypothesis test '{test_name}': {str(e)}")
                results[test_name] = {'error': str(e)}
        
        self.analysis_results['hypothesis_tests'] = results
        logger.info(f"Completed {len(results)} hypothesis tests")
        
        return results
    
    def outlier_analysis(self, df: pd.DataFrame, columns: List[str] = None, methods: List[str] = None) -> Dict:
        """
        Comprehensive outlier analysis
        
        Args:
            df (pd.DataFrame): Data to analyze
            columns (List[str]): Columns to analyze
            methods (List[str]): Outlier detection methods
            
        Returns:
            Dict: Outlier analysis results
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        if methods is None:
            methods = ['iqr', 'zscore', 'modified_zscore']
        
        results = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            data = df[col].dropna()
            if len(data) == 0:
                continue
                
            col_results = {}
            
            for method in methods:
                outliers = []
                
                if method == 'iqr':
                    Q1 = np.percentile(data, 25)
                    Q3 = np.percentile(data, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = data[(data < lower_bound) | (data > upper_bound)].tolist()
                    
                elif method == 'zscore':
                    z_scores = np.abs((data - np.mean(data)) / np.std(data))
                    outliers = data[z_scores > 3].tolist()
                    
                elif method == 'modified_zscore':
                    median = np.median(data)
                    mad = np.median(np.abs(data - median))
                    modified_z_scores = 0.6745 * (data - median) / mad
                    outliers = data[np.abs(modified_z_scores) > 3.5].tolist()
                
                col_results[method] = {
                    'outliers': outliers,
                    'outlier_count': len(outliers),
                    'outlier_percentage': round((len(outliers) / len(data)) * 100, 2)
                }
            
            results[col] = col_results
        
        self.analysis_results['outlier_analysis'] = results
        logger.info(f"Completed outlier analysis for {len(results)} columns")
        
        return results
    
    def _calculate_shannon_diversity(self, counts: pd.Series) -> float:
        """Calculate Shannon diversity index"""
        proportions = counts / counts.sum()
        return float(-np.sum(proportions * np.log2(proportions)))
    
    def _calculate_simpson_diversity(self, counts: pd.Series) -> float:
        """Calculate Simpson diversity index"""
        proportions = counts / counts.sum()
        return float(1 - np.sum(proportions ** 2))
    
    def _calculate_gini_coefficient(self, counts: pd.Series) -> float:
        """Calculate Gini coefficient for distribution inequality"""
        sorted_counts = np.sort(counts.values)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n)
    
    def generate_comprehensive_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive statistical analysis report
        
        Args:
            df (pd.DataFrame): Data to analyze
            
        Returns:
            Dict: Comprehensive analysis report
        """
        logger.info("Starting comprehensive statistical analysis...")
        
        # Run all analyses
        self.descriptive_statistics(df)
        self.price_distribution_analysis(df)
        self.rating_analysis(df)
        self.correlation_analysis(df)
        self.category_analysis(df)
        self.outlier_analysis(df)
        
        # Example hypothesis tests for books data
        if 'category' in df.columns and 'price' in df.columns:
            categories = df['category'].unique()
            if len(categories) >= 2:
                test_configs = [
                    {
                        'name': 'fiction_vs_nonfiction_price',
                        'type': 'mann_whitney',
                        'group_column': 'category',
                        'value_column': 'price',
                        'groups': [categories[0], categories[1]]
                    }
                ]
                self.hypothesis_testing(df, test_configs)
        
        # Create summary report
        report = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_overview': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns)
            },
            'analysis_results': self.analysis_results,
            'key_insights': self._extract_key_insights()
        }
        
        logger.info("Comprehensive statistical analysis completed")
        return report
    
    def _extract_key_insights(self) -> List[str]:
        """Extract key insights from analysis results"""
        insights = []
        
        # Price insights
        if 'price_distribution' in self.analysis_results:
            price_data = self.analysis_results['price_distribution']['overall_distribution']
            mean_price = price_data.get('mean', 0)
            median_price = price_data.get('median', 0)
            
            if mean_price > median_price * 1.2:
                insights.append("Price distribution is right-skewed, indicating presence of high-priced items")
        
        # Correlation insights
        if 'correlation_analysis' in self.analysis_results:
            sig_corrs = self.analysis_results['correlation_analysis'].get('significant_correlations', [])
            if sig_corrs:
                insights.append(f"Found {len(sig_corrs)} significant correlations between variables")
        
        # Category insights
        if 'category_analysis' in self.analysis_results:
            freq_data = self.analysis_results['category_analysis']['frequency_distribution']
            total_cats = freq_data.get('total_categories', 0)
            if total_cats > 0:
                insights.append(f"Dataset contains {total_cats} unique categories")
        
        # Outlier insights
        if 'outlier_analysis' in self.analysis_results:
            for col, methods in self.analysis_results['outlier_analysis'].items():
                for method, results in methods.items():
                    outlier_pct = results.get('outlier_percentage', 0)
                    if outlier_pct > 5:
                        insights.append(f"High outlier percentage ({outlier_pct}%) in {col} using {method} method")
        
        return insights

if __name__ == "__main__":
    # Example usage
    try:
        # Load cleaned data
        df = pd.read_csv('data/books_data_cleaned.csv')
        
        # Initialize analyzer
        analyzer = StatisticalAnalyzer()
        
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report(df)
        
        # Save results
        import json
        with open('data/statistical_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("Statistical analysis completed!")
        print(f"Analyzed {report['dataset_overview']['total_rows']} rows")
        print(f"Key insights: {len(report['key_insights'])}")
        
        # Display key insights
        for insight in report['key_insights']:
            print(f"- {insight}")
            
    except FileNotFoundError:
        print("Cleaned data file not found. Run data cleaning first.")
    except Exception as e:
        print(f"Error during statistical analysis: {str(e)}")
        logger.error(f"Statistical analysis failed: {str(e)}")
