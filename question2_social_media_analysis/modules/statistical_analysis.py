"""
Advanced Statistical Analysis Module
Comprehensive statistical analysis for retail pricing data
"""
import warnings
import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, pearsonr, spearmanr
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional

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
            except Exception:
                stats_dict['mode'] = None
                stats_dict['mode_count'] = None
            
            # Coefficient of variation
            if stats_dict['mean'] != 0:
                stats_dict['coef_var'] = float(stats_dict['std'] / stats_dict['mean'])
            else:
                stats_dict['coef_var'] = None
            
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
                # Example: t-test between first two categories
                cat1, cat2 = categories[:2]
                prices1 = df[df[category_col] == cat1][price_col].dropna()
                prices2 = df[df[category_col] == cat2][price_col].dropna()
                if len(prices1) > 1 and len(prices2) > 1:
                    t_stat, p_val = ttest_ind(prices1, prices2, equal_var=False)
                    results['category_comparisons'][f"{cat1}_vs_{cat2}_ttest"] = {
                        't_stat': float(t_stat),
                        'p_value': float(p_val)
                    }
        
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
        try:
            mode_result = stats.mode(ratings)
            mode_val = float(mode_result.mode[0])
        except Exception:
            mode_val = None
        results['rating_statistics'] = {
            'mean': float(np.mean(ratings)),
            'median': float(np.median(ratings)),
            'mode': mode_val,
            'std': float(np.std(ratings)),
            'skewness': float(stats.skew(ratings)),
            'kurtosis': float(stats.kurtosis(ratings))
        }
        
        # Price-Rating relationship
        if price_col in df.columns:
            valid_data = df[[price_col, rating_col]].dropna()
            if len(valid_data) > 2:
                pearson_corr, pearson_p = pearsonr(valid_data[price_col], valid_data[rating_col])
                spearman_corr, spearman_p = spearmanr(valid_data[price_col], valid_data[rating_col])
                results['price_rating_relationship'] = {
                    'pearson_correlation': float(pearson_corr),
                    'pearson_p_value': float(pearson_p),
                    'spearman_correlation': float(spearman_corr),
                    'spearman_p_value': float(spearman_p)
                }
        
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
                if i < j:
                    corr_val, p_val = pearsonr(correlation_data[col1], correlation_data[col2])
                    if abs(corr_val) > 0.5 and p_val < 0.05:
                        results['significant_correlations'].append({
                            'var1': col1,
                            'var2': col2,
                            'pearson_corr': float(corr_val),
                            'p_value': float(p_val)
                        })
        
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
            test_type = config.get('test_type')
            col1 = config.get('col1')
            col2 = config.get('col2')
            group_col = config.get('group_col')
            if test_type == 'ttest' and col1 and col2 and group_col:
                groups = df[group_col].dropna().unique()
                if len(groups) == 2:
                    data1 = df[df[group_col] == groups[0]][col1].dropna()
                    data2 = df[df[group_col] == groups[1]][col2].dropna()
                    if len(data1) > 1 and len(data2) > 1:
                        t_stat, p_val = ttest_ind(data1, data2, equal_var=False)
                        results[f"{groups[0]}_vs_{groups[1]}_ttest"] = {
                            't_stat': float(t_stat),
                            'p_value': float(p_val)
                        }
        
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
            methods = ['iqr', 'zscore']
        
        results = {}
        
        for col in columns:
            data = df[col].dropna()
            col_results = {}
            if 'iqr' in methods:
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                col_results['iqr_outliers'] = outliers.tolist()
            if 'zscore' in methods:
                z_scores = np.abs(stats.zscore(data))
                outliers = data[z_scores > 3]
                col_results['zscore_outliers'] = outliers.tolist()
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
            test_configs = [{
                'test_type': 'ttest',
                'col1': 'price',
                'col2': 'price',
                'group_col': 'category'
            }]
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
            pdist = self.analysis_results['price_distribution']
            mean_price = pdist['overall_distribution'].get('mean')
            median_price = pdist['overall_distribution'].get('median')
            insights.append(f"Average price: {mean_price:.2f}, Median price: {median_price:.2f}")
        
        # Correlation insights
        if 'correlation_analysis' in self.analysis_results:
            corr = self.analysis_results['correlation_analysis']
            for sig in corr.get('significant_correlations', []):
                insights.append(f"Strong correlation between {sig['var1']} and {sig['var2']}: {sig['pearson_corr']:.2f} (p={sig['p_value']:.3f})")
        
        # Category insights
        if 'category_analysis' in self.analysis_results:
            cat = self.analysis_results['category_analysis']
            most_pop = cat['frequency_distribution'].get('most_popular')
            least_pop = cat['frequency_distribution'].get('least_popular')
            insights.append(f"Most popular category: {most_pop}, Least popular: {least_pop}")
        
        # Outlier insights
        if 'outlier_analysis' in self.analysis_results:
            outliers = self.analysis_results['outlier_analysis']
            for col, res in outliers.items():
                if res.get('iqr_outliers'):
                    insights.append(f"{len(res['iqr_outliers'])} IQR outliers detected in {col}")
                if res.get('zscore_outliers'):
                    insights.append(f"{len(res['zscore_outliers'])} Z-score outliers detected in {col}")
        
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
            json.dump(report, f, indent=2)
        
        print("Statistical analysis completed!")
        print(f"Analyzed {report['dataset_overview']['total_rows']} rows")
        print(f"Key insights: {len(report['key_insights'])}")
        
        # Display key insights
        for insight in report['key_insights']:
            print(insight)
            
    except FileNotFoundError:
        print("Cleaned data file not found. Run data cleaning first.")
    except Exception as e:
        print(f"Error during statistical analysis: {str(e)}")
        logger.error(f"Statistical analysis failed: {str(e)}")