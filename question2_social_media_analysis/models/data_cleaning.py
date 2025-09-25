"""
Data Cleaning and Preprocessing Pipeline
Comprehensive data cleaning for scraped retail pricing data
"""

import pandas as pd
import numpy as np
import re
import string
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Optional
import unicodedata
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing pipeline
    """
    
    def __init__(self):
        self.quality_report = {}
        self.cleaning_log = []
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
            
        # Convert to string if not already
        text = str(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-.,!?;:]', '', text)
        
        # Strip and return
        return text.strip()
    
    def extract_hashtags_mentions(self, text: str) -> Dict[str, List[str]]:
        """
        Extract hashtags and mentions from text
        
        Args:
            text (str): Text to process
            
        Returns:
            Dict: Dictionary with hashtags and mentions
        """
        if pd.isna(text):
            return {'hashtags': [], 'mentions': []}
            
        text = str(text)
        
        hashtags = re.findall(r'#\w+', text.lower())
        mentions = re.findall(r'@\w+', text.lower())
        
        return {
            'hashtags': hashtags,
            'mentions': mentions
        }
    
    def standardize_datetime(self, date_col: pd.Series, format_hints: List[str] = None) -> pd.Series:
        """
        Standardize datetime columns across different formats
        
        Args:
            date_col (pd.Series): Date column to standardize
            format_hints (List[str]): Potential date formats to try
            
        Returns:
            pd.Series: Standardized datetime series
        """
        if format_hints is None:
            format_hints = [
                '%Y-%m-%d',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%d-%m-%Y',
                '%Y/%m/%d'
            ]
        
        def parse_date(date_str):
            if pd.isna(date_str):
                return pd.NaT
                
            date_str = str(date_str).strip()
            
            # Try pandas built-in parser first
            try:
                return pd.to_datetime(date_str)
            except:
                pass
            
            # Try specific formats
            for fmt in format_hints:
                try:
                    return datetime.strptime(date_str, fmt)
                except:
                    continue
                    
            # If all else fails, return NaT
            logger.warning(f"Could not parse date: {date_str}")
            return pd.NaT
        
        return date_col.apply(parse_date)
    
    def handle_missing_data(self, df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing data with different strategies for different columns
        
        Args:
            df (pd.DataFrame): DataFrame to clean
            strategy (Dict[str, str]): Strategy for each column type
            
        Returns:
            pd.DataFrame: DataFrame with missing data handled
        """
        if strategy is None:
            strategy = {
                'numeric': 'median',
                'categorical': 'mode',
                'text': 'empty_string',
                'datetime': 'forward_fill'
            }
        
        df_clean = df.copy()
        missing_report = {}
        
        for col in df_clean.columns:
            missing_count = df_clean[col].isna().sum()
            missing_percent = (missing_count / len(df_clean)) * 100
            
            if missing_count > 0:
                missing_report[col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_percent, 2)
                }
                
                # Determine column type and apply appropriate strategy
                if df_clean[col].dtype in ['float64', 'int64']:
                    if strategy['numeric'] == 'median':
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    elif strategy['numeric'] == 'mean':
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    elif strategy['numeric'] == 'zero':
                        df_clean[col].fillna(0, inplace=True)
                        
                elif df_clean[col].dtype == 'object':
                    # Check if it's categorical or text
                    unique_ratio = len(df_clean[col].unique()) / len(df_clean)
                    
                    if unique_ratio < 0.1:  # Likely categorical
                        if strategy['categorical'] == 'mode':
                            mode_val = df_clean[col].mode()
                            if len(mode_val) > 0:
                                df_clean[col].fillna(mode_val[0], inplace=True)
                            else:
                                df_clean[col].fillna('Unknown', inplace=True)
                        elif strategy['categorical'] == 'unknown':
                            df_clean[col].fillna('Unknown', inplace=True)
                    else:  # Likely text
                        if strategy['text'] == 'empty_string':
                            df_clean[col].fillna('', inplace=True)
                            
                elif df_clean[col].dtype in ['datetime64[ns]']:
                    if strategy['datetime'] == 'forward_fill':
                        df_clean[col].fillna(method='ffill', inplace=True)
        
        self.quality_report['missing_data'] = missing_report
        self.cleaning_log.append(f"Handled missing data for {len(missing_report)} columns")
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame, subset: List[str] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            df (pd.DataFrame): DataFrame to deduplicate
            subset (List[str]): Columns to consider for duplicates
            keep (str): Which duplicate to keep
            
        Returns:
            pd.DataFrame: Deduplicated DataFrame
        """
        initial_count = len(df)
        
        if subset:
            df_clean = df.drop_duplicates(subset=subset, keep=keep)
        else:
            df_clean = df.drop_duplicates(keep=keep)
            
        duplicates_removed = initial_count - len(df_clean)
        
        self.quality_report['duplicates'] = {
            'initial_rows': initial_count,
            'final_rows': len(df_clean),
            'duplicates_removed': duplicates_removed
        }
        
        self.cleaning_log.append(f"Removed {duplicates_removed} duplicate rows")
        
        return df_clean
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None, method: str = 'iqr') -> Dict[str, List[int]]:
        """
        Detect outliers in numerical columns
        
        Args:
            df (pd.DataFrame): DataFrame to check
            columns (List[str]): Columns to check for outliers
            method (str): Method to use ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            Dict[str, List[int]]: Dictionary of outlier indices for each column
        """
        outliers = {}
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns or df[col].dtype not in [np.number]:
                continue
                
            col_outliers = []
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = df[z_scores > 3].index.tolist()
            
            outliers[col] = col_outliers
            
        self.quality_report['outliers'] = outliers
        
        return outliers
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive data quality validation
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            Dict: Data quality report
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024*1024), 2),
            'column_info': {},
            'data_types': df.dtypes.to_dict(),
            'validation_timestamp': datetime.now().isoformat()
        }
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isna().sum()),
                'null_percentage': round((df[col].isna().sum() / len(df)) * 100, 2),
                'unique_count': int(df[col].nunique()),
                'unique_percentage': round((df[col].nunique() / len(df)) * 100, 2)
            }
            
            if df[col].dtype in [np.number]:
                col_info.update({
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'std': float(df[col].std()) if not df[col].isna().all() else None
                })
            
            quality_report['column_info'][col] = col_info
        
        return quality_report
    
    def clean_books_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Specialized cleaning for books data
        
        Args:
            df (pd.DataFrame): Raw books DataFrame
            
        Returns:
            pd.DataFrame: Cleaned books DataFrame
        """
        logger.info("Starting books data cleaning...")
        df_clean = df.copy()
        
        # Clean text columns
        text_columns = ['title', 'category']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self.clean_text)
        
        # Validate price column
        if 'price' in df_clean.columns:
            # Remove negative prices
            df_clean = df_clean[df_clean['price'] >= 0]
            
            # Cap extremely high prices (potential data errors)
            price_cap = df_clean['price'].quantile(0.99)
            df_clean.loc[df_clean['price'] > price_cap, 'price'] = price_cap
        
        # Validate rating column
        if 'rating' in df_clean.columns:
            # Ensure ratings are between 1 and 5
            df_clean['rating'] = df_clean['rating'].clip(1, 5)
        
        # Standardize category names
        if 'category' in df_clean.columns:
            df_clean['category'] = df_clean['category'].str.title()
            df_clean['category'] = df_clean['category'].replace('', 'Unknown')
        
        # Handle datetime columns
        datetime_columns = ['scraped_at']
        for col in datetime_columns:
            if col in df_clean.columns:
                df_clean[col] = self.standardize_datetime(df_clean[col])
        
        # Remove duplicates based on title and price
        df_clean = self.remove_duplicates(df_clean, subset=['title', 'price'])
        
        # Handle missing data
        df_clean = self.handle_missing_data(df_clean)
        
        # Detect outliers
        self.detect_outliers(df_clean, columns=['price', 'rating'])
        
        logger.info("Books data cleaning completed")
        return df_clean
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for analysis
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with derived features
        """
        df_enhanced = df.copy()
        
        # Price categories
        if 'price' in df_enhanced.columns:
            df_enhanced['price_category'] = pd.cut(
                df_enhanced['price'],
                bins=[0, 10, 25, 50, 100, float('inf')],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
        
        # Rating categories
        if 'rating' in df_enhanced.columns:
            df_enhanced['rating_category'] = pd.cut(
                df_enhanced['rating'],
                bins=[0, 2, 3, 4, 5],
                labels=['Poor', 'Fair', 'Good', 'Excellent']
            )
        
        # Title length
        if 'title' in df_enhanced.columns:
            df_enhanced['title_length'] = df_enhanced['title'].str.len()
            df_enhanced['title_word_count'] = df_enhanced['title'].str.split().str.len()
        
        # Value score (rating / price ratio)
        if 'rating' in df_enhanced.columns and 'price' in df_enhanced.columns:
            df_enhanced['value_score'] = df_enhanced['rating'] / (df_enhanced['price'] + 1)  # +1 to avoid division by zero
        
        return df_enhanced
    
    def generate_cleaning_report(self) -> Dict:
        """
        Generate comprehensive cleaning report
        
        Returns:
            Dict: Cleaning report with quality metrics and log
        """
        report = {
            'cleaning_timestamp': datetime.now().isoformat(),
            'quality_metrics': self.quality_report,
            'cleaning_steps': self.cleaning_log,
            'recommendations': []
        }
        
        # Add recommendations based on quality metrics
        if 'missing_data' in self.quality_report:
            high_missing = [col for col, info in self.quality_report['missing_data'].items() 
                          if info['percentage'] > 20]
            if high_missing:
                report['recommendations'].append(
                    f"Consider dropping columns with high missing data: {', '.join(high_missing)}"
                )
        
        if 'outliers' in self.quality_report:
            high_outliers = [col for col, outliers in self.quality_report['outliers'].items() 
                           if len(outliers) > 0]
            if high_outliers:
                report['recommendations'].append(
                    f"Review outliers in columns: {', '.join(high_outliers)}"
                )
        
        return report

class DataPreprocessor:
    """
    Advanced data preprocessing for machine learning readiness
    """
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df (pd.DataFrame): DataFrame to encode
            categorical_columns (List[str]): Columns to encode
            
        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col not in df_encoded.columns:
                continue
                
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                
            # Handle missing values
            df_encoded[col] = df_encoded[col].fillna('Unknown')
            
            # Fit and transform
            df_encoded[f'{col}_encoded'] = self.encoders[col].fit_transform(df_encoded[col])
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, numerical_columns: List[str] = None) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df (pd.DataFrame): DataFrame to scale
            numerical_columns (List[str]): Columns to scale
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_scaled = df.copy()
        
        for col in numerical_columns:
            if col not in df_scaled.columns:
                continue
                
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
            
            # Reshape for sklearn
            values = df_scaled[col].values.reshape(-1, 1)
            df_scaled[f'{col}_scaled'] = self.scalers[col].fit_transform(values).flatten()
        
        return df_scaled

if __name__ == "__main__":
    # Example usage
    cleaner = DataCleaner()
    
    # Load sample data (assuming it exists)
    try:
        df = pd.read_csv('data/books_data.csv')
        
        # Clean the data
        df_clean = cleaner.clean_books_data(df)
        
        # Create derived features
        df_enhanced = cleaner.create_derived_features(df_clean)
        
        # Generate quality report
        quality_report = cleaner.validate_data_quality(df_enhanced)
        cleaning_report = cleaner.generate_cleaning_report()
        
        # Save cleaned data
        df_enhanced.to_csv('data/books_data_cleaned.csv', index=False)
        
        # Preprocessing for ML
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.encode_categorical_features(df_enhanced)
        df_processed = preprocessor.scale_numerical_features(df_processed)
        
        print("Data cleaning completed successfully!")
        print(f"Original rows: {len(df)}, Cleaned rows: {len(df_enhanced)}")
        print(f"Quality report: {quality_report['total_columns']} columns processed")
        
    except FileNotFoundError:
        print("Sample data file not found. Run web scraper first.")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        logger.error(f"Data cleaning failed: {str(e)}")