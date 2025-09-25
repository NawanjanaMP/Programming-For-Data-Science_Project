"""
Data Visualization Module
Comprehensive visualization suite for retail pricing analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
from typing import List, Dict
import logging
import matplotlib.colors as mcolors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for matplotlib
sns.set_theme(style="whitegrid", palette=sns.color_palette("husl"))

class DataVisualizer:
    """
    Comprehensive data visualization for e-commerce analysis
    """
    
    def __init__(self, output_dir: str = 'visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Color schemes
        self.colors = {
            'primary': '#3498db',
            'secondary': '#2ecc71',
            'accent': '#e74c3c',
            'warning': '#f39c12',
            'info': '#9b59b6'
        }
        
        # Convert Plotly Set3 colors ("rgb(...)") to Matplotlib hex colors
        self.plotly_colors = [
            mcolors.to_hex([int(x)/255 for x in c.strip("rgb() ").split(",")])
            for c in px.colors.qualitative.Set3
        ]
    
    def create_price_distribution_plots(self, df: pd.DataFrame, price_col: str = 'price', 
                                      category_col: str = 'category') -> Dict[str, str]:
        """
        Create comprehensive price distribution visualizations
        """
        plots_created = {}
        
        if price_col not in df.columns:
            logger.error(f"Price column '{price_col}' not found")
            return plots_created
        
        # 1. Overall price distribution histogram
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histogram
        ax1.hist(df[price_col].dropna(), bins=30, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Price Distribution Histogram')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(df[price_col].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor=self.colors['secondary'], alpha=0.7))
        ax2.set_ylabel('Price')
        ax2.set_title('Price Distribution Box Plot')
        ax2.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(df[price_col].dropna(), dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot: Price vs Normal Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Log-transformed distribution
        log_prices = np.log1p(df[price_col].dropna())
        ax4.hist(log_prices, bins=30, alpha=0.7, color=self.colors['accent'], edgecolor='black')
        ax4.set_xlabel('Log(Price + 1)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Log-Transformed Price Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'price_distribution_comprehensive.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created['price_distribution_comprehensive'] = filename
        
        # 2. Price distribution by category
        if category_col in df.columns:
            categories = df[category_col].unique()
            if len(categories) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                category_data = [df[df[category_col] == cat][price_col].dropna() for cat in categories]
                parts = ax.violinplot(category_data, positions=range(len(categories)), 
                                    showmeans=True, showmedians=True)
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(self.plotly_colors[i % len(self.plotly_colors)])
                    pc.set_alpha(0.7)
                
                ax.set_xlabel('Category')
                ax.set_ylabel('Price')
                ax.set_title('Price Distribution by Category (Violin Plot)')
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                filename = os.path.join(self.output_dir, 'price_by_category_violin.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                plots_created['price_by_category_violin'] = filename
                
                # Box plots by category
                fig, ax = plt.subplots(figsize=(12, 8))
                df.boxplot(column=price_col, by=category_col, ax=ax)
                ax.set_title('Price Distribution by Category (Box Plots)')
                ax.set_xlabel('Category')
                ax.set_ylabel('Price')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                filename = os.path.join(self.output_dir, 'price_by_category_boxplot.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                plots_created['price_by_category_boxplot'] = filename
        
        logger.info(f"Created {len(plots_created)} price distribution plots")
        return plots_created
    
    # ---------------------------
    # Other methods go here
    # ---------------------------
    
    def create_rating_analysis_plots(self, df: pd.DataFrame, rating_col: str = 'rating') -> Dict[str, str]:
        plots_created = {}
        if rating_col not in df.columns:
            return plots_created
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[rating_col].dropna(), bins=10, kde=True, ax=ax, color=self.colors['primary'])
        ax.set_title('Rating Distribution')
        filename = os.path.join(self.output_dir, 'rating_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created['rating_distribution'] = filename
        return plots_created
    
    def create_correlation_plots(self, df: pd.DataFrame) -> Dict[str, str]:
        plots_created = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return plots_created
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
        filename = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created['correlation_heatmap'] = filename
        return plots_created
    
    def generate_comprehensive_visualization_report(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Run all visualization functions
        """
        plots_created = {}
        plots_created.update(self.create_price_distribution_plots(df))
        plots_created.update(self.create_rating_analysis_plots(df))
        plots_created.update(self.create_correlation_plots(df))
        return plots_created

if __name__ == "__main__":
    try:
        df = pd.read_csv('data/books_data_cleaned.csv')
        visualizer = DataVisualizer()
        plots_created = visualizer.generate_comprehensive_visualization_report(df)
        print(f"Visualization completed!")
        print(f"Created {len(plots_created)} visualizations")
        print(f"Output directory: {visualizer.output_dir}")
        for plot_name, plot_path in plots_created.items():
            print(f"- {plot_name}: {plot_path}")
    except FileNotFoundError:
        print("Cleaned data file not found. Run data cleaning first.")
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        logger.error(f"Visualization failed: {str(e)}")
