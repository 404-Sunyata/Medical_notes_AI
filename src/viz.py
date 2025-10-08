"""Visualization tools for radiology data analysis."""

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os
import logging

from .config import PLOTS_DIR
from .query_tools import QueryTools

logger = logging.getLogger(__name__)

class RadiologyVisualizer:
    """Create visualizations for radiology data analysis."""
    
    def __init__(self, query_tools: QueryTools):
        """
        Initialize with QueryTools instance.
        
        Args:
            query_tools: QueryTools instance with data
        """
        self.query_tools = query_tools
        self.df = query_tools.df
        
        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
    
    def plot_stone_distribution_by_side(self, save_path: Optional[str] = None) -> str:
        """
        Create bar chart of stone distribution by side.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        side_dist = self.query_tools.get_side_distribution()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Left Only', 'Right Only', 'Bilateral', 'No Stones']
        counts = [side_dist['left_only'], side_dist['right_only'], 
                 side_dist['bilateral'], side_dist['no_stones']]
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Stone Distribution by Side', fontsize=16, fontweight='bold')
        ax.set_xlabel('Stone Location', fontsize=12)
        ax.set_ylabel('Number of Records', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(PLOTS_DIR, 'stone_distribution_by_side.png')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Stone distribution plot saved to {save_path}")
        return save_path
    
    def plot_stone_size_histogram(self, side: Optional[str] = None, 
                                save_path: Optional[str] = None) -> str:
        """
        Create histogram of stone sizes.
        
        Args:
            side: 'left', 'right', or None for both sides
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        # Get size data
        if side == 'left':
            sizes = self.df[self.df['has_left_stone']]['left_stone_size_cm'].dropna()
            title_suffix = ' (Left Side)'
        elif side == 'right':
            sizes = self.df[self.df['has_right_stone']]['right_stone_size_cm'].dropna()
            title_suffix = ' (Right Side)'
        else:
            left_sizes = self.df[self.df['has_left_stone']]['left_stone_size_cm'].dropna()
            right_sizes = self.df[self.df['has_right_stone']]['right_stone_size_cm'].dropna()
            sizes = pd.concat([left_sizes, right_sizes])
            title_suffix = ' (Both Sides)'
        
        if len(sizes) == 0:
            logger.warning(f"No stone size data available for {side or 'both sides'}")
            return ""
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        n, bins, patches = ax.hist(sizes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add statistics
        mean_size = sizes.mean()
        median_size = sizes.median()
        
        ax.axvline(mean_size, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_size:.2f} cm')
        ax.axvline(median_size, color='green', linestyle='--', linewidth=2, label=f'Median: {median_size:.2f} cm')
        
        ax.set_title(f'Stone Size Distribution{title_suffix}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Stone Size (cm)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = f'n = {len(sizes)}\nMin: {sizes.min():.2f} cm\nMax: {sizes.max():.2f} cm'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            side_suffix = f'_{side}' if side else '_both'
            save_path = os.path.join(PLOTS_DIR, f'stone_size_histogram{side_suffix}.png')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Stone size histogram saved to {save_path}")
        return save_path
    
    def plot_kidney_size_scatter(self, save_path: Optional[str] = None) -> str:
        """
        Create scatter plot of kidney sizes.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        # Extract kidney size data
        kidney_data = []
        
        for _, row in self.df.iterrows():
            if pd.notna(row['right_kidney_size_cm']):
                # Parse kidney size (assuming format "L x W x AP cm")
                try:
                    size_str = str(row['right_kidney_size_cm'])
                    if 'x' in size_str:
                        parts = size_str.replace('cm', '').strip().split('x')
                        if len(parts) >= 2:
                            length = float(parts[0].strip())
                            width = float(parts[1].strip())
                            kidney_data.append({
                                'side': 'Right',
                                'length': length,
                                'width': width,
                                'recordid': row['recordid']
                            })
                except (ValueError, IndexError):
                    continue
            
            if pd.notna(row['left_kidney_size_cm']):
                try:
                    size_str = str(row['left_kidney_size_cm'])
                    if 'x' in size_str:
                        parts = size_str.replace('cm', '').strip().split('x')
                        if len(parts) >= 2:
                            length = float(parts[0].strip())
                            width = float(parts[1].strip())
                            kidney_data.append({
                                'side': 'Left',
                                'length': length,
                                'width': width,
                                'recordid': row['recordid']
                            })
                except (ValueError, IndexError):
                    continue
        
        if not kidney_data:
            logger.warning("No kidney size data available")
            return ""
        
        kidney_df = pd.DataFrame(kidney_data)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot by side
        for side in ['Left', 'Right']:
            side_data = kidney_df[kidney_df['side'] == side]
            if len(side_data) > 0:
                ax.scatter(side_data['length'], side_data['width'], 
                          label=f'{side} Kidney (n={len(side_data)})', 
                          alpha=0.7, s=60)
        
        ax.set_title('Kidney Size Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Length (cm)', fontsize=12)
        ax.set_ylabel('Width (cm)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add reference lines for normal kidney size ranges
        ax.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Normal length range')
        ax.axvline(x=12, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=5, color='blue', linestyle='--', alpha=0.5, label='Normal width range')
        ax.axhline(y=6, color='blue', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(PLOTS_DIR, 'kidney_size_scatter.png')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Kidney size scatter plot saved to {save_path}")
        return save_path
    
    def plot_bladder_volume_distribution(self, save_path: Optional[str] = None) -> str:
        """
        Create distribution plot of bladder volumes.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        volumes = self.df['bladder_volume_ml'].dropna()
        
        if len(volumes) == 0:
            logger.warning("No bladder volume data available")
            return ""
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        n, bins, patches = ax.hist(volumes, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        
        # Add statistics
        mean_volume = volumes.mean()
        median_volume = volumes.median()
        
        ax.axvline(mean_volume, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_volume:.1f} ml')
        ax.axvline(median_volume, color='green', linestyle='--', linewidth=2, label=f'Median: {median_volume:.1f} ml')
        
        ax.set_title('Bladder Volume Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Bladder Volume (ml)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = f'n = {len(volumes)}\nMin: {volumes.min():.1f} ml\nMax: {volumes.max():.1f} ml'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(PLOTS_DIR, 'bladder_volume_distribution.png')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Bladder volume distribution plot saved to {save_path}")
        return save_path
    
    def plot_temporal_trends(self, save_path: Optional[str] = None) -> str:
        """
        Create line plot of imaging trends over time.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        temporal_dist = self.query_tools.get_temporal_distribution()
        
        if not temporal_dist:
            logger.warning("No temporal data available")
            return ""
        
        # Convert to DataFrame for easier plotting
        years = sorted([int(year) for year in temporal_dist.keys()])
        counts = [temporal_dist[str(year)] for year in years]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(years, counts, marker='o', linewidth=2, markersize=8, color='darkblue')
        ax.fill_between(years, counts, alpha=0.3, color='lightblue')
        
        ax.set_title('Imaging Volume Trends Over Time', fontsize=16, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Imaging Studies', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on points
        for year, count in zip(years, counts):
            ax.annotate(f'{count}', (year, count), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(PLOTS_DIR, 'temporal_trends.png')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Temporal trends plot saved to {save_path}")
        return save_path
    
    def create_interactive_dashboard(self, save_path: Optional[str] = None) -> str:
        """
        Create an interactive Plotly dashboard.
        
        Args:
            save_path: Optional path to save the HTML file
            
        Returns:
            Path to saved HTML file
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Stone Distribution by Side', 'Stone Size Distribution',
                          'Bladder Volume Distribution', 'Imaging Trends'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Stone distribution by side
        side_dist = self.query_tools.get_side_distribution()
        fig.add_trace(
            go.Bar(x=list(side_dist.keys()), y=list(side_dist.values()),
                  name="Stone Distribution", marker_color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']),
            row=1, col=1
        )
        
        # Stone size distribution
        left_sizes = self.df[self.df['has_left_stone']]['left_stone_size_cm'].dropna()
        right_sizes = self.df[self.df['has_right_stone']]['right_stone_size_cm'].dropna()
        all_sizes = pd.concat([left_sizes, right_sizes])
        
        if len(all_sizes) > 0:
            fig.add_trace(
                go.Histogram(x=all_sizes, name="Stone Sizes", nbinsx=20),
                row=1, col=2
            )
        
        # Bladder volume distribution
        volumes = self.df['bladder_volume_ml'].dropna()
        if len(volumes) > 0:
            fig.add_trace(
                go.Histogram(x=volumes, name="Bladder Volumes", nbinsx=20),
                row=2, col=1
            )
        
        # Temporal trends
        temporal_dist = self.query_tools.get_temporal_distribution()
        if temporal_dist:
            years = sorted([int(year) for year in temporal_dist.keys()])
            counts = [temporal_dist[str(year)] for year in years]
            fig.add_trace(
                go.Scatter(x=years, y=counts, mode='lines+markers', name="Imaging Trends"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Radiology Data Dashboard",
            title_x=0.5,
            showlegend=False,
            height=800
        )
        
        # Save as HTML
        if save_path is None:
            save_path = os.path.join(PLOTS_DIR, 'interactive_dashboard.html')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        
        logger.info(f"Interactive dashboard saved to {save_path}")
        return save_path
    
    def create_all_plots(self) -> Dict[str, str]:
        """
        Create all available plots.
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        plots = {}
        
        try:
            plots['stone_distribution'] = self.plot_stone_distribution_by_side()
        except Exception as e:
            logger.error(f"Error creating stone distribution plot: {e}")
        
        try:
            plots['stone_size_histogram'] = self.plot_stone_size_histogram()
        except Exception as e:
            logger.error(f"Error creating stone size histogram: {e}")
        
        try:
            plots['kidney_size_scatter'] = self.plot_kidney_size_scatter()
        except Exception as e:
            logger.error(f"Error creating kidney size scatter plot: {e}")
        
        try:
            plots['bladder_volume'] = self.plot_bladder_volume_distribution()
        except Exception as e:
            logger.error(f"Error creating bladder volume plot: {e}")
        
        try:
            plots['temporal_trends'] = self.plot_temporal_trends()
        except Exception as e:
            logger.error(f"Error creating temporal trends plot: {e}")
        
        try:
            plots['interactive_dashboard'] = self.create_interactive_dashboard()
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
        
        return plots



