"""
Volatility smile visualization tools for SABR surface models.

This module provides comprehensive smile plotting capabilities including:
- Multi-model comparison plots (HF, LF, baseline, MDA-CNN)
- Error visualization with confidence intervals
- Interactive plots for detailed analysis
- Support for multiple parameter sets and market conditions
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import warnings

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data_generation.sabr_params import SABRParams, GridConfig
from evaluation.metrics import SurfaceEvaluator, StatisticalTester


@dataclass
class SmileData:
    """Container for volatility smile data."""
    strikes: np.ndarray
    volatilities: np.ndarray
    model_name: str
    maturity: float
    sabr_params: SABRParams
    forward_price: float
    errors: Optional[np.ndarray] = None
    confidence_intervals: Optional[Dict[str, np.ndarray]] = None


@dataclass
class SmilePlotConfig:
    """Configuration for smile plotting."""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    style: str = 'seaborn-v0_8'
    color_palette: str = 'Set1'
    line_width: float = 2.0
    marker_size: float = 6.0
    alpha: float = 0.8
    grid: bool = True
    legend: bool = True
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10


class SmilePlotter:
    """
    Comprehensive volatility smile plotting and visualization tool.
    
    Supports static matplotlib plots and interactive plotly visualizations
    for comparing multiple models and analyzing prediction errors.
    """
    
    def __init__(
        self,
        config: Optional[SmilePlotConfig] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize smile plotter.
        
        Args:
            config: Plotting configuration
            output_dir: Directory to save plots
        """
        self.config = config or SmilePlotConfig()
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)
        
        # Initialize statistical tester for confidence intervals
        self.statistical_tester = StatisticalTester()
    
    def plot_smile_comparison(
        self,
        smile_data_list: List[SmileData],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_errors: bool = True,
        show_confidence_intervals: bool = True,
        moneyness_axis: bool = True
    ) -> plt.Figure:
        """
        Create static matplotlib plot comparing multiple volatility smiles.
        
        Args:
            smile_data_list: List of SmileData objects to compare
            title: Plot title (auto-generated if None)
            save_path: Path to save plot (optional)
            show_errors: Whether to show error bars
            show_confidence_intervals: Whether to show confidence intervals
            moneyness_axis: Whether to use moneyness (K/F) instead of strikes
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figure_size, 
                                       dpi=self.config.dpi, height_ratios=[3, 1])
        
        # Handle empty data
        if not smile_data_list:
            # Create empty plot
            ax1.set_title("No Data Available", fontsize=self.config.title_fontsize)
            ax1.set_xlabel('Moneyness (K/F)' if moneyness_axis else 'Strike', fontsize=self.config.label_fontsize)
            ax1.set_ylabel('Implied Volatility', fontsize=self.config.label_fontsize)
            ax2.set_visible(False)
            plt.tight_layout()
            return fig
        
        # Get reference data for title generation
        ref_data = smile_data_list[0]
        
        # Generate title if not provided
        if title is None:
            title = (f"Volatility Smile Comparison\n"
                    f"T={ref_data.maturity:.2f}y, F={ref_data.forward_price:.1f}, "
                    f"α={ref_data.sabr_params.alpha:.3f}, β={ref_data.sabr_params.beta:.2f}, "
                    f"ν={ref_data.sabr_params.nu:.3f}, ρ={ref_data.sabr_params.rho:.2f}")
        
        # Define colors for different model types
        color_map = {
            'hf': '#1f77b4',      # Blue for high-fidelity
            'lf': '#ff7f0e',      # Orange for low-fidelity  
            'hagan': '#ff7f0e',   # Orange for Hagan (LF)
            'mc': '#1f77b4',      # Blue for Monte Carlo (HF)
            'baseline': '#2ca02c', # Green for baseline
            'mda_cnn': '#d62728', # Red for MDA-CNN
            'mda-cnn': '#d62728', # Red for MDA-CNN (alternative naming)
            'mlp': '#9467bd',     # Purple for MLP
            'cnn': '#8c564b'      # Brown for CNN-only
        }
        
        # Find reference model (typically HF/MC) for error calculation
        reference_data = None
        for data in smile_data_list:
            if any(ref_name in data.model_name.lower() for ref_name in ['hf', 'mc', 'monte']):
                reference_data = data
                break
        
        # Plot main volatility smiles
        for data in smile_data_list:
            # Determine x-axis values
            if moneyness_axis:
                x_values = data.strikes / data.forward_price
                x_label = 'Moneyness (K/F)'
            else:
                x_values = data.strikes
                x_label = 'Strike'
            
            # Get color for this model
            model_key = data.model_name.lower().replace(' ', '_').replace('-', '_')
            color = color_map.get(model_key, None)
            
            # Plot volatility smile
            line_style = '-' if 'hf' in model_key or 'mc' in model_key else '--'
            marker = 'o' if len(data.strikes) <= 20 else None
            
            ax1.plot(x_values, data.volatilities, 
                    color=color, linewidth=self.config.line_width,
                    linestyle=line_style, marker=marker, 
                    markersize=self.config.marker_size,
                    alpha=self.config.alpha, label=data.model_name)
            
            # Add confidence intervals if available
            if show_confidence_intervals and data.confidence_intervals:
                if 'lower' in data.confidence_intervals and 'upper' in data.confidence_intervals:
                    ax1.fill_between(x_values, 
                                   data.confidence_intervals['lower'],
                                   data.confidence_intervals['upper'],
                                   color=color, alpha=0.2)
        
        # Plot error comparison in bottom subplot
        if reference_data and show_errors:
            for data in smile_data_list:
                if data.model_name == reference_data.model_name:
                    continue  # Skip reference model
                
                # Calculate errors relative to reference
                if len(data.volatilities) == len(reference_data.volatilities):
                    errors = data.volatilities - reference_data.volatilities
                    
                    if moneyness_axis:
                        x_values = data.strikes / data.forward_price
                    else:
                        x_values = data.strikes
                    
                    model_key = data.model_name.lower().replace(' ', '_').replace('-', '_')
                    color = color_map.get(model_key, None)
                    
                    ax2.plot(x_values, errors, color=color, 
                           linewidth=self.config.line_width,
                           marker='o' if len(data.strikes) <= 20 else None,
                           markersize=self.config.marker_size * 0.8,
                           alpha=self.config.alpha, label=f'{data.model_name} - {reference_data.model_name}')
            
            # Add zero line for reference
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_ylabel('Volatility Error', fontsize=self.config.label_fontsize)
            ax2.legend(fontsize=self.config.tick_fontsize)
            ax2.grid(self.config.grid, alpha=0.3)
        else:
            # Hide bottom subplot if no errors to show
            ax2.set_visible(False)
            fig.subplots_adjust(hspace=0.1)
        
        # Format main plot
        ax1.set_title(title, fontsize=self.config.title_fontsize)
        ax1.set_xlabel(x_label, fontsize=self.config.label_fontsize)
        ax1.set_ylabel('Implied Volatility', fontsize=self.config.label_fontsize)
        
        if self.config.legend:
            ax1.legend(fontsize=self.config.tick_fontsize)
        
        ax1.grid(self.config.grid, alpha=0.3)
        ax1.tick_params(labelsize=self.config.tick_fontsize)
        
        if show_errors and reference_data:
            ax2.set_xlabel(x_label, fontsize=self.config.label_fontsize)
            ax2.tick_params(labelsize=self.config.tick_fontsize)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            if self.output_dir:
                save_path = self.output_dir / save_path
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_interactive_smile_comparison(
        self,
        smile_data_list: List[SmileData],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        moneyness_axis: bool = True,
        show_errors: bool = True
    ) -> go.Figure:
        """
        Create interactive plotly plot comparing multiple volatility smiles.
        
        Args:
            smile_data_list: List of SmileData objects to compare
            title: Plot title (auto-generated if None)
            save_path: Path to save HTML plot (optional)
            moneyness_axis: Whether to use moneyness (K/F) instead of strikes
            show_errors: Whether to include error subplot
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        if show_errors:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=('Volatility Smiles', 'Prediction Errors'),
                vertical_spacing=0.1
            )
        else:
            fig = go.Figure()
        
        # Handle empty data
        if not smile_data_list:
            fig.update_layout(
                title="No Data Available",
                xaxis_title='Moneyness (K/F)' if moneyness_axis else 'Strike',
                yaxis_title='Implied Volatility'
            )
            return fig
        
        # Get reference data for title generation
        ref_data = smile_data_list[0]
        
        # Generate title if not provided
        if title is None:
            title = (f"Interactive Volatility Smile Comparison<br>"
                    f"T={ref_data.maturity:.2f}y, F={ref_data.forward_price:.1f}, "
                    f"α={ref_data.sabr_params.alpha:.3f}, β={ref_data.sabr_params.beta:.2f}, "
                    f"ν={ref_data.sabr_params.nu:.3f}, ρ={ref_data.sabr_params.rho:.2f}")
        
        # Define colors for different model types
        color_map = {
            'hf': '#1f77b4', 'mc': '#1f77b4', 'monte': '#1f77b4',
            'lf': '#ff7f0e', 'hagan': '#ff7f0e',
            'baseline': '#2ca02c', 'mlp': '#9467bd',
            'mda_cnn': '#d62728', 'mda-cnn': '#d62728',
            'cnn': '#8c564b'
        }
        
        # Find reference model for error calculation
        reference_data = None
        for data in smile_data_list:
            if any(ref_name in data.model_name.lower() for ref_name in ['hf', 'mc', 'monte']):
                reference_data = data
                break
        
        # Plot volatility smiles
        for i, data in enumerate(smile_data_list):
            # Determine x-axis values
            if moneyness_axis:
                x_values = data.strikes / data.forward_price
                x_label = 'Moneyness (K/F)'
            else:
                x_values = data.strikes
                x_label = 'Strike'
            
            # Get color for this model
            model_key = data.model_name.lower().replace(' ', '_').replace('-', '_')
            color = None
            for key in color_map:
                if key in model_key:
                    color = color_map[key]
                    break
            if color is None:
                color = px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
            
            # Determine line style
            line_dash = 'solid' if any(ref in model_key for ref in ['hf', 'mc', 'monte']) else 'dash'
            
            # Create hover text
            hover_text = [
                f"Model: {data.model_name}<br>"
                f"Strike: {strike:.2f}<br>"
                f"Moneyness: {strike/data.forward_price:.3f}<br>"
                f"Volatility: {vol:.4f}<br>"
                f"Maturity: {data.maturity:.2f}y"
                for strike, vol in zip(data.strikes, data.volatilities)
            ]
            
            # Add main trace
            trace = go.Scatter(
                x=x_values,
                y=data.volatilities,
                mode='lines+markers',
                name=data.model_name,
                line=dict(color=color, width=3, dash=line_dash),
                marker=dict(size=6, color=color),
                hovertext=hover_text,
                hoverinfo='text'
            )
            
            if show_errors:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)
            
            # Add confidence intervals if available
            if data.confidence_intervals and 'lower' in data.confidence_intervals:
                # Upper bound
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=data.confidence_intervals['upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1 if show_errors else None, col=1 if show_errors else None
                )
                
                # Lower bound with fill
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=data.confidence_intervals['lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.2])}',
                        name=f'{data.model_name} CI',
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1 if show_errors else None, col=1 if show_errors else None
                )
        
        # Plot errors in bottom subplot
        if show_errors and reference_data:
            for data in smile_data_list:
                if data.model_name == reference_data.model_name:
                    continue
                
                if len(data.volatilities) == len(reference_data.volatilities):
                    errors = data.volatilities - reference_data.volatilities
                    
                    if moneyness_axis:
                        x_values = data.strikes / data.forward_price
                    else:
                        x_values = data.strikes
                    
                    model_key = data.model_name.lower().replace(' ', '_').replace('-', '_')
                    color = None
                    for key in color_map:
                        if key in model_key:
                            color = color_map[key]
                            break
                    
                    # Create error hover text
                    error_hover_text = [
                        f"Model: {data.model_name}<br>"
                        f"Strike: {strike:.2f}<br>"
                        f"Error: {error:.6f}<br>"
                        f"Relative Error: {error/ref_vol*100:.2f}%"
                        for strike, error, ref_vol in zip(data.strikes, errors, reference_data.volatilities)
                    ]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=errors,
                            mode='lines+markers',
                            name=f'{data.model_name} Error',
                            line=dict(color=color, width=2),
                            marker=dict(size=4, color=color),
                            hovertext=error_hover_text,
                            hoverinfo='text'
                        ),
                        row=2, col=1
                    )
            
            # Add zero line
            if moneyness_axis:
                x_range = [min(np.min(d.strikes / d.forward_price) for d in smile_data_list),
                          max(np.max(d.strikes / d.forward_price) for d in smile_data_list)]
            else:
                x_range = [min(np.min(d.strikes) for d in smile_data_list),
                          max(np.max(d.strikes) for d in smile_data_list)]
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[0, 0],
                    mode='lines',
                    line=dict(color='black', width=1, dash='dash'),
                    name='Zero Error',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800 if show_errors else 500,
            hovermode='closest',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        # Update axes labels
        if show_errors:
            fig.update_xaxes(title_text=x_label, row=2, col=1)
            fig.update_yaxes(title_text='Implied Volatility', row=1, col=1)
            fig.update_yaxes(title_text='Volatility Error', row=2, col=1)
        else:
            fig.update_xaxes(title_text=x_label)
            fig.update_yaxes(title_text='Implied Volatility')
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            if self.output_dir:
                save_path = self.output_dir / save_path
            fig.write_html(save_path)
        
        return fig
    
    def plot_multi_maturity_smiles(
        self,
        smile_data_dict: Dict[float, List[SmileData]],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        interactive: bool = True,
        moneyness_axis: bool = True
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot volatility smiles for multiple maturities.
        
        Args:
            smile_data_dict: Dictionary mapping maturity -> list of SmileData
            title: Plot title
            save_path: Path to save plot
            interactive: Whether to create interactive plotly plot
            moneyness_axis: Whether to use moneyness axis
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if interactive:
            return self._plot_multi_maturity_interactive(
                smile_data_dict, title, save_path, moneyness_axis
            )
        else:
            return self._plot_multi_maturity_static(
                smile_data_dict, title, save_path, moneyness_axis
            )
    
    def _plot_multi_maturity_static(
        self,
        smile_data_dict: Dict[float, List[SmileData]],
        title: Optional[str],
        save_path: Optional[str],
        moneyness_axis: bool
    ) -> plt.Figure:
        """Create static multi-maturity plot."""
        n_maturities = len(smile_data_dict)
        n_cols = min(3, n_maturities)
        n_rows = (n_maturities + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4),
                                dpi=self.config.dpi)
        
        if n_maturities == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten() if n_maturities > 1 else axes
        
        for i, (maturity, smile_data_list) in enumerate(sorted(smile_data_dict.items())):
            ax = axes_flat[i]
            
            for data in smile_data_list:
                if moneyness_axis:
                    x_values = data.strikes / data.forward_price
                    x_label = 'Moneyness (K/F)'
                else:
                    x_values = data.strikes
                    x_label = 'Strike'
                
                ax.plot(x_values, data.volatilities, 
                       label=data.model_name, linewidth=2, marker='o', markersize=4)
            
            ax.set_title(f'T = {maturity:.2f}y', fontsize=self.config.title_fontsize)
            ax.set_xlabel(x_label, fontsize=self.config.label_fontsize)
            ax.set_ylabel('Implied Volatility', fontsize=self.config.label_fontsize)
            ax.legend(fontsize=self.config.tick_fontsize)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_maturities, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        if title:
            fig.suptitle(title, fontsize=self.config.title_fontsize + 2)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            if self.output_dir:
                save_path = self.output_dir / save_path
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_multi_maturity_interactive(
        self,
        smile_data_dict: Dict[float, List[SmileData]],
        title: Optional[str],
        save_path: Optional[str],
        moneyness_axis: bool
    ) -> go.Figure:
        """Create interactive multi-maturity plot."""
        fig = go.Figure()
        
        # Create dropdown menu for maturity selection
        buttons = []
        
        for maturity, smile_data_list in sorted(smile_data_dict.items()):
            # Add traces for this maturity (initially invisible)
            for data in smile_data_list:
                if moneyness_axis:
                    x_values = data.strikes / data.forward_price
                    x_label = 'Moneyness (K/F)'
                else:
                    x_values = data.strikes
                    x_label = 'Strike'
                
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=data.volatilities,
                        mode='lines+markers',
                        name=data.model_name,
                        visible=False,  # Initially hidden
                        line=dict(width=3),
                        marker=dict(size=6)
                    )
                )
        
        # Make first maturity visible
        first_maturity_size = len(list(smile_data_dict.values())[0])
        for i in range(first_maturity_size):
            fig.data[i].visible = True
        
        # Create dropdown buttons
        for i, (maturity, smile_data_list) in enumerate(sorted(smile_data_dict.items())):
            # Calculate trace indices for this maturity
            start_idx = i * len(smile_data_list)
            end_idx = start_idx + len(smile_data_list)
            
            # Create visibility array
            visibility = [False] * len(fig.data)
            for j in range(start_idx, end_idx):
                visibility[j] = True
            
            buttons.append(
                dict(
                    label=f'T = {maturity:.2f}y',
                    method='update',
                    args=[{'visible': visibility}]
                )
            )
        
        # Update layout with dropdown
        fig.update_layout(
            title=title or 'Volatility Smiles - Multiple Maturities',
            xaxis_title=x_label,
            yaxis_title='Implied Volatility',
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ],
            height=600
        )
        
        if save_path:
            save_path = Path(save_path)
            if self.output_dir:
                save_path = self.output_dir / save_path
            fig.write_html(save_path)
        
        return fig
    
    def compute_smile_confidence_intervals(
        self,
        predictions: np.ndarray,
        strikes: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Compute confidence intervals for volatility smile predictions.
        
        Args:
            predictions: Array of predictions (n_samples, n_strikes)
            strikes: Strike prices
            confidence_level: Confidence level for intervals
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with 'lower' and 'upper' confidence bounds
        """
        if predictions.ndim == 1:
            # Single prediction, no confidence intervals possible
            return {}
        
        # Compute percentiles for confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(predictions, lower_percentile, axis=0)
        upper_bounds = np.percentile(predictions, upper_percentile, axis=0)
        
        return {
            'lower': lower_bounds,
            'upper': upper_bounds
        }


def create_smile_data(
    strikes: np.ndarray,
    volatilities: np.ndarray,
    model_name: str,
    maturity: float,
    sabr_params: SABRParams,
    forward_price: float,
    errors: Optional[np.ndarray] = None,
    confidence_intervals: Optional[Dict[str, np.ndarray]] = None
) -> SmileData:
    """
    Factory function to create SmileData object.
    
    Args:
        strikes: Strike prices
        volatilities: Volatility values
        model_name: Name of the model
        maturity: Time to maturity
        sabr_params: SABR parameters
        forward_price: Forward price
        errors: Optional error values
        confidence_intervals: Optional confidence intervals
        
    Returns:
        SmileData object
    """
    return SmileData(
        strikes=strikes,
        volatilities=volatilities,
        model_name=model_name,
        maturity=maturity,
        sabr_params=sabr_params,
        forward_price=forward_price,
        errors=errors,
        confidence_intervals=confidence_intervals
    )