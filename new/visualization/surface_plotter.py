"""
3D Surface visualization tools for SABR volatility surface models.

This module provides comprehensive 3D surface plotting capabilities including:
- 3D volatility surface visualization
- Error heatmap overlays showing prediction accuracy
- Surface difference plots (predicted vs actual)
- Multiple surface comparisons in single plot
- Interactive 3D plots with plotly
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


@dataclass
class SurfaceData:
    """Container for volatility surface data."""
    strikes: np.ndarray
    maturities: np.ndarray
    volatilities: np.ndarray  # 2D array (n_maturities, n_strikes)
    model_name: str
    sabr_params: SABRParams
    forward_price: float
    errors: Optional[np.ndarray] = None  # Error relative to reference surface
    confidence_intervals: Optional[Dict[str, np.ndarray]] = None


@dataclass
class SurfacePlotConfig:
    """Configuration for 3D surface plotting."""
    figure_size: Tuple[int, int] = (12, 10)
    dpi: int = 100
    style: str = 'seaborn-v0_8'
    colormap: str = 'viridis'
    error_colormap: str = 'RdBu_r'
    alpha: float = 0.8
    wireframe: bool = False
    surface_count: int = 50  # Resolution for surface mesh
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    colorbar: bool = True
    grid: bool = True


class SurfacePlotter:
    """
    Comprehensive 3D volatility surface plotting and visualization tool.
    
    Supports both static matplotlib 3D plots and interactive plotly visualizations
    for analyzing volatility surfaces, prediction errors, and model comparisons.
    """
    
    def __init__(
        self,
        config: Optional[SurfacePlotConfig] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize surface plotter.
        
        Args:
            config: Plotting configuration
            output_dir: Directory to save plots
        """
        self.config = config or SurfacePlotConfig()
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use(self.config.style)
    
    def plot_3d_surface(
        self,
        surface_data: SurfaceData,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_wireframe: bool = False,
        moneyness_axis: bool = True,
        elevation: float = 30,
        azimuth: float = 45
    ) -> plt.Figure:
        """
        Create static 3D matplotlib surface plot.
        
        Args:
            surface_data: SurfaceData object containing surface information
            title: Plot title (auto-generated if None)
            save_path: Path to save plot (optional)
            show_wireframe: Whether to show wireframe overlay
            moneyness_axis: Whether to use moneyness (K/F) instead of strikes
            elevation: Viewing elevation angle
            azimuth: Viewing azimuth angle
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare coordinate grids
        if moneyness_axis:
            X = surface_data.strikes / surface_data.forward_price
            x_label = 'Moneyness (K/F)'
        else:
            X = surface_data.strikes
            x_label = 'Strike'
        
        Y = surface_data.maturities
        
        # Create meshgrid for surface plotting
        X_mesh, Y_mesh = np.meshgrid(X, Y)
        Z_mesh = surface_data.volatilities
        
        # Generate title if not provided
        if title is None:
            title = (f"3D Volatility Surface - {surface_data.model_name}\n"
                    f"F={surface_data.forward_price:.1f}, "
                    f"α={surface_data.sabr_params.alpha:.3f}, β={surface_data.sabr_params.beta:.2f}, "
                    f"ν={surface_data.sabr_params.nu:.3f}, ρ={surface_data.sabr_params.rho:.2f}")
        
        # Plot surface
        surface = ax.plot_surface(
            X_mesh, Y_mesh, Z_mesh,
            cmap=self.config.colormap,
            alpha=self.config.alpha,
            linewidth=0.1 if show_wireframe else 0,
            antialiased=True,
            rcount=self.config.surface_count,
            ccount=self.config.surface_count
        )
        
        # Add wireframe if requested
        if show_wireframe:
            ax.plot_wireframe(
                X_mesh, Y_mesh, Z_mesh,
                color='black',
                alpha=0.3,
                linewidth=0.5
            )
        
        # Set labels and title
        ax.set_xlabel(x_label, fontsize=self.config.label_fontsize)
        ax.set_ylabel('Maturity (years)', fontsize=self.config.label_fontsize)
        ax.set_zlabel('Implied Volatility', fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize, pad=20)
        
        # Set viewing angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Add colorbar
        if self.config.colorbar:
            cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=20)
            cbar.set_label('Implied Volatility', fontsize=self.config.label_fontsize)
            cbar.ax.tick_params(labelsize=self.config.tick_fontsize)
        
        # Format ticks
        ax.tick_params(labelsize=self.config.tick_fontsize)
        
        # Add grid
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            if self.output_dir:
                save_path = self.output_dir / save_path
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_interactive_3d_surface(
        self,
        surface_data: SurfaceData,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        moneyness_axis: bool = True,
        show_contours: bool = True
    ) -> go.Figure:
        """
        Create interactive 3D plotly surface plot.
        
        Args:
            surface_data: SurfaceData object containing surface information
            title: Plot title (auto-generated if None)
            save_path: Path to save HTML plot (optional)
            moneyness_axis: Whether to use moneyness (K/F) instead of strikes
            show_contours: Whether to show contour projections
            
        Returns:
            Plotly figure object
        """
        # Prepare coordinate grids
        if moneyness_axis:
            x_values = surface_data.strikes / surface_data.forward_price
            x_label = 'Moneyness (K/F)'
        else:
            x_values = surface_data.strikes
            x_label = 'Strike'
        
        y_values = surface_data.maturities
        z_values = surface_data.volatilities
        
        # Generate title if not provided
        if title is None:
            title = (f"Interactive 3D Volatility Surface - {surface_data.model_name}<br>"
                    f"F={surface_data.forward_price:.1f}, "
                    f"α={surface_data.sabr_params.alpha:.3f}, β={surface_data.sabr_params.beta:.2f}, "
                    f"ν={surface_data.sabr_params.nu:.3f}, ρ={surface_data.sabr_params.rho:.2f}")
        
        # Create surface trace
        surface_trace = go.Surface(
            x=x_values,
            y=y_values,
            z=z_values,
            colorscale=self.config.colormap,
            opacity=self.config.alpha,
            name=surface_data.model_name,
            hovertemplate=(
                f"<b>{surface_data.model_name}</b><br>"
                f"{x_label}: %{{x:.3f}}<br>"
                "Maturity: %{y:.2f}y<br>"
                "Volatility: %{z:.4f}<br>"
                "<extra></extra>"
            )
        )
        
        fig = go.Figure(data=[surface_trace])
        
        # Add contour projections if requested
        if show_contours:
            # Contours on xy plane (bottom)
            fig.add_trace(
                go.Contour(
                    x=x_values,
                    y=y_values,
                    z=z_values,
                    colorscale=self.config.colormap,
                    opacity=0.3,
                    showscale=False,
                    contours=dict(
                        showlines=True,
                        coloring='lines'
                    ),
                    name='Contours'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_label,
                yaxis_title='Maturity (years)',
                zaxis_title='Implied Volatility',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=700,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            if self.output_dir:
                save_path = self.output_dir / save_path
            fig.write_html(save_path)
        
        return fig
    
    def plot_error_heatmap(
        self,
        surface_data: SurfaceData,
        reference_surface: SurfaceData,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        error_type: str = 'absolute',
        moneyness_axis: bool = True
    ) -> plt.Figure:
        """
        Create error heatmap showing prediction accuracy.
        
        Args:
            surface_data: Predicted surface data
            reference_surface: Reference (true) surface data
            title: Plot title (auto-generated if None)
            save_path: Path to save plot (optional)
            error_type: Type of error ('absolute', 'relative', 'percentage')
            moneyness_axis: Whether to use moneyness axis
            
        Returns:
            Matplotlib figure object
        """
        # Calculate errors
        if surface_data.volatilities.shape != reference_surface.volatilities.shape:
            raise ValueError("Surface shapes must match for error calculation")
        
        if error_type == 'absolute':
            errors = surface_data.volatilities - reference_surface.volatilities
            error_label = 'Absolute Error'
            fmt = '.4f'
        elif error_type == 'relative':
            errors = (surface_data.volatilities - reference_surface.volatilities) / reference_surface.volatilities
            error_label = 'Relative Error'
            fmt = '.3f'
        elif error_type == 'percentage':
            errors = ((surface_data.volatilities - reference_surface.volatilities) / reference_surface.volatilities) * 100
            error_label = 'Percentage Error (%)'
            fmt = '.2f'
        else:
            raise ValueError("error_type must be 'absolute', 'relative', or 'percentage'")
        
        # Prepare coordinate labels
        if moneyness_axis:
            x_labels = [f"{k/surface_data.forward_price:.3f}" for k in surface_data.strikes]
            x_label = 'Moneyness (K/F)'
        else:
            x_labels = [f"{k:.1f}" for k in surface_data.strikes]
            x_label = 'Strike'
        
        y_labels = [f"{t:.2f}" for t in surface_data.maturities]
        
        # Generate title if not provided
        if title is None:
            title = (f"Error Heatmap: {surface_data.model_name} vs {reference_surface.model_name}\n"
                    f"F={surface_data.forward_price:.1f}, "
                    f"α={surface_data.sabr_params.alpha:.3f}, β={surface_data.sabr_params.beta:.2f}")
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Determine color scale limits for symmetric colormap
        if error_type in ['relative', 'percentage']:
            vmax = np.max(np.abs(errors))
            vmin = -vmax
        else:
            vmax = np.max(errors)
            vmin = np.min(errors)
            if vmin * vmax < 0:  # If errors span zero, make symmetric
                vmax = max(abs(vmin), abs(vmax))
                vmin = -vmax
        
        # Create heatmap
        im = ax.imshow(
            errors,
            cmap=self.config.error_colormap,
            aspect='auto',
            vmin=vmin,
            vmax=vmax,
            origin='lower'
        )
        
        # Set ticks and labels
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        
        ax.set_xlabel(x_label, fontsize=self.config.label_fontsize)
        ax.set_ylabel('Maturity (years)', fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        
        # Add colorbar
        if self.config.colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(error_label, fontsize=self.config.label_fontsize)
            cbar.ax.tick_params(labelsize=self.config.tick_fontsize)
        
        # Add text annotations for error values
        if errors.size <= 100:  # Only add text for small grids
            for i in range(errors.shape[0]):
                for j in range(errors.shape[1]):
                    text_color = 'white' if abs(errors[i, j]) > 0.5 * vmax else 'black'
                    ax.text(j, i, f'{errors[i, j]:{fmt}}',
                           ha="center", va="center", color=text_color, fontsize=8)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            if self.output_dir:
                save_path = self.output_dir / save_path
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_surface_difference(
        self,
        surface_data: SurfaceData,
        reference_surface: SurfaceData,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        moneyness_axis: bool = True,
        interactive: bool = False
    ) -> Union[plt.Figure, go.Figure]:
        """
        Create 3D surface difference plot (predicted vs actual).
        
        Args:
            surface_data: Predicted surface data
            reference_surface: Reference (true) surface data
            title: Plot title (auto-generated if None)
            save_path: Path to save plot (optional)
            moneyness_axis: Whether to use moneyness axis
            interactive: Whether to create interactive plotly plot
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        # Calculate difference surface
        if surface_data.volatilities.shape != reference_surface.volatilities.shape:
            raise ValueError("Surface shapes must match for difference calculation")
        
        difference = surface_data.volatilities - reference_surface.volatilities
        
        # Create difference surface data
        diff_surface_data = SurfaceData(
            strikes=surface_data.strikes,
            maturities=surface_data.maturities,
            volatilities=difference,
            model_name=f"{surface_data.model_name} - {reference_surface.model_name}",
            sabr_params=surface_data.sabr_params,
            forward_price=surface_data.forward_price
        )
        
        if interactive:
            return self._plot_interactive_surface_difference(
                diff_surface_data, title, save_path, moneyness_axis
            )
        else:
            return self._plot_static_surface_difference(
                diff_surface_data, title, save_path, moneyness_axis
            )
    
    def _plot_static_surface_difference(
        self,
        diff_surface_data: SurfaceData,
        title: Optional[str],
        save_path: Optional[str],
        moneyness_axis: bool
    ) -> plt.Figure:
        """Create static 3D difference surface plot."""
        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare coordinate grids
        if moneyness_axis:
            X = diff_surface_data.strikes / diff_surface_data.forward_price
            x_label = 'Moneyness (K/F)'
        else:
            X = diff_surface_data.strikes
            x_label = 'Strike'
        
        Y = diff_surface_data.maturities
        X_mesh, Y_mesh = np.meshgrid(X, Y)
        Z_mesh = diff_surface_data.volatilities
        
        # Generate title if not provided
        if title is None:
            title = (f"Surface Difference: {diff_surface_data.model_name}\n"
                    f"F={diff_surface_data.forward_price:.1f}")
        
        # Determine color scale for symmetric difference plot
        vmax = np.max(np.abs(Z_mesh))
        vmin = -vmax
        
        # Plot surface with diverging colormap
        surface = ax.plot_surface(
            X_mesh, Y_mesh, Z_mesh,
            cmap=self.config.error_colormap,
            alpha=self.config.alpha,
            linewidth=0,
            antialiased=True,
            vmin=vmin,
            vmax=vmax,
            rcount=self.config.surface_count,
            ccount=self.config.surface_count
        )
        
        # Add zero plane for reference
        ax.plot_surface(
            X_mesh, Y_mesh, np.zeros_like(Z_mesh),
            alpha=0.1,
            color='gray'
        )
        
        # Set labels and title
        ax.set_xlabel(x_label, fontsize=self.config.label_fontsize)
        ax.set_ylabel('Maturity (years)', fontsize=self.config.label_fontsize)
        ax.set_zlabel('Volatility Difference', fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize, pad=20)
        
        # Add colorbar
        if self.config.colorbar:
            cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=20)
            cbar.set_label('Volatility Difference', fontsize=self.config.label_fontsize)
            cbar.ax.tick_params(labelsize=self.config.tick_fontsize)
        
        # Format ticks
        ax.tick_params(labelsize=self.config.tick_fontsize)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            if self.output_dir:
                save_path = self.output_dir / save_path
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_interactive_surface_difference(
        self,
        diff_surface_data: SurfaceData,
        title: Optional[str],
        save_path: Optional[str],
        moneyness_axis: bool
    ) -> go.Figure:
        """Create interactive 3D difference surface plot."""
        # Prepare coordinate grids
        if moneyness_axis:
            x_values = diff_surface_data.strikes / diff_surface_data.forward_price
            x_label = 'Moneyness (K/F)'
        else:
            x_values = diff_surface_data.strikes
            x_label = 'Strike'
        
        y_values = diff_surface_data.maturities
        z_values = diff_surface_data.volatilities
        
        # Generate title if not provided
        if title is None:
            title = f"Interactive Surface Difference: {diff_surface_data.model_name}"
        
        # Determine color scale for symmetric difference plot
        vmax = np.max(np.abs(z_values))
        vmin = -vmax
        
        # Create surface trace
        surface_trace = go.Surface(
            x=x_values,
            y=y_values,
            z=z_values,
            colorscale=self.config.error_colormap,
            opacity=self.config.alpha,
            name=diff_surface_data.model_name,
            cmin=vmin,
            cmax=vmax,
            hovertemplate=(
                f"<b>{diff_surface_data.model_name}</b><br>"
                f"{x_label}: %{{x:.3f}}<br>"
                "Maturity: %{y:.2f}y<br>"
                "Difference: %{z:.6f}<br>"
                "<extra></extra>"
            )
        )
        
        fig = go.Figure(data=[surface_trace])
        
        # Add zero plane for reference
        zero_plane = go.Surface(
            x=x_values,
            y=y_values,
            z=np.zeros_like(z_values),
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'gray'], [1, 'gray']],
            name='Zero Reference'
        )
        fig.add_trace(zero_plane)
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_label,
                yaxis_title='Maturity (years)',
                zaxis_title='Volatility Difference',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=700,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            if self.output_dir:
                save_path = self.output_dir / save_path
            fig.write_html(save_path)
        
        return fig
    
    def plot_multiple_surfaces_comparison(
        self,
        surface_data_list: List[SurfaceData],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        layout: str = 'grid',
        moneyness_axis: bool = True,
        interactive: bool = False
    ) -> Union[plt.Figure, go.Figure]:
        """
        Create multiple surface comparison plot.
        
        Args:
            surface_data_list: List of SurfaceData objects to compare
            title: Plot title (auto-generated if None)
            save_path: Path to save plot (optional)
            layout: Layout type ('grid', 'overlay', 'side_by_side')
            moneyness_axis: Whether to use moneyness axis
            interactive: Whether to create interactive plotly plot
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if not surface_data_list:
            raise ValueError("surface_data_list cannot be empty")
        
        if interactive:
            return self._plot_interactive_multiple_surfaces(
                surface_data_list, title, save_path, layout, moneyness_axis
            )
        else:
            return self._plot_static_multiple_surfaces(
                surface_data_list, title, save_path, layout, moneyness_axis
            )
    
    def _plot_static_multiple_surfaces(
        self,
        surface_data_list: List[SurfaceData],
        title: Optional[str],
        save_path: Optional[str],
        layout: str,
        moneyness_axis: bool
    ) -> plt.Figure:
        """Create static multiple surface comparison plot."""
        n_surfaces = len(surface_data_list)
        
        if layout == 'overlay':
            # Single plot with all surfaces overlaid
            fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            colors = plt.cm.Set1(np.linspace(0, 1, n_surfaces))
            
            for i, surface_data in enumerate(surface_data_list):
                if moneyness_axis:
                    X = surface_data.strikes / surface_data.forward_price
                    x_label = 'Moneyness (K/F)'
                else:
                    X = surface_data.strikes
                    x_label = 'Strike'
                
                Y = surface_data.maturities
                X_mesh, Y_mesh = np.meshgrid(X, Y)
                Z_mesh = surface_data.volatilities
                
                ax.plot_surface(
                    X_mesh, Y_mesh, Z_mesh,
                    alpha=0.7,
                    color=colors[i],
                    label=surface_data.model_name,
                    rcount=20,
                    ccount=20
                )
            
            ax.set_xlabel(x_label, fontsize=self.config.label_fontsize)
            ax.set_ylabel('Maturity (years)', fontsize=self.config.label_fontsize)
            ax.set_zlabel('Implied Volatility', fontsize=self.config.label_fontsize)
            
            if title is None:
                title = "Multiple Surface Comparison"
            ax.set_title(title, fontsize=self.config.title_fontsize, pad=20)
            
        elif layout == 'grid':
            # Grid layout with subplots
            n_cols = min(3, n_surfaces)
            n_rows = (n_surfaces + n_cols - 1) // n_cols
            
            fig = plt.figure(figsize=(n_cols * 5, n_rows * 4), dpi=self.config.dpi)
            
            for i, surface_data in enumerate(surface_data_list):
                ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
                
                if moneyness_axis:
                    X = surface_data.strikes / surface_data.forward_price
                    x_label = 'Moneyness (K/F)'
                else:
                    X = surface_data.strikes
                    x_label = 'Strike'
                
                Y = surface_data.maturities
                X_mesh, Y_mesh = np.meshgrid(X, Y)
                Z_mesh = surface_data.volatilities
                
                surface = ax.plot_surface(
                    X_mesh, Y_mesh, Z_mesh,
                    cmap=self.config.colormap,
                    alpha=self.config.alpha,
                    rcount=30,
                    ccount=30
                )
                
                ax.set_xlabel(x_label, fontsize=10)
                ax.set_ylabel('Maturity', fontsize=10)
                ax.set_zlabel('Volatility', fontsize=10)
                ax.set_title(surface_data.model_name, fontsize=12)
                ax.tick_params(labelsize=8)
            
            if title:
                fig.suptitle(title, fontsize=self.config.title_fontsize)
        
        else:
            raise ValueError("layout must be 'grid' or 'overlay'")
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            if self.output_dir:
                save_path = self.output_dir / save_path
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_interactive_multiple_surfaces(
        self,
        surface_data_list: List[SurfaceData],
        title: Optional[str],
        save_path: Optional[str],
        layout: str,
        moneyness_axis: bool
    ) -> go.Figure:
        """Create interactive multiple surface comparison plot."""
        if layout == 'overlay':
            # Single plot with all surfaces overlaid
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set1[:len(surface_data_list)]
            
            for i, surface_data in enumerate(surface_data_list):
                if moneyness_axis:
                    x_values = surface_data.strikes / surface_data.forward_price
                    x_label = 'Moneyness (K/F)'
                else:
                    x_values = surface_data.strikes
                    x_label = 'Strike'
                
                y_values = surface_data.maturities
                z_values = surface_data.volatilities
                
                surface_trace = go.Surface(
                    x=x_values,
                    y=y_values,
                    z=z_values,
                    opacity=0.8,
                    name=surface_data.model_name,
                    colorscale=[[0, colors[i]], [1, colors[i]]],
                    showscale=False,
                    hovertemplate=(
                        f"<b>{surface_data.model_name}</b><br>"
                        f"{x_label}: %{{x:.3f}}<br>"
                        "Maturity: %{y:.2f}y<br>"
                        "Volatility: %{z:.4f}<br>"
                        "<extra></extra>"
                    )
                )
                fig.add_trace(surface_trace)
            
            fig.update_layout(
                title=title or "Interactive Multiple Surface Comparison",
                scene=dict(
                    xaxis_title=x_label,
                    yaxis_title='Maturity (years)',
                    zaxis_title='Implied Volatility',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=900,
                height=700
            )
        
        elif layout == 'grid':
            # Grid layout with subplots
            n_surfaces = len(surface_data_list)
            n_cols = min(2, n_surfaces)
            n_rows = (n_surfaces + n_cols - 1) // n_cols
            
            subplot_titles = [surface.model_name for surface in surface_data_list]
            
            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                specs=[[{'type': 'surface'} for _ in range(n_cols)] for _ in range(n_rows)],
                subplot_titles=subplot_titles,
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            for i, surface_data in enumerate(surface_data_list):
                row = i // n_cols + 1
                col = i % n_cols + 1
                
                if moneyness_axis:
                    x_values = surface_data.strikes / surface_data.forward_price
                    x_label = 'Moneyness (K/F)'
                else:
                    x_values = surface_data.strikes
                    x_label = 'Strike'
                
                y_values = surface_data.maturities
                z_values = surface_data.volatilities
                
                surface_trace = go.Surface(
                    x=x_values,
                    y=y_values,
                    z=z_values,
                    colorscale=self.config.colormap,
                    showscale=i == 0,  # Only show colorbar for first surface
                    name=surface_data.model_name
                )
                
                fig.add_trace(surface_trace, row=row, col=col)
            
            # Update scene properties for all subplots
            for i in range(1, n_rows * n_cols + 1):
                scene_name = f'scene{i}' if i > 1 else 'scene'
                fig.update_layout(**{
                    scene_name: dict(
                        xaxis_title=x_label,
                        yaxis_title='Maturity',
                        zaxis_title='Volatility',
                        camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
                    )
                })
            
            fig.update_layout(
                title=title or "Interactive Multiple Surface Grid Comparison",
                height=400 * n_rows,
                width=500 * n_cols
            )
        
        else:
            raise ValueError("layout must be 'grid' or 'overlay'")
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            if self.output_dir:
                save_path = self.output_dir / save_path
            fig.write_html(save_path)
        
        return fig


def create_surface_data(
    strikes: np.ndarray,
    maturities: np.ndarray,
    volatilities: np.ndarray,
    model_name: str,
    sabr_params: SABRParams,
    forward_price: float,
    errors: Optional[np.ndarray] = None
) -> SurfaceData:
    """
    Factory function to create SurfaceData object.
    
    Args:
        strikes: Strike prices array
        maturities: Maturity times array
        volatilities: 2D volatility surface (n_maturities, n_strikes)
        model_name: Name of the model
        sabr_params: SABR parameters
        forward_price: Forward price
        errors: Optional error values
        
    Returns:
        SurfaceData object
    """
    return SurfaceData(
        strikes=strikes,
        maturities=maturities,
        volatilities=volatilities,
        model_name=model_name,
        sabr_params=sabr_params,
        forward_price=forward_price,
        errors=errors
    )