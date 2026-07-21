"""
Attention Heatmap Visualization Module

This module provides functionality for extracting attention weights from Vision Transformer (ViT)
models and generating visual heatmap overlays. The heatmaps show which image regions the model
focuses on during classification, providing explainability for AI predictions.

Key Components:
    - AttentionHeatmapGenerator: Main class for generating attention visualizations
    - Attention extraction from ViT final layer
    - Spatial reshaping and upsampling to image dimensions
    - Perceptually-uniform colormap application
    - Blending with original images for overlay visualization

Usage:
    generator = AttentionHeatmapGenerator(colormap='viridis', opacity=0.5)
    heatmap = generator.generate_heatmap(attention_weights, original_image, patch_size=16)

Author: PyroWatch Team
Date: 2026-04-16
"""

import logging
from typing import Optional, Dict, Tuple
import numpy as np
import torch
import cv2
from matplotlib import cm
from matplotlib.colors import Colormap

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class AttentionHeatmapGenerator:
    """
    Generates attention heatmap visualizations from ViT model attention weights.
    
    This class extracts attention information from Vision Transformer models and creates
    visual overlays showing which image regions the model focuses on during classification.
    The implementation follows the Attention Rollout methodology for ViT explainability.
    
    Attributes:
        colormap_name (str): Name of the matplotlib colormap to use
        opacity (float): Blending opacity for overlay (0.0-1.0)
        colormap_cache (Dict[str, Colormap]): Cache of colormap objects for performance
    """
    
    def __init__(self, colormap: str = 'viridis', opacity: float = 0.5):
        """
        Initialize the attention heatmap generator.
        
        Args:
            colormap: Perceptually-uniform colormap name ('viridis', 'plasma', 'inferno')
            opacity: Blending opacity for overlay (0.4-0.6 recommended)
            
        Raises:
            ValueError: If colormap name is invalid or opacity is out of range
        """
        # Validate colormap name
        valid_colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        if colormap not in valid_colormaps:
            raise ValueError(
                f"Invalid colormap '{colormap}'. Must be one of {valid_colormaps}"
            )
        
        # Validate opacity range
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(f"Opacity must be in range [0.0, 1.0], got {opacity}")
        
        if not 0.4 <= opacity <= 0.6:
            logger.warning(
                f"Opacity {opacity} is outside recommended range [0.4, 0.6]. "
                "Heatmap may be too transparent or too opaque."
            )
        
        self.colormap_name = colormap
        self.opacity = opacity
        self.colormap_cache: Dict[str, Colormap] = {}
        
        logger.info(
            f"AttentionHeatmapGenerator initialized with colormap='{colormap}', "
            f"opacity={opacity}"
        )

    
    def _aggregate_attention_heads(self, attention: torch.Tensor) -> np.ndarray:
        """
        Aggregate attention weights across all attention heads using mean.
        
        Args:
            attention: Attention tensor with shape (num_heads, seq_len, seq_len)
            
        Returns:
            Aggregated attention with shape (seq_len, seq_len) as numpy array
            
        Raises:
            ValueError: If attention tensor has unexpected shape
        """
        if attention.dim() != 3:
            raise ValueError(
                f"Expected 3D attention tensor (num_heads, seq_len, seq_len), "
                f"got shape {attention.shape}"
            )
        
        # Compute mean across attention heads (dim=0)
        aggregated = attention.mean(dim=0)
        
        # Convert to numpy array
        if aggregated.is_cuda:
            aggregated = aggregated.cpu()
        aggregated_np = aggregated.detach().numpy()
        
        logger.debug(
            f"Aggregated {attention.shape[0]} attention heads: "
            f"{attention.shape} -> {aggregated_np.shape}"
        )
        
        return aggregated_np

    
    def _extract_cls_attention(self, aggregated_attention: np.ndarray) -> np.ndarray:
        """
        Extract CLS token attention to image patches.
        
        The CLS token is the first token in the sequence. This method extracts
        the attention weights from the CLS token to all image patches (excluding
        the CLS token itself).
        
        Args:
            aggregated_attention: Aggregated attention with shape (seq_len, seq_len)
            
        Returns:
            1D array of attention values for image patches only (length: seq_len - 1)
            
        Raises:
            ValueError: If attention array has unexpected shape
        """
        if aggregated_attention.ndim != 2:
            raise ValueError(
                f"Expected 2D attention array (seq_len, seq_len), "
                f"got shape {aggregated_attention.shape}"
            )
        
        if aggregated_attention.shape[0] != aggregated_attention.shape[1]:
            raise ValueError(
                f"Expected square attention matrix, got shape {aggregated_attention.shape}"
            )
        
        # Extract first row (CLS token attention to all tokens)
        cls_attention = aggregated_attention[0, :]
        
        # Remove CLS token attention to itself (first element)
        cls_to_patches = cls_attention[1:]
        
        logger.debug(
            f"Extracted CLS attention: {aggregated_attention.shape} -> "
            f"{cls_to_patches.shape} (removed CLS self-attention)"
        )
        
        return cls_to_patches

    
    def _reshape_to_spatial(self, cls_attention: np.ndarray, patch_size: int) -> np.ndarray:
        """
        Reshape 1D attention array to 2D spatial grid.
        
        Args:
            cls_attention: 1D array of attention values for image patches
            patch_size: Size of ViT patches (typically 16 for ViT-base)
            
        Returns:
            2D spatial attention map with shape (grid_h, grid_w)
            
        Raises:
            ValueError: If attention array cannot be reshaped to square grid
        """
        num_patches = cls_attention.shape[0]
        
        # Calculate grid dimensions (assuming square grid)
        grid_size = int(np.sqrt(num_patches))
        
        # Verify that we have a perfect square
        if grid_size * grid_size != num_patches:
            # Handle non-square grids (rare but possible)
            logger.warning(
                f"Non-square patch grid detected: {num_patches} patches. "
                f"Attempting to reshape to closest square: {grid_size}x{grid_size}"
            )
            # Pad or truncate to make it square
            if grid_size * grid_size < num_patches:
                cls_attention = cls_attention[:grid_size * grid_size]
            else:
                padding = grid_size * grid_size - num_patches
                cls_attention = np.pad(cls_attention, (0, padding), mode='constant')
        
        # Reshape to 2D spatial grid
        spatial_attention = cls_attention.reshape(grid_size, grid_size)
        
        logger.debug(
            f"Reshaped attention to spatial grid: {cls_attention.shape} -> "
            f"{spatial_attention.shape} (patch_size={patch_size})"
        )
        
        return spatial_attention

    
    def _normalize_attention(self, attention: np.ndarray) -> np.ndarray:
        """
        Normalize attention values to [0, 1] range using min-max normalization.
        
        Args:
            attention: 2D attention array
            
        Returns:
            Normalized attention with values in [0, 1] range
        """
        # Handle edge cases
        if attention.size == 0:
            logger.warning("Empty attention array provided for normalization")
            return attention
        
        min_val = attention.min()
        max_val = attention.max()
        
        # Handle all zeros or all same value
        if max_val == min_val:
            logger.debug(
                f"Attention has constant value {min_val}. Returning zeros."
            )
            return np.zeros_like(attention)
        
        # Min-max normalization
        normalized = (attention - min_val) / (max_val - min_val)
        
        # Ensure values are in [0, 1] range (handle floating point errors)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        logger.debug(
            f"Normalized attention: min={min_val:.4f}, max={max_val:.4f} -> "
            f"[0.0, 1.0]"
        )
        
        return normalized

    
    def _upsample_to_image_size(
        self, 
        attention: np.ndarray, 
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Upsample attention map to match image dimensions using bilinear interpolation.
        
        Args:
            attention: 2D attention map (grid_h, grid_w)
            target_size: Target dimensions (height, width)
            
        Returns:
            Upsampled attention map with shape (height, width)
            
        Raises:
            ValueError: If target size is invalid
        """
        target_h, target_w = target_size
        
        if target_h <= 0 or target_w <= 0:
            raise ValueError(
                f"Invalid target size: ({target_h}, {target_w}). "
                "Both dimensions must be positive."
            )
        
        # Use cv2.resize with bilinear interpolation for fast upsampling
        # Note: cv2.resize expects (width, height) order
        upsampled = cv2.resize(
            attention, 
            (target_w, target_h), 
            interpolation=cv2.INTER_LINEAR
        )
        
        logger.debug(
            f"Upsampled attention: {attention.shape} -> {upsampled.shape}"
        )
        
        return upsampled

    
    def _apply_colormap(self, normalized_attention: np.ndarray) -> np.ndarray:
        """
        Apply perceptually-uniform colormap to normalized attention values.
        
        Args:
            normalized_attention: Normalized attention with values in [0, 1]
            
        Returns:
            RGB image with shape (height, width, 3) and dtype uint8
        """
        # Get or create colormap object (with caching)
        if self.colormap_name not in self.colormap_cache:
            self.colormap_cache[self.colormap_name] = cm.get_cmap(self.colormap_name)
            logger.debug(f"Created and cached colormap '{self.colormap_name}'")
        
        colormap = self.colormap_cache[self.colormap_name]
        
        # Apply colormap (returns RGBA with values in [0, 1])
        colored = colormap(normalized_attention)
        
        # Convert to RGB (drop alpha channel) and scale to [0, 255]
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        
        logger.debug(
            f"Applied colormap '{self.colormap_name}': "
            f"{normalized_attention.shape} -> {colored_rgb.shape}"
        )
        
        return colored_rgb

    
    def _blend_with_image(
        self, 
        heatmap: np.ndarray, 
        original_image: np.ndarray
    ) -> np.ndarray:
        """
        Blend colored heatmap with original image to create overlay.
        
        Args:
            heatmap: Colored heatmap (H, W, 3) with dtype uint8
            original_image: Original image (H, W, 3) with dtype uint8
            
        Returns:
            Blended overlay (H, W, 3) with dtype uint8
            
        Raises:
            ValueError: If images have incompatible shapes
        """
        if heatmap.shape != original_image.shape:
            raise ValueError(
                f"Heatmap and image must have same shape. "
                f"Got heatmap: {heatmap.shape}, image: {original_image.shape}"
            )
        
        # Convert to float for blending
        heatmap_float = heatmap.astype(np.float32)
        image_float = original_image.astype(np.float32)
        
        # Weighted average: overlay = alpha * heatmap + (1-alpha) * image
        blended = self.opacity * heatmap_float + (1 - self.opacity) * image_float
        
        # Clip to valid range and convert back to uint8
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        logger.debug(
            f"Blended heatmap with image at opacity={self.opacity}: "
            f"{heatmap.shape} + {original_image.shape} -> {blended.shape}"
        )
        
        return blended

    
    def generate_heatmap(
        self,
        attention_weights: torch.Tensor,
        original_image: np.ndarray,
        patch_size: int = 16
    ) -> Optional[np.ndarray]:
        """
        Generate attention heatmap overlay from ViT attention weights.
        
        This is the main entry point for heatmap generation. It orchestrates the
        complete pipeline: extraction → aggregation → spatial reshaping → upsampling
        → normalization → colormap application → blending.
        
        Args:
            attention_weights: Attention tensor from ViT final layer
                              Shape: (num_heads, num_patches+1, num_patches+1)
            original_image: Original input image as numpy array (H, W, 3)
            patch_size: Size of ViT patches (default 16 for ViT-base)
            
        Returns:
            Heatmap overlay as numpy array (H, W, 3) or None if generation fails
        """
        import time
        start_time = time.time()
        
        try:
            # Extraction pipeline
            aggregated = self._aggregate_attention_heads(attention_weights)
            cls_attention = self._extract_cls_attention(aggregated)
            spatial = self._reshape_to_spatial(cls_attention, patch_size)
            
            # Visualization pipeline
            upsampled = self._upsample_to_image_size(spatial, original_image.shape[:2])
            normalized = self._normalize_attention(upsampled)
            colored = self._apply_colormap(normalized)
            overlay = self._blend_with_image(colored, original_image)
            
            elapsed = time.time() - start_time
            logger.info(
                f"Successfully generated attention heatmap in {elapsed:.3f}s "
                f"(image size: {original_image.shape[:2]})"
            )
            
            # Warn if generation took too long
            if elapsed > 2.0:
                logger.warning(
                    f"Heatmap generation exceeded 2 second threshold: {elapsed:.3f}s"
                )
            
            return overlay
            
        except ValueError as e:
            logger.error(f"Invalid attention dimensions: {e}")
            return None
            
        except RuntimeError as e:
            logger.error(f"Memory error during heatmap generation: {e}")
            # Attempt lower resolution fallback
            try:
                logger.info("Attempting low-resolution fallback...")
                return self._generate_low_res_heatmap(
                    attention_weights, original_image, patch_size
                )
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return None
                
        except Exception as e:
            logger.error(
                f"Unexpected error in heatmap generation: {e}", 
                exc_info=True
            )
            return None
    
    def _generate_low_res_heatmap(
        self,
        attention_weights: torch.Tensor,
        original_image: np.ndarray,
        patch_size: int
    ) -> Optional[np.ndarray]:
        """
        Generate lower resolution heatmap as fallback for memory constraints.
        
        Args:
            attention_weights: Attention tensor from ViT final layer
            original_image: Original input image
            patch_size: Size of ViT patches
            
        Returns:
            Lower resolution heatmap overlay or None if still fails
        """
        # Downsample original image to half resolution
        h, w = original_image.shape[:2]
        downsampled_image = cv2.resize(
            original_image, 
            (w // 2, h // 2), 
            interpolation=cv2.INTER_AREA
        )
        
        # Generate heatmap at lower resolution
        aggregated = self._aggregate_attention_heads(attention_weights)
        cls_attention = self._extract_cls_attention(aggregated)
        spatial = self._reshape_to_spatial(cls_attention, patch_size)
        upsampled = self._upsample_to_image_size(spatial, downsampled_image.shape[:2])
        normalized = self._normalize_attention(upsampled)
        colored = self._apply_colormap(normalized)
        overlay = self._blend_with_image(colored, downsampled_image)
        
        # Upsample back to original resolution
        overlay_full_res = cv2.resize(
            overlay, 
            (w, h), 
            interpolation=cv2.INTER_LINEAR
        )
        
        logger.info(
            f"Generated low-resolution fallback heatmap: "
            f"{downsampled_image.shape[:2]} -> {overlay_full_res.shape[:2]}"
        )
        
        return overlay_full_res
