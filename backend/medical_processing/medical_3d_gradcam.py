#!/usr/bin/env python3
"""
3D GradCAM Implementation for Medical Image Analysis
Specifically designed for 3D CNN models used in medical imaging (CT, MRI, etc.)
Supports TensorFlow/Keras models with 3D convolutional layers.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime
import base64
import io
import psutil

try:
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow.keras.backend as K
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("❌ TensorFlow not available. This module requires TensorFlow for 3D CNN models.")

@dataclass
class GradCAM3DConfig:
    """Configuration for 3D GradCAM generation"""
    target_layer_name: Optional[str] = None  # Auto-detect if None
    colormap: str = "jet"    # jet, hot, viridis, plasma
    overlay_alpha: float = 0.4
    threshold_percentile: int = 95
    slice_visualization: str = "middle"  # middle, all, max_activation
    enhance_contrast: bool = True
    save_individual_slices: bool = True
    output_format: str = "png"

@dataclass
class GradCAM3DResult:
    """Result of 3D GradCAM generation"""
    original_volume_info: Dict[str, Any]
    heatmap_slices: List[str]  # Paths to individual slice heatmaps
    overlay_slices: List[str]  # Paths to overlay images
    volume_visualization: str  # Path to 3D volume visualization
    confidence_score: float
    predicted_class: str
    top_predictions: List[Dict[str, Any]]
    activation_regions: List[Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]

class Medical3DGradCAMGenerator:
    """3D GradCAM generator specifically for 3D medical image analysis"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the 3D GradCAM generator
        
        Args:
            model_path: Path to the 3D CNN model (.h5, .keras, .pb)
            device: Computing device ("auto", "cpu", "gpu")
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for 3D GradCAM. Please install TensorFlow.")
        
        self.model_path = model_path
        self.model = None
        self.model_input_shape = None
        self.device = self._setup_device(device)
        
        # Medical imaging specific settings
        self.medical_colormaps = {
            'bone': cm.bone,
            'hot': cm.hot,
            'jet': cm.jet,
            'viridis': cm.viridis,
            'plasma': cm.plasma,
            'inferno': cm.inferno,
            'coolwarm': cm.coolwarm
        }
        
        # Load the model
        self.load_model()
        
    def _check_memory_requirements(self, shape: Tuple, dtype_size: int = 4) -> bool:
        """Check if array allocation is safe"""
        import psutil
        
        required_memory = np.prod(shape) * dtype_size  # bytes
        available_memory = psutil.virtual_memory().available
        
        # Use only 25% of available memory to be safe
        safe_memory = available_memory * 0.25
        
        if required_memory > safe_memory:
            print(f"⚠️ Memory warning: Required {required_memory / (1024**3):.1f} GB, "
                  f"available {available_memory / (1024**3):.1f} GB")
            return False
        return True
        
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == "auto":
            # Check if GPU is available
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth for GPU
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    device = "gpu"
                    print(f"✅ Using GPU: {len(gpus)} GPU(s) available")
                except RuntimeError as e:
                    print(f"⚠️ GPU setup failed: {e}")
                    device = "cpu"
            else:
                device = "cpu"
                print("ℹ️ Using CPU (no GPU available)")
        return device
    
    def load_model(self):
        """Load the 3D CNN model with compatibility fixes"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            print(f"🔬 Loading 3D CNN model from: {self.model_path}")
            
            # Try different loading strategies to handle version incompatibilities
            self.model = None
            
            # Strategy 1: Try loading with compile=False to avoid optimizer issues
            try:
                print("   🔄 Trying to load without compilation...")
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                print("   ✅ Model loaded without compilation")
            except Exception as e:
                print(f"   ⚠️ Failed to load without compilation: {e}")
            
            # Strategy 2: If that failed, try with custom objects
            if self.model is None:
                try:
                    print("   🔄 Trying to load with custom objects...")
                    
                    # Define custom objects that might be missing
                    custom_objects = {
                        'ExponentialDecay': tf.keras.optimizers.schedules.ExponentialDecay,
                        'Adam': tf.keras.optimizers.Adam
                    }
                    
                    self.model = tf.keras.models.load_model(
                        self.model_path, 
                        custom_objects=custom_objects,
                        compile=False
                    )
                    print("   ✅ Model loaded with custom objects")
                except Exception as e:
                    print(f"   ⚠️ Failed to load with custom objects: {e}")
            
            # Strategy 3: Load just the architecture and weights separately
            if self.model is None:
                try:
                    print("   🔄 Trying to load architecture and weights separately...")
                    
                    # Load model architecture only
                    import h5py
                    with h5py.File(self.model_path, 'r') as f:
                        # Try to reconstruct model from saved architecture
                        if 'model_config' in f.attrs:
                            model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
                            self.model = tf.keras.models.model_from_json(json.dumps(model_config))
                            self.model.load_weights(self.model_path)
                            print("   ✅ Model loaded from architecture and weights")
                        else:
                            raise Exception("No model config found in file")
                            
                except Exception as e:
                    print(f"   ⚠️ Failed to load architecture separately: {e}")
            
            # If all strategies failed, raise an error
            if self.model is None:
                raise Exception("All model loading strategies failed")
            
            # Get model input shape
            self.model_input_shape = self.model.input_shape
            print(f"✅ Model loaded successfully")
            print(f"📊 Input shape: {self.model_input_shape}")
            print(f"📊 Output shape: {self.model.output_shape}")
            
            # Verify it's a 3D model
            if len(self.model_input_shape) != 5:  # (batch, width, height, depth, channels)
                print("⚠️ Warning: Model doesn't appear to be a 3D CNN (expected 5D input)")
            
            # Print model summary
            print("\n📋 Model Architecture:")
            self.model.summary()
            
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def _find_target_layer(self, target_layer_name: Optional[str] = None) -> keras.layers.Layer:
        """Find the target convolutional layer for GradCAM"""
        if target_layer_name:
            try:
                return self.model.get_layer(target_layer_name)
            except:
                print(f"⚠️ Layer '{target_layer_name}' not found, auto-detecting...")
        
        # Auto-detect the last 3D convolutional layer
        conv3d_layers = []
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Conv3D):
                conv3d_layers.append(layer)
        
        if not conv3d_layers:
            raise ValueError("No 3D convolutional layers found in the model")
        
        target_layer = conv3d_layers[-1]  # Use the last Conv3D layer
        print(f"🎯 Using target layer: {target_layer.name} (shape: {target_layer.output.shape})")
        return target_layer
    
    def preprocess_3d_volume(self, volume_data: np.ndarray) -> np.ndarray:
        """
        Preprocess 3D medical volume for model input with memory optimization
        
        Args:
            volume_data: Raw 3D volume data (H, W, D) or (H, W, D, C)
            
        Returns:
            Preprocessed volume ready for model input
        """
        # Ensure we're using float32 to save memory
        if volume_data.dtype != np.float32:
            volume_data = volume_data.astype(np.float32)
        
        # Handle different input formats
        if volume_data.ndim == 3:
            # Add channel dimension: (H, W, D) -> (H, W, D, 1)
            volume_data = np.expand_dims(volume_data, axis=-1)
        
        print(f"📊 Input volume shape: {volume_data.shape}")
        print(f"💾 Input volume memory: {volume_data.nbytes / (1024**2):.1f} MB")
        
        # Get expected input shape (excluding batch dimension)
        expected_shape = self.model_input_shape[1:]  # Remove batch dimension
        
        # Check if we need to resize and if the expected shape is reasonable
        current_size = np.prod(volume_data.shape)
        expected_size = np.prod(expected_shape)
        
        # If expected size is too large, limit it to prevent memory issues
        max_voxels = 64 * 1024 * 1024  # 64M voxels max (about 256MB for float32)
        
        if expected_size > max_voxels:
            print(f"⚠️ Model expects very large input ({expected_size} voxels). Limiting to prevent memory issues.")
            
            # Scale down the expected shape proportionally
            scale_factor = (max_voxels / expected_size) ** (1/3)  # Cube root for 3D scaling
            
            target_h = max(32, int(expected_shape[0] * scale_factor))
            target_w = max(32, int(expected_shape[1] * scale_factor))
            target_d = max(8, int(expected_shape[2] * scale_factor))
            target_c = expected_shape[3]
            
            expected_shape = (target_h, target_w, target_d, target_c)
            print(f"🔄 Adjusted target shape to: {expected_shape}")
        
        # Resize if necessary
        if volume_data.shape != expected_shape:
            print(f"🔄 Resizing volume from {volume_data.shape} to {expected_shape}")
            
            target_h, target_w, target_d, target_c = expected_shape
            current_h, current_w, current_d, current_c = volume_data.shape
            
            # Pre-allocate resized volume with float32
            resized_volume = np.zeros(expected_shape, dtype=np.float32)
            
            # Resize slice by slice to be memory efficient
            for d in range(target_d):
                # Map depth index
                src_d = min(int(d * current_d / target_d), current_d - 1)
                
                for c in range(target_c):
                    src_c = min(c, current_c - 1)
                    # Resize 2D slice
                    slice_2d = cv2.resize(
                        volume_data[:, :, src_d, src_c], 
                        (target_w, target_h), 
                        interpolation=cv2.INTER_LINEAR
                    )
                    resized_volume[:, :, d, c] = slice_2d
            
            # Clear original volume to free memory
            del volume_data
            volume_data = resized_volume
        
        # Normalize to [0, 1] range
        if volume_data.max() > 1.0:
            volume_data = volume_data / 255.0
        
        # Add batch dimension: (H, W, D, C) -> (1, H, W, D, C)
        volume_data = np.expand_dims(volume_data, axis=0)
        
        print(f"✅ Preprocessed volume shape: {volume_data.shape}")
        print(f"💾 Final volume memory: {volume_data.nbytes / (1024**2):.1f} MB")
        return volume_data
    
    def generate_3d_gradcam(
        self, 
        volume_data: np.ndarray, 
        config: GradCAM3DConfig = None,
        save_dir: str = "heatmaps_3d"
    ) -> GradCAM3DResult:
        """
        Generate 3D GradCAM heatmap for a 3D medical volume
        
        Args:
            volume_data: 3D medical volume (H, W, D) or (H, W, D, C)
            config: Configuration for GradCAM generation
            save_dir: Directory to save outputs
            
        Returns:
            GradCAM3DResult containing all generated visualizations
        """
        if config is None:
            config = GradCAM3DConfig()
        
        os.makedirs(save_dir, exist_ok=True)
        start_time = datetime.now()
        
        print("🔥 Starting 3D GradCAM generation...")
        
        # Preprocess volume
        processed_volume = self.preprocess_3d_volume(volume_data)
        
        # Find target layer
        target_layer = self._find_target_layer(config.target_layer_name)
        
        # Create GradCAM model
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs], 
            outputs=[target_layer.output, self.model.output]
        )
        
        # Forward pass and compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(processed_volume)
            
            # Handle binary classification (single output) vs multi-class
            if predictions.shape[1] == 1:
                # Binary classification - use the single output
                loss = predictions[0, 0]
                predicted_class_idx = 0
            else:
                # Multi-class classification
                predicted_class_idx = tf.argmax(predictions[0])
                loss = predictions[:, predicted_class_idx]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Compute 3D GradCAM
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
        conv_outputs = conv_outputs[0]
        
        # Weight the feature maps by importance
        heatmap_3d = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_outputs), axis=-1
        ).numpy()
        
        # Apply ReLU to keep only positive values
        heatmap_3d = np.maximum(heatmap_3d, 0)
        
        # Normalize heatmap
        if heatmap_3d.max() > 0:
            heatmap_3d = heatmap_3d / heatmap_3d.max()
        
        print(f"🔥 3D GradCAM computed. Shape: {heatmap_3d.shape}")
        
        # Resize heatmap to original volume size
        original_shape = volume_data.shape[:3] if volume_data.ndim == 4 else volume_data.shape
        heatmap_resized = self._resize_3d_heatmap(heatmap_3d, original_shape)
        
        # Extract predictions
        predictions_list = self._extract_predictions(predictions[0].numpy())
        
        # Generate visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create slice-wise visualizations
        heatmap_slices, overlay_slices = self._create_slice_visualizations(
            volume_data, heatmap_resized, config, save_dir, timestamp
        )
        
        # Create 3D volume visualization
        volume_viz_path = self._create_volume_visualization(
            heatmap_resized, config, save_dir, timestamp
        )
        
        # Find activation regions
        activation_regions = self._find_3d_activation_regions(
            heatmap_resized, config.threshold_percentile
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = GradCAM3DResult(
            original_volume_info={
                'shape': original_shape,
                'dtype': str(volume_data.dtype),
                'min_value': float(volume_data.min()),
                'max_value': float(volume_data.max())
            },
            heatmap_slices=heatmap_slices,
            overlay_slices=overlay_slices,
            volume_visualization=volume_viz_path,
            confidence_score=float(predictions_list[0]['confidence']) if predictions_list else 0.0,
            predicted_class=predictions_list[0]['class'] if predictions_list else "Unknown",
            top_predictions=predictions_list,
            activation_regions=activation_regions,
            processing_time=processing_time,
            metadata={
                'model_path': self.model_path,
                'model_input_shape': self.model_input_shape,
                'target_layer': target_layer.name,
                'config': config.__dict__,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        print(f"✅ 3D GradCAM completed in {processing_time:.2f}s")
        print(f"📊 Generated {len(heatmap_slices)} slice visualizations")
        print(f"🎯 Predicted: {result.predicted_class} ({result.confidence_score:.2%})")
        
        return result
    
    def _resize_3d_heatmap(self, heatmap_3d: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Resize 3D heatmap to target shape with memory optimization"""
        target_h, target_w, target_d = target_shape
        current_h, current_w, current_d = heatmap_3d.shape
        
        print(f"🔄 Resizing heatmap from {heatmap_3d.shape} to {target_shape}")
        
        # Check if resize is needed
        if (current_h, current_w, current_d) == target_shape:
            return heatmap_3d.astype(np.float32)
        
        # Check memory requirements and limit if necessary
        target_size = target_h * target_w * target_d * 4  # 4 bytes per float32
        max_size = 512 * 1024 * 1024  # 512MB limit
        
        if target_size > max_size:
            print(f"⚠️ Target heatmap size ({target_size / (1024**2):.1f} MB) too large. Limiting dimensions.")
            
            # Scale down proportionally
            scale_factor = (max_size / target_size) ** (1/3)
            target_h = max(32, int(target_h * scale_factor))
            target_w = max(32, int(target_w * scale_factor))
            target_d = max(8, int(target_d * scale_factor))
            
            print(f"🔄 Adjusted heatmap target to: {target_h}x{target_w}x{target_d}")
        
        # Create output array with float32
        resized_heatmap = np.zeros((target_h, target_w, target_d), dtype=np.float32)
        
        # Resize slice by slice to minimize memory usage
        for d in range(target_d):
            # Map depth index
            src_d = min(int(d * current_d / target_d), current_d - 1)
            
            # Resize 2D slice
            resized_slice = cv2.resize(
                heatmap_3d[:, :, src_d], 
                (target_w, target_h), 
                interpolation=cv2.INTER_LINEAR
            )
            resized_heatmap[:, :, d] = resized_slice
        
        print(f"✅ Heatmap resized. Memory: {resized_heatmap.nbytes / (1024**2):.1f} MB")
        return resized_heatmap
    
    def _extract_predictions(self, output: np.ndarray) -> List[Dict]:
        """Extract predictions from model output"""
        
        # Handle both 1D and 2D output arrays
        if len(output.shape) == 2:
            output = output[0]  # Take first batch element
        
        if len(output) == 1:
            # Binary classification with sigmoid output
            prob_abnormal = float(output[0])
            prob_normal = 1.0 - prob_abnormal
            
            predictions = [
                {'class': 'Abnormal', 'confidence': prob_abnormal, 'probability': prob_abnormal},
                {'class': 'Normal', 'confidence': prob_normal, 'probability': prob_normal}
            ]
        else:
            # Multi-class classification with softmax
            # Apply softmax if not already applied
            if output.max() > 1.0 or output.min() < 0.0:
                probabilities = tf.nn.softmax(output).numpy()
            else:
                probabilities = output
                
            predictions = []
            class_names = ['Normal', 'Abnormal', 'Tumor', 'Other']  # Extend as needed
            
            for i, prob in enumerate(probabilities):
                class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
                predictions.append({
                    'class': class_name,
                    'confidence': float(prob),
                    'probability': float(prob)
                })
        
        return sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    def _create_slice_visualizations(
        self, 
        volume_data: np.ndarray, 
        heatmap_3d: np.ndarray, 
        config: GradCAM3DConfig,
        save_dir: str,
        timestamp: str
    ) -> Tuple[List[str], List[str]]:
        """Create slice-wise visualizations"""
        
        heatmap_paths = []
        overlay_paths = []
        
        # Determine which slices to visualize
        depth = heatmap_3d.shape[2]
        
        if config.slice_visualization == "middle":
            slice_indices = [depth // 2]
        elif config.slice_visualization == "max_activation":
            # Find slice with maximum activation
            slice_activations = np.sum(heatmap_3d, axis=(0, 1))
            max_slice = np.argmax(slice_activations)
            slice_indices = [max_slice]
        elif config.slice_visualization == "all":
            slice_indices = list(range(0, depth, max(1, depth // 10)))  # Max 10 slices
        else:
            slice_indices = [depth // 2]  # Default to middle
        
        print(f"📸 Creating visualizations for slices: {slice_indices}")
        
        for slice_idx in slice_indices:
            # Get original slice (handle both 3D and 4D volumes)
            if volume_data.ndim == 4:
                original_slice = volume_data[:, :, slice_idx, 0]  # Take first channel
            else:
                original_slice = volume_data[:, :, slice_idx]
            
            # Get heatmap slice
            heatmap_slice = heatmap_3d[:, :, slice_idx]
            
            # Normalize original slice to 0-255
            original_normalized = ((original_slice - original_slice.min()) / 
                                 (original_slice.max() - original_slice.min() + 1e-8) * 255).astype(np.uint8)
            
            # Apply colormap to heatmap
            heatmap_colored = self._apply_colormap(heatmap_slice, config.colormap)
            
            # Create overlay
            overlay = self._create_overlay(original_normalized, heatmap_colored, config.overlay_alpha)
            
            # Save images
            slice_name = f"slice_{slice_idx:03d}_{timestamp}"
            
            # Save heatmap
            heatmap_path = os.path.join(save_dir, f"{slice_name}_heatmap.{config.output_format}")
            Image.fromarray(heatmap_colored).save(heatmap_path)
            heatmap_paths.append(heatmap_path)
            
            # Save overlay
            overlay_path = os.path.join(save_dir, f"{slice_name}_overlay.{config.output_format}")
            Image.fromarray(overlay).save(overlay_path)
            overlay_paths.append(overlay_path)
        
        return heatmap_paths, overlay_paths
    
    def _create_volume_visualization(
        self, 
        heatmap_3d: np.ndarray, 
        config: GradCAM3DConfig,
        save_dir: str,
        timestamp: str
    ) -> str:
        """Create 3D volume visualization"""
        
        # Create a montage of slices for 3D visualization
        depth = heatmap_3d.shape[2]
        rows = int(np.ceil(np.sqrt(depth)))
        cols = int(np.ceil(depth / rows))
        
        slice_height, slice_width = heatmap_3d.shape[:2]
        montage_height = rows * slice_height
        montage_width = cols * slice_width
        
        # Create montage
        montage = np.zeros((montage_height, montage_width), dtype=np.float32)
        
        for i in range(depth):
            row = i // cols
            col = i % cols
            
            start_row = row * slice_height
            end_row = start_row + slice_height
            start_col = col * slice_width
            end_col = start_col + slice_width
            
            montage[start_row:end_row, start_col:end_col] = heatmap_3d[:, :, i]
        
        # Apply colormap
        montage_colored = self._apply_colormap(montage, config.colormap)
        
        # Add grid lines for clarity
        montage_colored = self._add_grid_lines(montage_colored, rows, cols, slice_height, slice_width)
        
        # Save volume visualization
        volume_path = os.path.join(save_dir, f"volume_3d_{timestamp}.{config.output_format}")
        Image.fromarray(montage_colored).save(volume_path)
        
        print(f"📦 3D volume visualization saved: {volume_path}")
        return volume_path
    
    def _add_grid_lines(self, image: np.ndarray, rows: int, cols: int, slice_height: int, slice_width: int) -> np.ndarray:
        """Add grid lines to volume montage"""
        result = image.copy()
        
        # Add horizontal lines
        for row in range(1, rows):
            y = row * slice_height
            result[y:y+2, :] = [255, 255, 255]  # White lines
        
        # Add vertical lines
        for col in range(1, cols):
            x = col * slice_width
            result[:, x:x+2] = [255, 255, 255]  # White lines
        
        return result
    
    def _apply_colormap(self, heatmap: np.ndarray, colormap_name: str) -> np.ndarray:
        """Apply colormap to heatmap"""
        # Enhance contrast
        heatmap_enhanced = np.power(heatmap, 0.7)  # Gamma correction
        
        # Select colormap
        if colormap_name in self.medical_colormaps:
            colormap = self.medical_colormaps[colormap_name]
        else:
            colormap = cm.jet
        
        # Apply colormap
        colored = colormap(heatmap_enhanced)[:, :, :3]  # Remove alpha channel
        colored = (colored * 255).astype(np.uint8)
        
        return colored
    
    def _create_overlay(self, original: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
        """Create overlay of original image and heatmap"""
        # Convert grayscale original to RGB
        if len(original.shape) == 2:
            original_rgb = np.stack([original] * 3, axis=2)
        else:
            original_rgb = original
        
        # Blend images
        overlay = cv2.addWeighted(original_rgb, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
    
    def _find_3d_activation_regions(self, heatmap_3d: np.ndarray, threshold_percentile: int) -> List[Dict[str, Any]]:
        """Find significant 3D activation regions"""
        threshold = np.percentile(heatmap_3d, threshold_percentile)
        binary_mask = (heatmap_3d >= threshold).astype(np.uint8)
        
        regions = []
        
        # Find connected components in 3D
        # For simplicity, we'll analyze slice by slice and combine
        for z in range(heatmap_3d.shape[2]):
            slice_mask = binary_mask[:, :, z]
            contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > 50:  # Filter small regions
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate region statistics
                    region_mask = np.zeros_like(slice_mask)
                    cv2.fillPoly(region_mask, [contour], 1)
                    region_values = heatmap_3d[:, :, z][region_mask == 1]
                    
                    regions.append({
                        'id': len(regions),
                        'slice': int(z),
                        'bbox_2d': [int(x), int(y), int(w), int(h)],
                        'area': int(cv2.contourArea(contour)),
                        'max_activation': float(np.max(region_values)),
                        'mean_activation': float(np.mean(region_values)),
                        'centroid': [int(x + w//2), int(y + h//2), int(z)]
                    })
        
        return sorted(regions, key=lambda x: x['max_activation'], reverse=True)


def load_3d_medical_volume(file_path: str) -> np.ndarray:
    """
    Load 3D medical volume from various formats
    
    Args:
        file_path: Path to the medical volume file
        
    Returns:
        3D numpy array representing the medical volume
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ['.nii', '.nii.gz']:
        # NIfTI format (common for medical imaging)
        try:
            import nibabel as nib
            nii_img = nib.load(file_path)
            volume = nii_img.get_fdata()
            print(f"✅ Loaded NIfTI volume: {volume.shape}")
            return volume
        except ImportError:
            raise ImportError("nibabel required for NIfTI files. Install with: pip install nibabel")
    
    elif file_ext in ['.dcm']:
        # DICOM format
        try:
            import pydicom
            import glob
            
            # If single DICOM file, try to load DICOM series from directory
            dicom_dir = os.path.dirname(file_path)
            dicom_files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
            
            if len(dicom_files) > 1:
                # Load DICOM series
                slices = [pydicom.dcmread(f) for f in dicom_files]
                slices.sort(key=lambda x: x.ImagePositionPatient[2])  # Sort by position
                volume = np.stack([s.pixel_array for s in slices], axis=2)
            else:
                # Single DICOM file
                ds = pydicom.dcmread(file_path)
                volume = ds.pixel_array
                if volume.ndim == 2:
                    volume = np.expand_dims(volume, axis=2)  # Add depth dimension
            
            print(f"✅ Loaded DICOM volume: {volume.shape}")
            return volume
        except ImportError:
            raise ImportError("pydicom required for DICOM files. Install with: pip install pydicom")
    
    elif file_ext in ['.npy']:
        # Numpy array
        volume = np.load(file_path)
        print(f"✅ Loaded numpy volume: {volume.shape}")
        return volume
    
    elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
        # 2D image - create fake 3D volume for testing with memory-efficient processing
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
        
        print(f"📊 Original image shape: {image.shape}")
        
        # Resize image to reasonable size to prevent memory issues
        max_size = 512  # Maximum dimension size
        h, w = image.shape
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            print(f"🔄 Resizing image from {h}x{w} to {new_h}x{new_w} to prevent memory issues")
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to float32 to save memory (instead of default float64)
        image = image.astype(np.float32)
        
        # Create a 3D volume by stacking the same image multiple times
        depth = 32  # Reduced depth to save memory (was 64)
        
        # Memory-efficient volume creation
        h, w = image.shape
        volume = np.zeros((h, w, depth), dtype=np.float32)  # Pre-allocate with float32
        
        # Fill the volume slice by slice to be memory efficient
        for d in range(depth):
            volume[:, :, d] = image
            
        print(f"✅ Created 3D volume from 2D image: {volume.shape} (dtype: {volume.dtype})")
        print(f"💾 Memory usage: {volume.nbytes / (1024**2):.1f} MB")
        return volume
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


# Utility function for easy usage
def generate_3d_gradcam_for_medical_image(
    model_path: str,
    volume_path: str,
    output_dir: str = "gradcam_results",
    config: GradCAM3DConfig = None
) -> GradCAM3DResult:
    """
    Generate 3D GradCAM for a medical volume - high-level convenience function
    
    Args:
        model_path: Path to the 3D CNN model (.h5, .keras)
        volume_path: Path to the 3D medical volume
        output_dir: Directory to save results
        config: Configuration for GradCAM generation
        
    Returns:
        GradCAM3DResult with all generated visualizations
    """
    
    print("🏥 3D Medical GradCAM Generator")
    print("=" * 50)
    
    # Load volume
    print(f"📂 Loading medical volume: {volume_path}")
    volume_data = load_3d_medical_volume(volume_path)
    
    # Initialize generator
    print(f"🤖 Loading 3D CNN model: {model_path}")
    generator = Medical3DGradCAMGenerator(model_path)
    
    # Generate GradCAM
    result = generator.generate_3d_gradcam(volume_data, config, output_dir)
    
    print("\n✅ 3D GradCAM Generation Complete!")
    print("=" * 50)
    print(f"🎯 Prediction: {result.predicted_class}")
    print(f"📊 Confidence: {result.confidence_score:.2%}")
    print(f"⏱️ Processing time: {result.processing_time:.2f}s")
    print(f"📸 Slice visualizations: {len(result.heatmap_slices)}")
    print(f"📦 Volume visualization: {os.path.basename(result.volume_visualization)}")
    print(f"📁 Results saved to: {output_dir}")
    
    return result


if __name__ == "__main__":
    # Demo usage
    print("🔥 3D Medical GradCAM Demo")
    
    # Example usage with your model
    model_path = "3d_image_classification.h5"
    
    if os.path.exists(model_path):
        print(f"✅ Found model: {model_path}")
        
        # Create a demo 3D volume for testing
        demo_volume = np.random.rand(128, 128, 64).astype(np.float32)
        demo_volume_path = "demo_3d_volume.npy"
        np.save(demo_volume_path, demo_volume)
        print(f"✅ Created demo 3D volume: {demo_volume_path}")
        
        try:
            # Generate 3D GradCAM
            config = GradCAM3DConfig(
                colormap="jet",
                overlay_alpha=0.5,
                slice_visualization="middle",
                enhance_contrast=True
            )
            
            result = generate_3d_gradcam_for_medical_image(
                model_path=model_path,
                volume_path=demo_volume_path,
                config=config
            )
            
            print("🎉 Demo completed successfully!")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup
        try:
            os.remove(demo_volume_path)
        except:
            pass
    else:
        print(f"❌ Model not found: {model_path}")
        print("Please ensure your 3d_image_classification.h5 model is in the current directory.")
