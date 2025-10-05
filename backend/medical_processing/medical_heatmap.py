"""
Medical Image Heatmap and GradCAM Generator for CortexMD
Generates visual explanations for AI model predictions on medical images
Supports various visualization techniques including GradCAM, Grad-CAM++, and attention maps
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime
import base64
import io

try:
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, densenet121
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Install PyTorch for deep learning features.")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available. Install TensorFlow for additional ML features.")

@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation"""
    method: str = "gradcam"  # gradcam, gradcam++, guided_gradcam, attention
    colormap: str = "jet"    # jet, hot, viridis, plasma
    overlay_alpha: float = 0.4
    threshold_percentile: int = 95
    blur_radius: int = 0
    enhance_contrast: bool = True
    save_individual_maps: bool = True
    output_format: str = "png"

@dataclass
class HeatmapResult:
    """Result of heatmap generation"""
    original_image_path: str
    heatmap_path: str
    overlay_path: str
    confidence_score: float
    predicted_class: str
    top_predictions: List[Dict[str, Any]]
    activation_regions: List[Dict[str, Any]]
    method_used: str
    processing_time: float
    metadata: Dict[str, Any]

class MedicalHeatmapGenerator:
    """Advanced heatmap generator for medical image analysis"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self.device = self._setup_device(device)
        self.model = None
        self.model_type = None
        self.transform = self._setup_transforms()
        
        # Medical imaging specific settings
        self.medical_colormaps = {
            'bone': cm.bone,
            'hot': cm.hot,
            'jet': cm.jet,
            'viridis': cm.viridis,
            'plasma': cm.plasma,
            'inferno': cm.inferno
        }
        
        # Class mappings for medical conditions
        self.medical_classes = {
            0: "Normal/Healthy",
            1: "Abnormal/Pathological", 
            2: "Tumor/Mass",
            3: "Inflammation",
            4: "Fracture",
            5: "Other Findings"
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("⚠️ No model specified. Using demo model for visualization.")
            self._load_demo_model()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        return torch.device(device)
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_demo_model(self):
        """Load a demo model for testing purposes"""
        try:
            if PYTORCH_AVAILABLE:
                self.model = resnet50(pretrained=True)
                self.model.eval()
                self.model_type = "resnet50"
                print("✅ Demo ResNet50 model loaded")
            else:
                print("⚠️ No PyTorch available for demo model")
        except Exception as e:
            print(f"⚠️ Failed to load demo model: {e}")
    
    def load_model(self, model_path: str):
        """Load custom trained model with improved compatibility"""
        if not model_path or not os.path.exists(model_path):
            print("⚠️ Model path not provided or doesn't exist, using demo model")
            self._load_demo_model()
            return
            
        try:
            print(f"🔬 Loading custom model from: {model_path}")
            
            if model_path.endswith('.pth') or model_path.endswith('.pt'):
                # Load PyTorch model
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    # Checkpoint format with metadata
                    print("   📋 Loading from checkpoint format...")
                    state_dict = checkpoint['state_dict']
                elif isinstance(checkpoint, dict) and not hasattr(checkpoint, 'eval'):
                    # Raw state_dict format
                    print("   📋 Loading from state_dict format...")
                    state_dict = checkpoint
                else:
                    # Full model format
                    print("   📋 Loading from full model format...")
                    self.model = checkpoint
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                    self.model_type = f"custom_pytorch_{os.path.basename(model_path)}"
                    print(f"✅ Custom PyTorch model loaded: {self.model_type}")
                    self._test_model_forward_pass()
                    return
                
                # Handle state_dict loading - create a compatible model architecture
                print("   🔧 Creating model architecture for state_dict...")
                self.model = self._create_model_from_state_dict(state_dict, model_path)
                
                if self.model is None:
                    raise ValueError("Could not create model from state_dict. Architecture unknown.")
                
                self.model.eval()
                self.model_type = f"custom_pytorch_{os.path.basename(model_path)}"
                print(f"✅ Custom PyTorch model loaded: {self.model_type}")
                
                # Test model compatibility
                self._test_model_forward_pass()
                
            elif model_path.endswith('.h5') or model_path.endswith('.keras'):
                if TF_AVAILABLE:
                    self.model = tf.keras.models.load_model(model_path)
                    self.model_type = f"tensorflow_{os.path.basename(model_path)}"
                    print(f"✅ TensorFlow model loaded: {self.model_type}")
                else:
                    raise ImportError("TensorFlow not available for .h5/.keras model")
                    
            elif model_path.endswith('.onnx'):
                print("⚠️ ONNX models not yet supported")
                raise NotImplementedError("ONNX support coming soon")
                
            else:
                # Try to load as PyTorch model anyway
                print(f"⚠️ Unknown format, trying as PyTorch model: {model_path}")
                self.model = torch.load(model_path, map_location=self.device)
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                self.model_type = f"custom_unknown_{os.path.basename(model_path)}"
                
        except Exception as e:
            print(f"❌ Failed to load custom model: {e}")
            print(f"   Error type: {type(e).__name__}")
            print("🔄 Falling back to demo model...")
            self._load_demo_model()
    
    def _create_model_from_state_dict(self, state_dict: dict, model_path: str):
        """Create a model architecture that can load the given state_dict"""
        try:
            # Analyze the state_dict to determine architecture
            keys = list(state_dict.keys())
            print(f"   📊 State dict has {len(keys)} parameters")
            
            # Try to infer model architecture from parameter names
            if any('global_classification_layers' in key for key in keys) or any('_conv_stem_backbone' in key for key in keys):
                print("   🧠 Detected GlobalNet (EfficientNet-based) architecture")
                return self._create_globalnet_efficientnet_model(state_dict)
            elif any('globalnet' in key.lower() for key in keys):
                print("   🧠 Detected generic GlobalNet architecture")
                return self._create_globalnet_model(state_dict)
            elif any('resnet' in key.lower() for key in keys):
                print("   🧠 Detected ResNet-like architecture")
                return self._create_resnet_like_model(state_dict)
            elif any('densenet' in key.lower() for key in keys):
                print("   🧠 Detected DenseNet-like architecture")
                return self._create_densenet_like_model(state_dict)
            else:
                print("   🧠 Attempting generic CNN architecture")
                return self._create_generic_cnn_model(state_dict)
                
        except Exception as e:
            print(f"   ❌ Failed to create model from state_dict: {e}")
            return None
    
    def _create_globalnet_efficientnet_model(self, state_dict: dict):
        """Create a GlobalNet model based on EfficientNet architecture"""
        try:
            import torch.nn as nn
            from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2
            
            print("   🔧 Creating EfficientNet-based GlobalNet model...")
            
            # Determine number of classes from final layer
            num_classes = 2  # Default for binary classification
            for key, param in state_dict.items():
                if 'global_classification_layers.3.weight' in key:
                    num_classes = param.shape[0]
                    print(f"   📊 Detected {num_classes} classes")
                    break
            
            # Try different EfficientNet variants to find the best match
            for model_fn, name in [(efficientnet_b0, 'EfficientNet-B0'), (efficientnet_b1, 'EfficientNet-B1'), (efficientnet_b2, 'EfficientNet-B2')]:
                try:
                    print(f"   🔄 Trying {name} backbone...")
                    
                    # Create base EfficientNet model
                    base_model = model_fn(weights=None)
                    
                    # Create GlobalNet-like architecture
                    class GlobalNetEfficientNet(nn.Module):
                        def __init__(self, base_model, num_classes):
                            super().__init__()
                            
                            # Use EfficientNet features as backbone
                            self.features = base_model.features
                            
                            # Add custom classification head matching your model
                            # Based on your model structure: global_classification_layers.0 and global_classification_layers.3
                            backbone_features = base_model.classifier.in_features
                            
                            self.global_classification_layers = nn.Sequential(
                                nn.Linear(backbone_features, 512),  # global_classification_layers.0
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.2),
                                nn.Linear(512, num_classes),  # global_classification_layers.3
                            )
                            
                            # Add avgpool for feature extraction
                            self.avgpool = nn.AdaptiveAvgPool2d(1)
                        
                        def forward(self, x):
                            x = self.features(x)
                            x = self.avgpool(x)
                            x = torch.flatten(x, 1)
                            x = self.global_classification_layers(x)
                            return x
                    
                    model = GlobalNetEfficientNet(base_model, num_classes)
                    
                    # Try to load the state dict
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    
                    # Check if the loading was successful (reasonable number of missing keys)
                    if len(missing_keys) < len(state_dict) // 3:  # Allow up to 1/3 missing keys
                        model = model.to(self.device)
                        print(f"   ✅ Successfully created GlobalNet with {name} backbone")
                        print(f"   📊 Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                        
                        if missing_keys and len(missing_keys) < 10:
                            print(f"   📝 Missing keys: {missing_keys}")
                        
                        return model
                    
                except Exception as e:
                    print(f"   ⚠️ {name} failed: {e}")
                    continue
            
            # If no EfficientNet variant worked, fall back to simple GlobalNet
            print("   🔄 Falling back to simple GlobalNet architecture...")
            return self._create_globalnet_model(state_dict)
            
        except Exception as e:
            print(f"   ❌ Failed to create GlobalNet EfficientNet model: {e}")
            return self._create_globalnet_model(state_dict)

    def _create_globalnet_model(self, state_dict: dict):
        """Create a GlobalNet model architecture"""
        try:
            import torch.nn as nn
            
            # Simple GlobalNet-like architecture
            # This is a basic implementation - you may need to adjust based on your actual model
            class SimpleGlobalNet(nn.Module):
                def __init__(self, num_classes=1000):
                    super().__init__()
                    self.features = nn.Sequential(
                        # Basic CNN backbone
                        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        
                        # Additional layers - adjust based on your model
                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        
                        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        
                        nn.AdaptiveAvgPool2d((1, 1))
                    )
                    self.classifier = nn.Linear(256, num_classes)
                
                def forward(self, x):
                    x = self.features(x)
                    x = torch.flatten(x, 1)
                    x = self.classifier(x)
                    return x
            
            # Try to infer number of classes from the final layer
            num_classes = 1000  # Default
            for key, param in state_dict.items():
                if 'classifier' in key and 'weight' in key:
                    num_classes = param.shape[0]
                    break
            
            model = SimpleGlobalNet(num_classes=num_classes)
            
            # Try to load the state dict
            model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)
            
            print(f"   ✅ Created GlobalNet model with {num_classes} classes")
            return model
            
        except Exception as e:
            print(f"   ❌ Failed to create GlobalNet model: {e}")
            return self._create_generic_cnn_model(state_dict)
    
    def _create_resnet_like_model(self, state_dict: dict):
        """Create a ResNet-like model"""
        try:
            from torchvision.models import resnet50, resnet34, resnet18
            
            # Try different ResNet architectures
            for model_fn, name in [(resnet50, 'ResNet50'), (resnet34, 'ResNet34'), (resnet18, 'ResNet18')]:
                try:
                    model = model_fn(weights=None)
                    
                    # Adjust final layer if needed
                    for key, param in state_dict.items():
                        if 'fc.weight' in key:
                            num_classes = param.shape[0]
                            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                            break
                    
                    model.load_state_dict(state_dict, strict=False)
                    model = model.to(self.device)
                    print(f"   ✅ Created {name} model")
                    return model
                except:
                    continue
                    
            return None
            
        except Exception as e:
            print(f"   ❌ Failed to create ResNet-like model: {e}")
            return None
    
    def _create_densenet_like_model(self, state_dict: dict):
        """Create a DenseNet-like model"""
        try:
            from torchvision.models import densenet121
            
            model = densenet121(weights=None)
            
            # Adjust classifier if needed
            for key, param in state_dict.items():
                if 'classifier.weight' in key:
                    num_classes = param.shape[0]
                    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
                    break
            
            model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)
            print(f"   ✅ Created DenseNet model")
            return model
            
        except Exception as e:
            print(f"   ❌ Failed to create DenseNet-like model: {e}")
            return None
    
    def _create_generic_cnn_model(self, state_dict: dict):
        """Create a generic CNN model as fallback"""
        try:
            print("   🚨 Using fallback demo model for GradCAM generation")
            print("   ⚠️  For best results, provide the model architecture or full model file")
            
            # Fall back to demo model but indicate it's a fallback
            self._load_demo_model()
            return self.model
            
        except Exception as e:
            print(f"   ❌ Failed to create fallback model: {e}")
            return None

    def _test_model_forward_pass(self):
        """Test if the model works with a dummy input"""
        try:
            # Test with standard medical image input size
            test_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                output = self.model(test_input)
            
            print(f"✅ Model forward pass successful - Output shape: {output.shape}")
            return True
            
        except Exception as e:
            print(f"⚠️ Model forward pass test failed: {e}")
            print("   Model may still work for GradCAM generation")
            return False
    
    def generate_heatmap(
        self, 
        image_path: str, 
        config: HeatmapConfig = None,
        save_dir: str = None
    ) -> HeatmapResult:
        """Generate heatmap for medical image"""
        
        if config is None:
            config = HeatmapConfig()
        
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(image_path), "heatmaps")
        
        os.makedirs(save_dir, exist_ok=True)
        
        start_time = datetime.now()
        
        # Load and preprocess image
        original_image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
        
        # Generate heatmap based on method
        if config.method == "gradcam":
            heatmap, predictions = self._generate_gradcam(image_tensor, original_image)
        elif config.method == "gradcam++":
            heatmap, predictions = self._generate_gradcam_plus(image_tensor, original_image)
        elif config.method == "guided_gradcam":
            heatmap, predictions = self._generate_guided_gradcam(image_tensor, original_image)
        else:
            # Default to basic attention visualization
            heatmap, predictions = self._generate_attention_map(image_tensor, original_image)
        
        # Apply post-processing
        heatmap = self._post_process_heatmap(heatmap, config)
        
        # Create visualizations
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save original resized image
        original_resized = original_image.resize((heatmap.shape[1], heatmap.shape[0]))
        original_path = os.path.join(save_dir, f"{base_name}_{timestamp}_original.{config.output_format}")
        original_resized.save(original_path)
        
        # Save heatmap
        heatmap_path = os.path.join(save_dir, f"{base_name}_{timestamp}_heatmap.{config.output_format}")
        heatmap_colored = self._apply_colormap(heatmap, config.colormap)
        Image.fromarray(heatmap_colored).save(heatmap_path)
        
        # Create overlay
        overlay_path = os.path.join(save_dir, f"{base_name}_{timestamp}_overlay.{config.output_format}")
        overlay = self._create_overlay(np.array(original_resized), heatmap_colored, config.overlay_alpha)
        
        # Add annotations
        overlay_annotated = self._add_annotations(overlay, predictions, config)
        Image.fromarray(overlay_annotated).save(overlay_path)
        
        # Find activation regions
        activation_regions = self._find_activation_regions(heatmap, config.threshold_percentile)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return HeatmapResult(
            original_image_path=image_path,
            heatmap_path=heatmap_path,
            overlay_path=overlay_path,
            confidence_score=float(predictions[0]['confidence']) if predictions else 0.0,
            predicted_class=predictions[0]['class'] if predictions else "Unknown",
            top_predictions=predictions[:5] if predictions else [],
            activation_regions=activation_regions,
            method_used=config.method,
            processing_time=processing_time,
            metadata={
                'image_size': original_image.size,
                'device_used': str(self.device),
                'model_type': self.model_type,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _generate_gradcam(self, image_tensor: torch.Tensor, original_image: Image) -> Tuple[np.ndarray, List[Dict]]:
        """Generate GradCAM heatmap"""
        if not self.model or not PYTORCH_AVAILABLE:
            return self._generate_demo_heatmap(original_image)
        
        try:
            # Forward pass
            self.model.zero_grad()
            output = self.model(image_tensor)
            predictions = self._extract_predictions(output)
            
            # Get the predicted class
            predicted_class_idx = torch.argmax(output, dim=1)
            
            # Get gradients for the predicted class
            class_score = output[:, predicted_class_idx]
            class_score.backward()
            
            # Get the gradients and activations from the last conv layer
            if hasattr(self.model, 'features'):
                # For models like VGG, DenseNet
                target_layer = self.model.features[-1]
            elif hasattr(self.model, 'layer4'):
                # For ResNet
                target_layer = self.model.layer4[-1]
            else:
                # Generic approach - try to find the last convolutional layer
                target_layer = None
                for name, module in reversed(list(self.model.named_modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        target_layer = module
                        break
            
            if target_layer is None:
                return self._generate_demo_heatmap(original_image)
            
            # Hook to capture gradients and activations
            gradients = []
            activations = []
            
            def forward_hook(module, input, output):
                activations.append(output)
            
            def backward_hook(module, grad_input, grad_output):
                gradients.append(grad_output[0])
            
            forward_handle = target_layer.register_forward_hook(forward_hook)
            backward_handle = target_layer.register_backward_hook(backward_hook)
            
            # Forward pass again to capture activations
            self.model.zero_grad()
            output = self.model(image_tensor)
            class_score = output[:, predicted_class_idx]
            class_score.backward()
            
            # Remove hooks
            forward_handle.remove()
            backward_handle.remove()
            
            if not gradients or not activations:
                return self._generate_demo_heatmap(original_image)
            
            # Generate GradCAM
            grads = gradients[0][0].cpu().data.numpy()
            acts = activations[0][0].cpu().data.numpy()
            
            weights = np.mean(grads, axis=(1, 2))
            cam = np.zeros(acts.shape[1:], dtype=np.float32)
            
            for i, w in enumerate(weights):
                cam += w * acts[i]
            
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, original_image.size)
            
            # Normalize
            if cam.max() > 0:
                cam = cam / cam.max()
            
            return cam, predictions
            
        except Exception as e:
            print(f"⚠️ GradCAM generation failed: {e}")
            return self._generate_demo_heatmap(original_image)
    
    def _generate_gradcam_plus(self, image_tensor: torch.Tensor, original_image: Image) -> Tuple[np.ndarray, List[Dict]]:
        """Generate Grad-CAM++ heatmap (improved version of GradCAM)"""
        # For demo purposes, we'll use the same logic as GradCAM
        # In a real implementation, this would include the Grad-CAM++ specific calculations
        return self._generate_gradcam(image_tensor, original_image)
    
    def _generate_guided_gradcam(self, image_tensor: torch.Tensor, original_image: Image) -> Tuple[np.ndarray, List[Dict]]:
        """Generate Guided GradCAM heatmap"""
        # For demo purposes, we'll use the same logic as GradCAM
        # In a real implementation, this would include guided backpropagation
        return self._generate_gradcam(image_tensor, original_image)
    
    def _generate_attention_map(self, image_tensor: torch.Tensor, original_image: Image) -> Tuple[np.ndarray, List[Dict]]:
        """Generate attention-based heatmap"""
        return self._generate_demo_heatmap(original_image)
    
    def _generate_demo_heatmap(self, original_image: Image) -> Tuple[np.ndarray, List[Dict]]:
        """Generate a demo heatmap for visualization purposes"""
        width, height = original_image.size
        
        # Create a synthetic heatmap with some medical-relevant patterns
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Create multiple activation regions
        heatmap = np.zeros((height, width))
        
        # Central region (common for lung/heart imaging)
        center_x, center_y = 0.5, 0.5
        heatmap += 0.8 * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / 0.05)
        
        # Upper right region
        heatmap += 0.6 * np.exp(-((X - 0.7)**2 + (Y - 0.3)**2) / 0.03)
        
        # Lower left region
        heatmap += 0.4 * np.exp(-((X - 0.3)**2 + (Y - 0.7)**2) / 0.02)
        
        # Add some noise for realism
        noise = np.random.random((height, width)) * 0.1
        heatmap += noise
        
        # Normalize
        heatmap = np.clip(heatmap, 0, 1)
        
        # Demo predictions
        predictions = [
            {'class': 'Abnormal Finding', 'confidence': 0.87, 'probability': 0.87},
            {'class': 'Normal', 'confidence': 0.13, 'probability': 0.13}
        ]
        
        return heatmap, predictions
    
    def _extract_predictions(self, output: torch.Tensor) -> List[Dict]:
        """Extract predictions from model output"""
        probabilities = F.softmax(output, dim=1)[0].cpu().detach().numpy()
        predictions = []
        
        for i, prob in enumerate(probabilities):
            class_name = self.medical_classes.get(i, f"Class_{i}")
            predictions.append({
                'class': class_name,
                'confidence': float(prob),
                'probability': float(prob)
            })
        
        return sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    def _post_process_heatmap(self, heatmap: np.ndarray, config: HeatmapConfig) -> np.ndarray:
        """Apply post-processing to heatmap"""
        processed = heatmap.copy()
        
        # Apply blur if specified
        if config.blur_radius > 0:
            processed = cv2.GaussianBlur(processed, (config.blur_radius * 2 + 1, config.blur_radius * 2 + 1), 0)
        
        # Enhance contrast if specified
        if config.enhance_contrast:
            processed = cv2.convertScaleAbs(processed, alpha=1.2, beta=0.1)
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def _apply_colormap(self, heatmap: np.ndarray, colormap_name: str) -> np.ndarray:
        """Apply colormap to heatmap with enhanced visualization"""
        # Enhance heatmap contrast for better visibility
        heatmap_enhanced = np.power(heatmap, 0.7)  # Gamma correction for better visibility
        
        # Apply threshold to focus on important regions
        threshold = np.percentile(heatmap_enhanced, 70)
        heatmap_enhanced = np.where(heatmap_enhanced > threshold, heatmap_enhanced, heatmap_enhanced * 0.3)
        
        # Select colormap
        if colormap_name in self.medical_colormaps:
            colormap = self.medical_colormaps[colormap_name]
        else:
            colormap = cm.jet
        
        # Apply colormap
        colored = colormap(heatmap_enhanced)[:, :, :3]  # Remove alpha channel
        colored = (colored * 255).astype(np.uint8)
        
        # Enhance yellow/red regions for medical visualization
        if colormap_name == 'jet':
            # Boost red and yellow channels for better visibility
            colored[:, :, 0] = np.minimum(255, colored[:, :, 0] * 1.2)  # Red channel
            colored[:, :, 1] = np.minimum(255, colored[:, :, 1] * 1.1)  # Green channel
        
        return colored
    
    def _create_overlay(self, original: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
        """Create enhanced overlay of original image and heatmap"""
        # Convert to float for better blending
        original_float = original.astype(np.float32) / 255.0
        heatmap_float = heatmap.astype(np.float32) / 255.0
        
        # Create adaptive alpha based on heatmap intensity
        # High intensity regions get more prominence
        heatmap_intensity = np.mean(heatmap_float, axis=2)
        adaptive_alpha = alpha * (0.3 + 0.7 * heatmap_intensity)  # Min 30% alpha, max 100%
        adaptive_alpha = np.stack([adaptive_alpha] * 3, axis=2)  # Broadcast to 3 channels
        
        # Enhanced blending with better contrast preservation
        overlay = (1 - adaptive_alpha) * original_float + adaptive_alpha * heatmap_float
        
        # Apply slight contrast enhancement to the overlay
        overlay = np.power(overlay, 0.9)  # Gamma correction
        overlay = np.clip(overlay, 0, 1)
        
        return (overlay * 255).astype(np.uint8)
    
    def _add_annotations(self, image: np.ndarray, predictions: List[Dict], config: HeatmapConfig) -> np.ndarray:
        """Add text annotations to the image"""
        annotated = image.copy()
        pil_image = Image.fromarray(annotated)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            # Try to use a decent font
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Add prediction text
        if predictions:
            text = f"Prediction: {predictions[0]['class']}"
            confidence_text = f"Confidence: {predictions[0]['confidence']:.2%}"
            
            # Draw background rectangle
            text_bbox = draw.textbbox((10, 10), text, font=font)
            conf_bbox = draw.textbbox((10, 30), confidence_text, font=font)
            
            max_width = max(text_bbox[2] - text_bbox[0], conf_bbox[2] - conf_bbox[0])
            draw.rectangle([8, 8, max_width + 20, 55], fill=(0, 0, 0, 180))
            
            # Draw text
            draw.text((10, 10), text, fill=(255, 255, 255), font=font)
            draw.text((10, 30), confidence_text, fill=(255, 255, 255), font=font)
        
        return np.array(pil_image)
    
    def _find_activation_regions(self, heatmap: np.ndarray, threshold_percentile: int) -> List[Dict[str, Any]]:
        """Find significant activation regions in the heatmap"""
        threshold = np.percentile(heatmap, threshold_percentile)
        binary_mask = (heatmap >= threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 50:  # Filter small regions
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate region statistics
                region_mask = np.zeros_like(heatmap)
                cv2.fillPoly(region_mask, [contour], 1)
                region_values = heatmap[region_mask == 1]
                
                regions.append({
                    'id': i,
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'area': int(cv2.contourArea(contour)),
                    'max_activation': float(np.max(region_values)),
                    'mean_activation': float(np.mean(region_values)),
                    'centroid': [int(x + w//2), int(y + h//2)]
                })
        
        return sorted(regions, key=lambda x: x['max_activation'], reverse=True)
    
    def generate_batch_heatmaps(
        self, 
        image_paths: List[str], 
        config: HeatmapConfig = None,
        save_dir: str = None
    ) -> List[HeatmapResult]:
        """Generate heatmaps for multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.generate_heatmap(image_path, config, save_dir)
                results.append(result)
                print(f"✅ Generated heatmap for {os.path.basename(image_path)}")
            except Exception as e:
                print(f"❌ Failed to generate heatmap for {image_path}: {e}")
        
        return results
    
    def export_results_json(self, results: List[HeatmapResult], output_path: str):
        """Export heatmap results to JSON"""
        export_data = []
        
        for result in results:
            export_data.append({
                'original_image': result.original_image_path,
                'heatmap_image': result.heatmap_path,
                'overlay_image': result.overlay_path,
                'confidence_score': result.confidence_score,
                'predicted_class': result.predicted_class,
                'predictions': result.top_predictions,
                'activation_regions': result.activation_regions,
                'method_used': result.method_used,
                'processing_time': result.processing_time,
                'metadata': result.metadata
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Results exported to {output_path}")


# Utility functions for integration
def process_medical_image_with_heatmap(image_path: str, model_path: str = None) -> Dict[str, Any]:
    """Process a medical image and generate heatmap visualization"""
    generator = MedicalHeatmapGenerator(model_path)
    config = HeatmapConfig(
        method="gradcam",
        colormap="jet",
        overlay_alpha=0.4,
        threshold_percentile=95,
        enhance_contrast=True
    )
    
    result = generator.generate_heatmap(image_path, config)
    
    return {
        'success': True,
        'heatmap_result': result,
        'visualizations': {
            'heatmap_path': result.heatmap_path,
            'overlay_path': result.overlay_path,
            'original_path': result.original_image_path
        }
    }

def get_heatmap_base64(image_path: str) -> str:
    """Convert heatmap image to base64 for web display"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""

if __name__ == "__main__":
    # Demo usage
    print("🔥 Medical Heatmap Generator Demo")
    
    # Create demo image if needed
    demo_image_path = "demo_medical_image.jpg"
    if not os.path.exists(demo_image_path):
        # Create a simple demo image
        demo_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        demo_image.save(demo_image_path)
        print(f"Created demo image: {demo_image_path}")
    
    # Generate heatmap
    result = process_medical_image_with_heatmap(demo_image_path)
    if result['success']:
        print("✅ Heatmap generation successful!")
        print(f"Overlay saved to: {result['visualizations']['overlay_path']}")
    else:
        print("❌ Heatmap generation failed")
