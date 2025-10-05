#!/usr/bin/env python3
"""
Model Configuration Manager for CortexMD
Automatically detects and configures available AI models for GradCAM/heatmap generation
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class ModelConfigurationManager:
    """Manages AI model configuration for GradCAM and heatmap generation"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize model configuration manager
        
        Args:
            base_dir: Base directory to search for models. If None, uses current directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.supported_extensions = {'.h5', '.keras', '.pth', '.pt'}
        self.models_cache = {}
        
    def scan_for_models(self) -> Dict[str, Dict[str, any]]:
        """Scan base directory for available AI models
        
        Returns:
            Dict mapping model names to their metadata
        """
        models = {}
        
        # Scan for model files
        for ext in self.supported_extensions:
            pattern = f"*{ext}"
            for model_path in self.base_dir.glob(pattern):
                if model_path.is_file():
                    model_info = self._analyze_model(model_path)
                    if model_info:
                        models[model_path.name] = model_info
        
        # Also check models subdirectory if it exists
        models_dir = self.base_dir / 'models'
        if models_dir.exists():
            for ext in self.supported_extensions:
                pattern = f"*{ext}"
                for model_path in models_dir.glob(pattern):
                    if model_path.is_file():
                        model_info = self._analyze_model(model_path)
                        if model_info:
                            models[f"models/{model_path.name}"] = model_info
        
        self.models_cache = models
        return models
    
    def _analyze_model(self, model_path: Path) -> Optional[Dict[str, any]]:
        """Analyze a model file and extract metadata
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Dictionary containing model metadata
        """
        try:
            stats = model_path.stat()
            extension = model_path.suffix.lower()
            
            # Determine framework
            framework = None
            if extension in ['.h5', '.keras']:
                framework = 'tensorflow'
            elif extension in ['.pth', '.pt']:
                framework = 'pytorch'
            
            return {
                'path': str(model_path),
                'filename': model_path.name,
                'extension': extension,
                'framework': framework,
                'size_mb': round(stats.st_size / (1024 * 1024), 2),
                'modified': stats.st_mtime,
                'exists': True
            }
        except Exception as e:
            logger.warning(f"Failed to analyze model {model_path}: {e}")
            return None
    
    def get_default_model(self) -> Optional[Dict[str, any]]:
        """Get the default model to use for GradCAM generation
        
        Returns:
            Model metadata for the default model, or None if no models found
        """
        if not self.models_cache:
            self.scan_for_models()
        
        # Priority order for default model selection
        priority_models = [
            '3d_image_classification.h5',
            'GlobalNet_pretrain20_T_0.8497.pth'
        ]
        
        # Check for priority models first
        for model_name in priority_models:
            if model_name in self.models_cache:
                return self.models_cache[model_name]
        
        # If no priority models found, return any available model
        if self.models_cache:
            return next(iter(self.models_cache.values()))
        
        return None
    
    def get_model_by_name(self, model_name: str) -> Optional[Dict[str, any]]:
        """Get model metadata by name
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Model metadata or None if not found
        """
        if not self.models_cache:
            self.scan_for_models()
        
        return self.models_cache.get(model_name)
    
    def set_environment_model(self, model_name: str) -> bool:
        """Set environment variable for GradCAM model
        
        Args:
            model_name: Name of the model to set as default
            
        Returns:
            True if successful, False otherwise
        """
        model_info = self.get_model_by_name(model_name)
        if not model_info:
            logger.error(f"Model not found: {model_name}")
            return False
        
        model_path = model_info['path']
        os.environ['GRADCAM_MODEL_PATH'] = model_path
        os.environ['GRADCAM_MODEL_NAME'] = model_name
        os.environ['GRADCAM_MODEL_FRAMEWORK'] = model_info['framework']
        
        logger.info(f"Set environment model: {model_name} ({model_path})")
        return True
    
    def get_current_environment_model(self) -> Dict[str, any]:
        """Get currently configured environment model
        
        Returns:
            Dictionary with current model configuration
        """
        model_path = os.getenv('GRADCAM_MODEL_PATH')
        model_name = os.getenv('GRADCAM_MODEL_NAME')
        framework = os.getenv('GRADCAM_MODEL_FRAMEWORK')
        
        if model_path and os.path.exists(model_path):
            return {
                'has_custom_model': True,
                'model_path': model_path,
                'model_name': model_name or os.path.basename(model_path),
                'framework': framework,
                'status': 'Custom environment model loaded'
            }
        else:
            # Check for default models
            default_model = self.get_default_model()
            if default_model:
                return {
                    'has_custom_model': False,
                    'model_path': default_model['path'],
                    'model_name': default_model['filename'],
                    'framework': default_model['framework'],
                    'status': f'Using default model: {default_model["filename"]}'
                }
            else:
                return {
                    'has_custom_model': False,
                    'model_path': None,
                    'model_name': None,
                    'framework': None,
                    'status': 'No models found'
                }
    
    def configure_optimal_model(self) -> Dict[str, any]:
        """Automatically configure the optimal available model
        
        Returns:
            Configuration result
        """
        # Scan for available models
        models = self.scan_for_models()
        
        if not models:
            return {
                'success': False,
                'message': 'No compatible models found',
                'models_found': 0
            }
        
        # Get default model
        default_model = self.get_default_model()
        if not default_model:
            return {
                'success': False,
                'message': 'No default model could be selected',
                'models_found': len(models)
            }
        
        # Set environment variables
        model_name = default_model['filename']
        success = self.set_environment_model(model_name)
        
        if success:
            return {
                'success': True,
                'message': f'Successfully configured model: {model_name}',
                'model_name': model_name,
                'model_path': default_model['path'],
                'framework': default_model['framework'],
                'models_found': len(models),
                'available_models': list(models.keys())
            }
        else:
            return {
                'success': False,
                'message': f'Failed to configure model: {model_name}',
                'models_found': len(models)
            }
    
    def export_configuration(self, output_file: str = 'model_config.json') -> bool:
        """Export current configuration to JSON file
        
        Args:
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = {
                'current_model': self.get_current_environment_model(),
                'available_models': self.scan_for_models(),
                'base_directory': str(self.base_dir),
                'supported_extensions': list(self.supported_extensions)
            }
            
            output_path = self.base_dir / output_file
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration exported to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False

# Global instance
model_config = ModelConfigurationManager()

def get_model_manager() -> ModelConfigurationManager:
    """Get the global model configuration manager instance"""
    return model_config

def auto_configure_models():
    """Automatically configure models on module import"""
    try:
        config_result = model_config.configure_optimal_model()
        if config_result['success']:
            logger.info(f"✅ {config_result['message']}")
            logger.info(f"📊 Found {config_result['models_found']} compatible models")
        else:
            logger.warning(f"⚠️ {config_result['message']}")
    except Exception as e:
        logger.error(f"❌ Failed to auto-configure models: {e}")

if __name__ == "__main__":
    # CLI usage
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "scan":
            models = model_config.scan_for_models()
            print(f"Found {len(models)} models:")
            for name, info in models.items():
                print(f"  - {name} ({info['framework']}, {info['size_mb']}MB)")
        
        elif command == "configure":
            result = model_config.configure_optimal_model()
            print(json.dumps(result, indent=2))
        
        elif command == "status":
            status = model_config.get_current_environment_model()
            print(json.dumps(status, indent=2))
        
        elif command == "export":
            output_file = sys.argv[2] if len(sys.argv) > 2 else 'model_config.json'
            success = model_config.export_configuration(output_file)
            print(f"Export {'successful' if success else 'failed'}")
        
        else:
            print("Usage: python model_config_manager.py [scan|configure|status|export]")
    else:
        # Run auto-configuration
        auto_configure_models()
