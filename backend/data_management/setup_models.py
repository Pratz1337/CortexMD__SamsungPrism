#!/usr/bin/env python3
"""
CortexMD Model Setup and Configuration Script
Automatically detects and configures available AI models for optimal GradCAM/heatmap generation
"""

import os
import sys
import logging
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from model_config_manager import ModelConfigurationManager

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main setup function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🤖 CortexMD AI Model Setup & Configuration")
    print("=" * 50)
    
    # Initialize model configuration manager
    model_manager = ModelConfigurationManager(str(backend_dir))
    
    print("🔍 Step 1: Scanning for available models...")
    models = model_manager.scan_for_models()
    
    if not models:
        print("❌ No compatible models found!")
        print("   Please ensure you have at least one of the following model files:")
        print("   - 3d_image_classification.h5 (TensorFlow/Keras)")
        print("   - GlobalNet_pretrain20_T_0.8497.pth (PyTorch)")
        print("   - Any custom .h5, .keras, .pth, or .pt model files")
        return False
    
    print(f"✅ Found {len(models)} compatible models:")
    for name, info in models.items():
        print(f"   - {name} ({info['framework']}, {info['size_mb']}MB)")
    
    print("\n🎯 Step 2: Configuring optimal model...")
    config_result = model_manager.configure_optimal_model()
    
    if config_result['success']:
        print(f"✅ {config_result['message']}")
        print(f"   Model: {config_result['model_name']}")
        print(f"   Framework: {config_result['framework']}")
        print(f"   Path: {config_result['model_path']}")
    else:
        print(f"❌ {config_result['message']}")
        return False
    
    print("\n🔧 Step 3: Verifying environment configuration...")
    current_config = model_manager.get_current_environment_model()
    
    print(f"   Status: {current_config['status']}")
    print(f"   Model: {current_config['model_name']}")
    print(f"   Framework: {current_config['framework']}")
    
    print("\n📄 Step 4: Exporting configuration...")
    export_success = model_manager.export_configuration('model_config.json')
    
    if export_success:
        print("✅ Configuration exported to model_config.json")
    else:
        print("⚠️ Failed to export configuration")
    
    print("\n🚀 Setup Complete!")
    print("   Your CortexMD application is now configured with optimal AI models")
    print("   for GradCAM heatmap generation.")
    print("\n💡 Next Steps:")
    print("   1. Start the backend server: python app.py")
    print("   2. Start the frontend: cd frontend && npm run dev")
    print("   3. Use the Model Configuration UI to switch models if needed")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
