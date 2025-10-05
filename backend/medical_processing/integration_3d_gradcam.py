#!/usr/bin/env python3
"""
Integration module for 3D GradCAM with the existing CortexMD backend.
Provides API endpoints and service functions for 3D medical image heatmap generation.
"""

import os
import sys
import json
import base64
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from io import BytesIO
from PIL import Image

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def integrate_3d_gradcam_with_diagnosis(
    image_files: List[str],
    model_path: str = "3d_image_classification.h5",
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Integrate 3D GradCAM generation with the main diagnosis pipeline.
    
    Args:
        image_files: List of paths to medical image files
        model_path: Path to the 3D CNN model
        output_dir: Directory to save heatmap outputs
    
    Returns:
        Dictionary containing heatmap results and metadata
    """
    
    if output_dir is None:
        output_dir = os.path.join("uploads", f"heatmaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔥 Starting 3D GradCAM integration for {len(image_files)} files")
    
    try:
        from medical_3d_gradcam import Medical3DGradCAMGenerator, GradCAM3DConfig, load_3d_medical_volume
        
        # Check if model exists
        if not os.path.exists(model_path):
            return {
                'success': False,
                'error': f'3D model not found: {model_path}',
                'heatmap_data': []
            }
        
        # Initialize generator
        generator = Medical3DGradCAMGenerator(model_path)
        
        # Configure for medical use
        config = GradCAM3DConfig(
            colormap="hot",
            overlay_alpha=0.6,
            threshold_percentile=85,
            slice_visualization="middle",
            enhance_contrast=True,
            save_individual_slices=True
        )
        
        heatmap_results = []
        
        for i, image_file in enumerate(image_files):
            print(f"📊 Processing image {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
            
            try:
                # Load image/volume
                volume_data = load_3d_medical_volume(image_file)
                
                # Generate GradCAM
                result = generator.generate_3d_gradcam(
                    volume_data=volume_data,
                    config=config,
                    save_dir=os.path.join(output_dir, f"image_{i}")
                )
                
                # Convert result to serializable format
                heatmap_data = {
                    'success': True,
                    'image_index': i,
                    'original_file': image_file,
                    'analysis': {
                        'predicted_class': result.predicted_class,
                        'confidence_score': result.confidence_score,
                        'processing_time': result.processing_time,
                        'activation_regions_count': len(result.activation_regions)
                    },
                    'predictions': result.top_predictions,
                    'visualizations': {
                        'heatmap_slices': result.heatmap_slices,
                        'overlay_slices': result.overlay_slices,
                        'volume_visualization': result.volume_visualization
                    },
                    'activation_regions': result.activation_regions[:10],  # Top 10 regions
                    'metadata': result.metadata
                }
                
                # Convert images to base64 for web display
                heatmap_data['base64_images'] = _convert_images_to_base64(result)
                
                # Fallback: Look for generated volume files in output directory if base64 conversion failed
                if not heatmap_data.get('base64_images') or not any(heatmap_data['base64_images'].values()):
                    print(f"🔄 Attempting fallback base64 conversion from output directory: {output_dir}")
                    fallback_base64 = _fallback_base64_conversion(output_dir, image_file)
                    if fallback_base64:
                        heatmap_data['base64_images'] = fallback_base64
                        print(f"✅ Fallback conversion successful: {len(fallback_base64)} images")
                
                heatmap_results.append(heatmap_data)
                
                print(f"   ✅ Success - Prediction: {result.predicted_class} ({result.confidence_score:.2%})")
                
            except Exception as e:
                error_data = {
                    'success': False,
                    'image_index': i,
                    'original_file': image_file,
                    'error': str(e),
                    'analysis': None,
                    'visualizations': None
                }
                heatmap_results.append(error_data)
                print(f"   ❌ Failed: {e}")
        
        # Summary
        successful_heatmaps = len([r for r in heatmap_results if r['success']])
        
        return {
            'success': True,
            'total_images': len(image_files),
            'successful_heatmaps': successful_heatmaps,
            'output_directory': output_dir,
            'heatmap_data': heatmap_results,
            'model_info': {
                'model_path': model_path,
                'input_shape': generator.model_input_shape,
                'model_type': '3D CNN'
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'3D GradCAM integration failed: {str(e)}',
            'heatmap_data': []
        }

def _convert_images_to_base64(result) -> Dict[str, str]:
    """Convert result images to base64 for web display"""
    base64_images = {}
    
    try:
        print(f"🔍 DEBUG - Converting images to base64:")
        print(f"   - Result type: {type(result)}")
        print(f"   - Has heatmap_slices: {hasattr(result, 'heatmap_slices')}")
        print(f"   - Has overlay_slices: {hasattr(result, 'overlay_slices')}")
        print(f"   - Has volume_visualization: {hasattr(result, 'volume_visualization')}")
        
        # Convert main heatmap slice
        if hasattr(result, 'heatmap_slices') and result.heatmap_slices:
            heatmap_file = result.heatmap_slices[0]
            print(f"   - Heatmap file: {heatmap_file}")
            print(f"   - Heatmap file exists: {os.path.exists(heatmap_file)}")
            if os.path.exists(heatmap_file):
                with open(heatmap_file, 'rb') as f:
                    base64_data = base64.b64encode(f.read()).decode('utf-8')
                    base64_images['heatmap'] = base64_data
                    print(f"   - Heatmap base64 length: {len(base64_data)}")
        
        # Convert main overlay slice
        if hasattr(result, 'overlay_slices') and result.overlay_slices:
            overlay_file = result.overlay_slices[0]
            print(f"   - Overlay file: {overlay_file}")
            print(f"   - Overlay file exists: {os.path.exists(overlay_file)}")
            if os.path.exists(overlay_file):
                with open(overlay_file, 'rb') as f:
                    base64_data = base64.b64encode(f.read()).decode('utf-8')
                    base64_images['overlay'] = base64_data
                    print(f"   - Overlay base64 length: {len(base64_data)}")
        
        # Convert volume visualization
        if hasattr(result, 'volume_visualization') and result.volume_visualization and os.path.exists(result.volume_visualization):
            volume_file = result.volume_visualization
            print(f"   - Volume file: {volume_file}")
            print(f"   - Volume file exists: {os.path.exists(volume_file)}")
            with open(volume_file, 'rb') as f:
                base64_data = base64.b64encode(f.read()).decode('utf-8')
                base64_images['volume'] = base64_data
                print(f"   - Volume base64 length: {len(base64_data)}")
        
        print(f"   - Total base64 images created: {len(base64_images)}")
                
    except Exception as e:
        print(f"⚠️ Error converting images to base64: {e}")
        import traceback
        traceback.print_exc()
    
    return base64_images

def _fallback_base64_conversion(output_dir: str, image_file: str = None) -> Dict[str, str]:
    """Fallback method to convert generated images to base64 by searching the output directory"""
    base64_images = {}
    
    try:
        print(f"🔍 Searching for images in: {output_dir}")
        
        if not os.path.exists(output_dir):
            print(f"   - Output directory doesn't exist: {output_dir}")
            return base64_images
        
        # Look for PNG files in the output directory
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    print(f"   - Found PNG file: {file_path}")
                    
                    try:
                        with open(file_path, 'rb') as f:
                            base64_data = base64.b64encode(f.read()).decode('utf-8')
                            
                            # Categorize based on filename
                            if 'volume_3d' in file.lower():
                                base64_images['volume'] = base64_data
                                print(f"   - Added volume image: {len(base64_data)} chars")
                            elif 'heatmap' in file.lower() or 'gradcam' in file.lower():
                                base64_images['heatmap'] = base64_data
                                print(f"   - Added heatmap image: {len(base64_data)} chars")
                            elif 'overlay' in file.lower():
                                base64_images['overlay'] = base64_data
                                print(f"   - Added overlay image: {len(base64_data)} chars")
                            else:
                                # Default to volume if we can't categorize
                                if 'volume' not in base64_images:
                                    base64_images['volume'] = base64_data
                                    print(f"   - Added as volume (default): {len(base64_data)} chars")
                                    
                    except Exception as e:
                        print(f"   - Error reading {file_path}: {e}")
        
        print(f"🔄 Fallback conversion complete: {len(base64_images)} images found")
        return base64_images
        
    except Exception as e:
        print(f"⚠️ Fallback base64 conversion failed: {e}")
        return base64_images

def create_heatmap_api_response(heatmap_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create API response format for heatmap results that matches existing backend structure
    """
    
    if not heatmap_results['success']:
        return {
            'success': False,
            'error': heatmap_results.get('error', 'Unknown error'),
            'heatmap_visualization': {
                'available': False,
                'error': heatmap_results.get('error', 'Unknown error')
            }
        }
    
    # Create response in the format expected by the frontend
    response = {
        'success': True,
        'heatmap_visualization': {
            'available': True,
            'total_images': heatmap_results['total_images'],
            'successful_heatmaps': heatmap_results['successful_heatmaps'],
            'model_type': '3D CNN',
            'generation_time': datetime.now().isoformat()
        },
        'heatmap_data': []
    }
    
    # Process each heatmap result
    for heatmap_data in heatmap_results['heatmap_data']:
        if heatmap_data['success']:
            api_heatmap = {
                'success': True,
                'image_file': os.path.basename(heatmap_data['original_file']),
                'analysis': heatmap_data['analysis'],
                'visualizations': {
                    'heatmap_image': heatmap_data['base64_images'].get('heatmap', ''),
                    'overlay_image': heatmap_data['base64_images'].get('overlay', ''),
                    'volume_image': heatmap_data['base64_images'].get('volume', ''),
                    'file_paths': {
                        'heatmap_slices': heatmap_data['visualizations']['heatmap_slices'],
                        'overlay_slices': heatmap_data['visualizations']['overlay_slices'],
                        'volume_visualization': heatmap_data['visualizations']['volume_visualization']
                    }
                },
                'predictions': heatmap_data['predictions'],
                'activation_regions': heatmap_data['activation_regions'],
                'medical_interpretation': _generate_medical_interpretation(heatmap_data)
            }
        else:
            api_heatmap = {
                'success': False,
                'image_file': os.path.basename(heatmap_data['original_file']),
                'error': heatmap_data['error']
            }
        
        response['heatmap_data'].append(api_heatmap)
    
    return response

def _generate_medical_interpretation(heatmap_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate medical interpretation of the heatmap results"""
    
    analysis = heatmap_data['analysis']
    predictions = heatmap_data['predictions']
    regions = heatmap_data['activation_regions']
    
    interpretation = {
        'primary_finding': predictions[0]['class'] if predictions else 'Unknown',
        'confidence_level': _categorize_confidence(analysis['confidence_score']),
        'attention_areas': len(regions),
        'clinical_notes': []
    }
    
    # Generate clinical notes based on findings
    if analysis['predicted_class'].lower() == 'abnormal':
        interpretation['clinical_notes'].append(
            f"Model detected abnormal findings with {analysis['confidence_score']:.1%} confidence"
        )
        interpretation['clinical_notes'].append(
            f"Found {len(regions)} regions of high attention"
        )
        
        if regions:
            max_activation = max(r['max_activation'] for r in regions)
            interpretation['clinical_notes'].append(
                f"Highest activation region shows {max_activation:.3f} intensity"
            )
    else:
        interpretation['clinical_notes'].append(
            f"Model suggests normal findings with {analysis['confidence_score']:.1%} confidence"
        )
    
    interpretation['clinical_notes'].append(
        f"Analysis completed in {analysis['processing_time']:.2f} seconds"
    )
    
    return interpretation

def _categorize_confidence(confidence: float) -> str:
    """Categorize confidence score into clinical terms"""
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.8:
        return "High"
    elif confidence >= 0.7:
        return "Moderate"
    elif confidence >= 0.6:
        return "Low"
    else:
        return "Very Low"

def flask_api_endpoint_3d_gradcam():
    """
    Flask API endpoint for 3D GradCAM generation.
    This function would be integrated into the main Flask app.
    """
    from flask import request, jsonify
    
    try:
        # Get uploaded files
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        
        # Save uploaded files temporarily
        temp_files = []
        upload_dir = "uploads/temp_3d_gradcam"
        os.makedirs(upload_dir, exist_ok=True)
        
        for file in files:
            if file.filename:
                file_path = os.path.join(upload_dir, file.filename)
                file.save(file_path)
                temp_files.append(file_path)
        
        # Get model path (default or from request)
        model_path = request.form.get('model_path', '3d_image_classification.h5')
        
        # Generate 3D GradCAM
        heatmap_results = integrate_3d_gradcam_with_diagnosis(
            image_files=temp_files,
            model_path=model_path
        )
        
        # Create API response
        response = create_heatmap_api_response(heatmap_results)
        
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'API endpoint error: {str(e)}'
        }), 500


    
    
 

if __name__ == "__main__":
    print("🏥 3D GradCAM Integration Module")
    




