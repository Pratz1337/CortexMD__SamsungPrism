"""
Medical Imaging Data Ingestion Pipeline for CortexMD
Handles DICOM, standard image formats, and AR image processing compatibility
Enhanced with NVIDIA Clara Imaging capabilities
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import hashlib
import numpy as np

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

# Clara Imaging integration
try:
    from clara_imaging import ClaraImaging
    CLARA_AVAILABLE = True
except ImportError:
    CLARA_AVAILABLE = False
    print("Warning: Clara Imaging not available. Install Clara SDK for enhanced features.")

@dataclass
class ImageMetadata:
    """Metadata for medical images"""
    filename: str
    file_size: int
    format: str
    dimensions: Tuple[int, int]
    modality: Optional[str] = None
    study_date: Optional[str] = None
    patient_id: Optional[str] = None
    study_description: Optional[str] = None
    body_part: Optional[str] = None
    view_position: Optional[str] = None
    clara_processed: bool = False
    segmentation_data: Optional[Dict] = None
    volume_data: Optional[Dict] = None
    
@dataclass  
class ProcessedImage:
    """Processed medical image with metadata"""
    image_data: Any  # PIL Image or numpy array
    metadata: ImageMetadata
    preprocessing_applied: List[str]
    clinical_annotations: Dict[str, Any]
    quality_score: float

class MedicalImageProcessor:
    """Enhanced medical image processor with DICOM support and Clara integration"""
    
    def __init__(self):
        self.supported_formats = {
            'dicom': ['.dcm', '.dicom'],
            'standard': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
            'ar_compatible': ['.jpg', '.png']  # AR systems typically use these
        }
        
        # Initialize Clara Imaging if available
        self.clara_imaging = None
        if CLARA_AVAILABLE:
            try:
                self.clara_imaging = ClaraImaging()
                print("✅ Clara Imaging initialized successfully")
            except Exception as e:
                print(f"⚠️ Clara Imaging initialization failed: {e}")
                self.clara_imaging = None
        
        self.modality_preprocessing = {
            'CT': ['contrast_enhancement', 'noise_reduction'],
            'MR': ['contrast_enhancement', 'bias_correction'],
            'XR': ['contrast_enhancement', 'edge_enhancement'],
            'US': ['speckle_reduction', 'contrast_enhancement'],
            'CR': ['contrast_enhancement', 'edge_enhancement'],
            'DR': ['contrast_enhancement', 'edge_enhancement']
        }
        
        self.body_part_detection = {
            'chest': ['chest', 'thorax', 'lung', 'heart', 'rib'],
            'abdomen': ['abdomen', 'stomach', 'liver', 'kidney', 'pelvis'],
            'head': ['head', 'brain', 'skull', 'cranium'],
            'extremity': ['arm', 'leg', 'hand', 'foot', 'knee', 'ankle', 'wrist'],
            'spine': ['spine', 'cervical', 'thoracic', 'lumbar', 'vertebra']
        }
    
    def process_image(self, image_path: str, clinical_context: Optional[Dict[str, Any]] = None) -> ProcessedImage:
        """Process a medical image with comprehensive analysis"""
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Determine file type and extract metadata
        file_extension = os.path.splitext(image_path)[1].lower()
        
        if file_extension in self.supported_formats['dicom'] and DICOM_AVAILABLE:
            metadata, image_data = self._process_dicom(image_path)
        elif file_extension in self.supported_formats['standard']:
            metadata, image_data = self._process_standard_image(image_path)
        else:
            raise ValueError(f"Unsupported image format: {file_extension}")
        
        # Apply preprocessing based on modality
        preprocessing_applied = []
        if metadata.modality and metadata.modality in self.modality_preprocessing:
            for process in self.modality_preprocessing[metadata.modality]:
                image_data = self._apply_preprocessing(image_data, process)
                preprocessing_applied.append(process)
        
        # Extract clinical annotations from context
        clinical_annotations = self._extract_clinical_annotations(clinical_context, metadata)
        
        # Calculate quality score
        quality_score = self._calculate_image_quality(image_data, metadata)
        
        return ProcessedImage(
            image_data=image_data,
            metadata=metadata,
            preprocessing_applied=preprocessing_applied,
            clinical_annotations=clinical_annotations,
            quality_score=quality_score
        )
    
    def _process_dicom(self, dicom_path: str) -> Tuple[ImageMetadata, Any]:
        """Process DICOM file and extract metadata"""
        if not DICOM_AVAILABLE:
            raise ImportError("pydicom library required for DICOM processing")
        
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            
            # Extract DICOM metadata
            metadata = ImageMetadata(
                filename=os.path.basename(dicom_path),
                file_size=os.path.getsize(dicom_path),
                format='DICOM',
                dimensions=(int(dicom_data.Columns), int(dicom_data.Rows)),
                modality=getattr(dicom_data, 'Modality', None),
                study_date=getattr(dicom_data, 'StudyDate', None),
                patient_id=getattr(dicom_data, 'PatientID', None),
                study_description=getattr(dicom_data, 'StudyDescription', None),
                body_part=getattr(dicom_data, 'BodyPartExamined', None),
                view_position=getattr(dicom_data, 'ViewPosition', None)
            )
            
            # Convert DICOM to PIL Image
            pixel_array = dicom_data.pixel_array
            
            # Normalize pixel values to 0-255 range for PIL
            if pixel_array.max() > 255:
                pixel_array = (pixel_array / pixel_array.max() * 255).astype('uint8')
            
            # Handle different pixel representations
            if len(pixel_array.shape) == 2:  # Grayscale
                image_data = Image.fromarray(pixel_array, mode='L')
            else:  # Color
                image_data = Image.fromarray(pixel_array)
            
            return metadata, image_data
            
        except Exception as e:
            raise ValueError(f"Error processing DICOM file: {e}")
    
    def _process_standard_image(self, image_path: str) -> Tuple[ImageMetadata, Image.Image]:
        """Process standard image formats"""
        try:
            image = Image.open(image_path)
            
            # Extract basic metadata
            metadata = ImageMetadata(
                filename=os.path.basename(image_path),
                file_size=os.path.getsize(image_path),
                format=image.format or 'Unknown',
                dimensions=image.size,
                modality=self._infer_modality_from_filename(image_path),
                body_part=self._infer_body_part_from_filename(image_path)
            )
            
            # Convert to RGB if necessary
            if image.mode not in ['RGB', 'L']:
                if image.mode == 'RGBA':
                    # Create white background for RGBA images
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                else:
                    image = image.convert('RGB')
            
            return metadata, image
            
        except Exception as e:
            raise ValueError(f"Error processing image file: {e}")
    
    def _infer_modality_from_filename(self, filename: str) -> Optional[str]:
        """Infer imaging modality from filename"""
        filename_lower = filename.lower()
        
        modality_keywords = {
            'CT': ['ct', 'cat', 'computed', 'tomography'],
            'MR': ['mr', 'mri', 'magnetic', 'resonance'],
            'XR': ['xr', 'xray', 'x-ray', 'radiograph'],
            'US': ['us', 'ultrasound', 'sonogram'],
            'CR': ['cr', 'computed radiography'],
            'DR': ['dr', 'digital radiography']
        }
        
        for modality, keywords in modality_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                return modality
        
        return None
    
    def _infer_body_part_from_filename(self, filename: str) -> Optional[str]:
        """Infer body part from filename"""
        filename_lower = filename.lower()
        
        for body_part, keywords in self.body_part_detection.items():
            if any(keyword in filename_lower for keyword in keywords):
                return body_part
        
        return None
    
    def _apply_preprocessing(self, image: Image.Image, process_type: str) -> Image.Image:
        """Apply specific preprocessing to image"""
        
        if process_type == 'contrast_enhancement':
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.2)
        
        elif process_type == 'noise_reduction':
            return image.filter(ImageFilter.MedianFilter(size=3))
        
        elif process_type == 'edge_enhancement':
            return image.filter(ImageFilter.EDGE_ENHANCE)
        
        elif process_type == 'bias_correction':
            # Simple bias correction using histogram equalization
            if image.mode == 'L':
                import PIL.ImageOps
                return PIL.ImageOps.equalize(image)
            else:
                # For color images, apply to each channel
                if image.mode == 'RGB':
                    r, g, b = image.split()
                    r = PIL.ImageOps.equalize(r)
                    g = PIL.ImageOps.equalize(g) 
                    b = PIL.ImageOps.equalize(b)
                    return Image.merge('RGB', (r, g, b))
        
        elif process_type == 'speckle_reduction':
            # Apply Gaussian blur for speckle reduction in ultrasound
            return image.filter(ImageFilter.GaussianBlur(radius=1))
        
        return image
    
    def _extract_clinical_annotations(self, clinical_context: Optional[Dict[str, Any]], 
                                    metadata: ImageMetadata) -> Dict[str, Any]:
        """Extract clinical annotations from context and metadata"""
        annotations = {
            'modality': metadata.modality,
            'body_part': metadata.body_part,
            'study_description': metadata.study_description,
            'view_position': metadata.view_position
        }
        
        if clinical_context:
            # Extract relevant clinical information
            if 'symptoms' in clinical_context:
                annotations['relevant_symptoms'] = self._match_symptoms_to_imaging(
                    clinical_context['symptoms'], metadata.body_part
                )
            
            if 'indication' in clinical_context:
                annotations['clinical_indication'] = clinical_context['indication']
            
            if 'findings' in clinical_context:
                annotations['reported_findings'] = clinical_context['findings']
        
        return annotations
    
    def _match_symptoms_to_imaging(self, symptoms: List[str], body_part: Optional[str]) -> List[str]:
        """Match symptoms to imaging body part"""
        if not body_part:
            return symptoms
        
        relevant_symptoms = []
        
        symptom_body_part_mapping = {
            'chest': ['chest pain', 'shortness of breath', 'cough', 'palpitations'],
            'abdomen': ['abdominal pain', 'nausea', 'vomiting', 'diarrhea'],
            'head': ['headache', 'dizziness', 'confusion', 'visual changes'],
            'extremity': ['joint pain', 'swelling', 'weakness', 'numbness'],
            'spine': ['back pain', 'neck pain', 'weakness', 'numbness']
        }
        
        relevant_keywords = symptom_body_part_mapping.get(body_part, [])
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            if any(keyword in symptom_lower for keyword in relevant_keywords):
                relevant_symptoms.append(symptom)
        
        return relevant_symptoms if relevant_symptoms else symptoms
    
    def _calculate_image_quality(self, image: Image.Image, metadata: ImageMetadata) -> float:
        """Calculate image quality score"""
        quality_score = 1.0
        
        # Size quality (larger is generally better for medical imaging)
        width, height = image.size
        pixel_count = width * height
        
        if pixel_count < 100000:  # Less than 0.1 megapixels
            quality_score -= 0.3
        elif pixel_count < 500000:  # Less than 0.5 megapixels
            quality_score -= 0.1
        
        # Format quality
        if metadata.format in ['JPEG', 'JPG']:
            quality_score -= 0.1  # JPEG compression can lose medical detail
        
        # Modality-specific quality checks
        if metadata.modality == 'DICOM':
            quality_score += 0.1  # DICOM is preferred for medical imaging
        
        return max(0.0, min(1.0, quality_score))
    
    def prepare_for_ar_viewing(self, processed_image: ProcessedImage) -> Dict[str, Any]:
        """Prepare image for AR/mixed reality viewing"""
        
        image = processed_image.image_data
        
        # Convert to AR-compatible format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create multiple resolutions for AR optimization
        ar_assets = {
            'high_res': image,
            'medium_res': image.resize((int(image.width * 0.5), int(image.height * 0.5)), Image.Resampling.LANCZOS),
            'low_res': image.resize((int(image.width * 0.25), int(image.height * 0.25)), Image.Resampling.LANCZOS),
            'thumbnail': image.resize((128, 128), Image.Resampling.LANCZOS)
        }
        
        # Extract key clinical annotations for AR overlay
        ar_annotations = {
            'modality': processed_image.metadata.modality,
            'body_part': processed_image.metadata.body_part,
            'study_date': processed_image.metadata.study_date,
            'quality_score': processed_image.quality_score,
            'relevant_symptoms': processed_image.clinical_annotations.get('relevant_symptoms', []),
            'clinical_indication': processed_image.clinical_annotations.get('clinical_indication', ''),
            'preprocessing_applied': processed_image.preprocessing_applied
        }
        
        return {
            'assets': ar_assets,
            'annotations': ar_annotations,
            'metadata': {
                'original_dimensions': processed_image.metadata.dimensions,
                'file_format': processed_image.metadata.format,
                'processing_timestamp': datetime.now().isoformat()
            }
        }
    
    def generate_imaging_report(self, processed_images: List[ProcessedImage]) -> Dict[str, Any]:
        """Generate comprehensive imaging report"""
        
        report = {
            'study_summary': {
                'total_images': len(processed_images),
                'modalities': list(set(img.metadata.modality for img in processed_images if img.metadata.modality)),
                'body_parts': list(set(img.metadata.body_part for img in processed_images if img.metadata.body_part)),
                'average_quality': sum(img.quality_score for img in processed_images) / len(processed_images) if processed_images else 0
            },
            'image_details': [],
            'clinical_correlation': {},
            'preprocessing_summary': {}
        }
        
        # Detailed image information
        for i, img in enumerate(processed_images):
            report['image_details'].append({
                'image_index': i,
                'filename': img.metadata.filename,
                'modality': img.metadata.modality,
                'body_part': img.metadata.body_part,
                'dimensions': img.metadata.dimensions,
                'quality_score': img.quality_score,
                'preprocessing_applied': img.preprocessing_applied,
                'clinical_annotations': img.clinical_annotations
            })
        
        # Clinical correlation summary
        all_symptoms = []
        for img in processed_images:
            symptoms = img.clinical_annotations.get('relevant_symptoms', [])
            all_symptoms.extend(symptoms)
        
        report['clinical_correlation'] = {
            'symptoms_addressed': list(set(all_symptoms)),
            'imaging_coverage': self._assess_imaging_coverage(processed_images),
            'recommendations': self._generate_imaging_recommendations(processed_images)
        }
        
        # Preprocessing summary
        all_preprocessing = []
        for img in processed_images:
            all_preprocessing.extend(img.preprocessing_applied)
        
        from collections import Counter
        preprocessing_counts = Counter(all_preprocessing)
        report['preprocessing_summary'] = dict(preprocessing_counts)
        
        return report
    
    def _assess_imaging_coverage(self, processed_images: List[ProcessedImage]) -> Dict[str, Any]:
        """Assess how well imaging covers clinical presentation"""
        
        body_parts_imaged = set(img.metadata.body_part for img in processed_images if img.metadata.body_part)
        modalities_used = set(img.metadata.modality for img in processed_images if img.metadata.modality)
        
        coverage_assessment = {
            'body_parts_covered': list(body_parts_imaged),
            'modalities_used': list(modalities_used),
            'comprehensive_coverage': len(body_parts_imaged) >= 2 and len(modalities_used) >= 1,
            'quality_adequate': all(img.quality_score >= 0.7 for img in processed_images)
        }
        
        return coverage_assessment
    
    def _generate_imaging_recommendations(self, processed_images: List[ProcessedImage]) -> List[str]:
        """Generate imaging recommendations based on current studies"""
        
        recommendations = []
        
        # Quality-based recommendations
        low_quality_images = [img for img in processed_images if img.quality_score < 0.7]
        if low_quality_images:
            recommendations.append("Consider retaking images with low quality scores")
        
        # Modality-specific recommendations
        modalities = set(img.metadata.modality for img in processed_images if img.metadata.modality)
        
        if 'XR' in modalities and len([img for img in processed_images if img.metadata.modality == 'XR']) == 1:
            recommendations.append("Consider additional views for complete radiographic evaluation")
        
        if not modalities:
            recommendations.append("Unable to determine imaging modality - verify image metadata")
        
        # Coverage recommendations
        body_parts = set(img.metadata.body_part for img in processed_images if img.metadata.body_part)
        if len(body_parts) == 1 and len(processed_images) > 1:
            recommendations.append("Multiple images of same body part - ensure adequate coverage of pathology")
        
        return recommendations if recommendations else ["Imaging study appears adequate for clinical evaluation"]

    def process_with_clara(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Process medical image using NVIDIA Clara"""
        if not self.clara_imaging:
            return None
            
        try:
            # Enhanced DICOM processing with Clara
            processed_data = self.clara_imaging.process_dicom(image_path)
            return {
                'processed': True,
                'data': processed_data,
                'enhanced': True
            }
        except Exception as e:
            print(f"Clara processing failed: {e}")
            return None

    def perform_3d_reconstruction(self, dicom_data: Any) -> Optional[Dict[str, Any]]:
        """Perform 3D volume reconstruction using Clara"""
        if not self.clara_imaging:
            return None
            
        try:
            volume_data = self.clara_imaging.reconstruct_3d(dicom_data)
            return {
                'volume_rendered': True,
                'volume_data': volume_data,
                'reconstruction_method': 'clara_3d'
            }
        except Exception as e:
            print(f"3D reconstruction failed: {e}")
            return None

    def segment_medical_image(self, dicom_data: Any) -> Optional[Dict[str, Any]]:
        """Perform advanced image segmentation using Clara"""
        if not self.clara_imaging:
            return None
            
        try:
            segmentation_data = self.clara_imaging.segment_image(dicom_data)
            return {
                'segmented': True,
                'segmentation_data': segmentation_data,
                'organs_identified': True,
                'pathology_detected': True
            }
        except Exception as e:
            print(f"Image segmentation failed: {e}")
            return None

    def enhanced_process_image(self, image_path: str) -> ProcessedImage:
        """Enhanced image processing with Clara integration"""
        # First process with standard pipeline
        processed_image = self.process_image(image_path)
        
        # Enhance with Clara if available
        if self.clara_imaging and processed_image:
            try:
                # Clara DICOM processing
                clara_result = self.process_with_clara(image_path)
                if clara_result:
                    processed_image.metadata.clara_processed = True
                    processed_image.preprocessing_applied.append('clara_enhanced')
                
                # 3D reconstruction for DICOM
                if processed_image.metadata.format.lower() in ['dicom', 'dcm']:
                    volume_result = self.perform_3d_reconstruction(processed_image.image_data)
                    if volume_result:
                        processed_image.metadata.volume_data = volume_result
                        processed_image.preprocessing_applied.append('3d_reconstruction')
                
                # Image segmentation
                segmentation_result = self.segment_medical_image(processed_image.image_data)
                if segmentation_result:
                    processed_image.metadata.segmentation_data = segmentation_result
                    processed_image.preprocessing_applied.append('clara_segmentation')
                    
            except Exception as e:
                print(f"Clara enhancement failed: {e}")
        
        return processed_image

    def generate_heatmap_visualization(self, image_path: str, model_path: str = None) -> Dict[str, Any]:
        """Generate heatmap visualization for medical image analysis"""
        try:
            from medical_heatmap import MedicalHeatmapGenerator, HeatmapConfig
            
            # Initialize heatmap generator
            generator = MedicalHeatmapGenerator(model_path)
            
            # Configure heatmap generation
            config = HeatmapConfig(
                method="gradcam",
                colormap="jet",
                overlay_alpha=0.4,
                threshold_percentile=95,
                enhance_contrast=True,
                save_individual_maps=True
            )
            
            # Generate heatmap
            result = generator.generate_heatmap(image_path, config)
            
            return {
                'success': True,
                'heatmap_result': result,
                'visualizations': {
                    'heatmap_path': result.heatmap_path,
                    'overlay_path': result.overlay_path,
                    'original_path': result.original_image_path
                },
                'analysis': {
                    'confidence_score': result.confidence_score,
                    'predicted_class': result.predicted_class,
                    'activation_regions': result.activation_regions,
                    'processing_time': result.processing_time
                }
            }
            
        except ImportError as e:
            return {
                'success': False,
                'error': f'Heatmap dependencies not available: {e}',
                'visualizations': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Heatmap generation failed: {e}',
                'visualizations': None
            }

class DICOMValidator:
    """Validate DICOM files for medical imaging pipeline"""
    
    def __init__(self):
        self.required_tags = [
            'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 
            'Modality', 'StudyDate', 'Rows', 'Columns'
        ]
        
        self.recommended_tags = [
            'PatientName', 'StudyDescription', 'SeriesDescription',
            'BodyPartExamined', 'ViewPosition', 'ImageType'
        ]
    
    def validate_dicom(self, dicom_path: str) -> Dict[str, Any]:
        """Validate DICOM file completeness and integrity"""
        
        if not DICOM_AVAILABLE:
            return {'error': 'pydicom library not available'}
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'missing_required': [],
            'missing_recommended': []
        }
        
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            
            # Check required tags
            for tag in self.required_tags:
                if not hasattr(dicom_data, tag) or not getattr(dicom_data, tag):
                    validation_result['missing_required'].append(tag)
                    validation_result['valid'] = False
            
            # Check recommended tags
            for tag in self.recommended_tags:
                if not hasattr(dicom_data, tag) or not getattr(dicom_data, tag):
                    validation_result['missing_recommended'].append(tag)
            
            # Check image data integrity
            try:
                pixel_array = dicom_data.pixel_array
                if pixel_array.size == 0:
                    validation_result['errors'].append("Empty pixel data")
                    validation_result['valid'] = False
            except:
                validation_result['errors'].append("Cannot read pixel data")
                validation_result['valid'] = False
            
            # Check for warnings
            if validation_result['missing_recommended']:
                validation_result['warnings'].append(f"Missing recommended tags: {', '.join(validation_result['missing_recommended'])}")
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"DICOM read error: {str(e)}")
        
        return validation_result
