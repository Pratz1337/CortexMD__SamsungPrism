"""
Video Processing Module for CortexMD
Handles MP4, AVI, MOV and other video formats with frame extraction and medical analysis
"""

import os
import cv2
import json
import tempfile
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from PIL import Image, ImageEnhance, ImageFilter
import hashlib
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """Metadata for medical videos"""
    filename: str
    file_size: int
    format: str
    duration: float
    fps: float
    total_frames: int
    dimensions: Tuple[int, int]
    video_type: Optional[str] = None  # ultrasound, endoscopy, xray_motion, mri_sequence, general
    study_date: Optional[str] = None
    patient_id: Optional[str] = None
    study_description: Optional[str] = None
    body_part: Optional[str] = None
    view_position: Optional[str] = None

@dataclass
class VideoFrame:
    """Individual video frame with analysis data"""
    frame_number: int
    timestamp: float
    image_data: np.ndarray
    analysis: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    is_key_frame: bool = False
    motion_score: Optional[float] = None

@dataclass
class ProcessedVideo:
    """Processed medical video with metadata and extracted frames"""
    video_path: str
    metadata: VideoMetadata
    key_frames: List[VideoFrame]
    temporal_analysis: Dict[str, Any]
    quality_metrics: Dict[str, float]
    medical_findings: Dict[str, Any]
    processing_log: List[str]

class MedicalVideoProcessor:
    """Enhanced medical video processor with frame extraction and analysis"""
    
    def __init__(self):
        self.supported_formats = {
            'video': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.3gp', '.webm', '.m4v'],
            'medical_types': {
                'ultrasound': ['ultrasound', 'us', 'echo', 'doppler'],
                'endoscopy': ['endoscopy', 'scope', 'colonoscopy', 'gastroscopy'],
                'xray_motion': ['fluoroscopy', 'xray', 'fluoro', 'motion'],
                'mri_sequence': ['mri', 'fmri', 'dti', 'sequence'],
                'general': ['medical', 'clinical', 'procedure']
            }
        }
        
        # Frame extraction settings
        self.extraction_settings = {
            'max_key_frames': 20,
            'motion_threshold': 0.3,
            'quality_threshold': 0.5,
            'min_frame_interval': 1.0,  # seconds
            'max_frame_interval': 10.0  # seconds
        }
    
    def is_video_file(self, filepath: str) -> bool:
        """Check if file is a supported video format"""
        if not os.path.exists(filepath):
            return False
        
        file_extension = os.path.splitext(filepath)[1].lower()
        return file_extension in self.supported_formats['video']
    
    def process_video(self, video_path: str, analysis_type: str = 'general', 
                     clinical_context: Optional[Dict[str, Any]] = None) -> ProcessedVideo:
        """Process a medical video with comprehensive analysis"""
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not self.is_video_file(video_path):
            file_extension = os.path.splitext(video_path)[1].lower()
            raise ValueError(f"Unsupported video format: {file_extension}")
        
        processing_log = []
        processing_log.append(f"Starting video processing: {video_path}")
        
        try:
            # Extract video metadata
            metadata = self._extract_video_metadata(video_path, analysis_type)
            processing_log.append(f"Extracted metadata: {metadata.duration}s, {metadata.fps} fps, {metadata.total_frames} frames")
            
            # Extract key frames
            key_frames = self._extract_key_frames(video_path, metadata)
            processing_log.append(f"Extracted {len(key_frames)} key frames")
            
            # Perform temporal analysis
            temporal_analysis = self._analyze_temporal_patterns(key_frames, metadata)
            processing_log.append("Completed temporal analysis")
            
            # Calculate quality metrics
            quality_metrics = self._calculate_video_quality(key_frames, metadata)
            processing_log.append("Calculated quality metrics")
            
            # Perform medical analysis based on video type
            medical_findings = self._analyze_medical_content(key_frames, analysis_type, clinical_context)
            processing_log.append(f"Completed {analysis_type} medical analysis")
            
            return ProcessedVideo(
                video_path=video_path,
                metadata=metadata,
                key_frames=key_frames,
                temporal_analysis=temporal_analysis,
                quality_metrics=quality_metrics,
                medical_findings=medical_findings,
                processing_log=processing_log
            )
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            raise ValueError(f"Error processing video file: {str(e)}")
    
    def _extract_video_metadata(self, video_path: str, analysis_type: str) -> VideoMetadata:
        """Extract comprehensive metadata from video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Get file info
            filename = os.path.basename(video_path)
            file_size = os.path.getsize(video_path)
            file_format = os.path.splitext(video_path)[1].lower()
            
            return VideoMetadata(
                filename=filename,
                file_size=file_size,
                format=file_format,
                duration=duration,
                fps=fps,
                total_frames=frame_count,
                dimensions=(width, height),
                video_type=analysis_type,
                study_date=datetime.now().isoformat(),
                patient_id=None  # Will be set externally
            )
            
        except Exception as e:
            raise ValueError(f"Error extracting video metadata: {str(e)}")
    
    def _extract_key_frames(self, video_path: str, metadata: VideoMetadata) -> List[VideoFrame]:
        """Extract key frames from video using motion detection and quality analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            key_frames = []
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video for frame extraction: {video_path}")
            
            # Calculate frame sampling strategy
            total_frames = metadata.total_frames
            max_frames = min(self.extraction_settings['max_key_frames'], total_frames)
            
            if total_frames <= max_frames:
                # Extract all frames if video is short
                frame_indices = list(range(0, total_frames, max(1, total_frames // max_frames)))
            else:
                # Smart sampling: always include first, last, and evenly spaced frames
                frame_indices = []
                frame_indices.append(0)  # First frame
                frame_indices.append(total_frames - 1)  # Last frame
                
                # Add evenly spaced frames
                step = total_frames // (max_frames - 2)
                for i in range(step, total_frames - step, step):
                    frame_indices.append(i)
                
                frame_indices = sorted(set(frame_indices))[:max_frames]
            
            # Extract frames
            prev_frame = None
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    timestamp = frame_idx / metadata.fps
                    
                    # Calculate motion score
                    motion_score = 0.0
                    if prev_frame is not None:
                        motion_score = self._calculate_motion_score(prev_frame, frame)
                    
                    # Calculate quality score
                    quality_score = self._calculate_frame_quality(frame)
                    
                    # Determine if this is a key frame
                    is_key_frame = (
                        frame_idx == 0 or  # First frame
                        frame_idx == total_frames - 1 or  # Last frame
                        motion_score > self.extraction_settings['motion_threshold'] or
                        quality_score > self.extraction_settings['quality_threshold']
                    )
                    
                    key_frames.append(VideoFrame(
                        frame_number=frame_idx,
                        timestamp=timestamp,
                        image_data=frame.copy(),
                        quality_score=quality_score,
                        is_key_frame=is_key_frame,
                        motion_score=motion_score
                    ))
                    
                    prev_frame = frame.copy()
            
            cap.release()
            
            # Sort by timestamp and return top quality frames if we have too many
            key_frames.sort(key=lambda x: x.timestamp)
            
            if len(key_frames) > self.extraction_settings['max_key_frames']:
                # Keep the highest quality frames
                key_frames.sort(key=lambda x: (x.quality_score or 0), reverse=True)
                key_frames = key_frames[:self.extraction_settings['max_key_frames']]
                key_frames.sort(key=lambda x: x.timestamp)  # Re-sort by time
            
            return key_frames
            
        except Exception as e:
            raise ValueError(f"Error extracting key frames: {str(e)}")
    
    def _calculate_motion_score(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate motion score between two frames"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)
            
            # Calculate motion score as normalized mean difference
            motion_score = np.mean(diff) / 255.0
            
            return motion_score
            
        except Exception as e:
            logger.warning(f"Error calculating motion score: {e}")
            return 0.0
    
    def _calculate_frame_quality(self, frame: np.ndarray) -> float:
        """Calculate quality score for a frame"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (focus measure)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray)
            
            # Calculate brightness (mean)
            brightness = np.mean(gray)
            
            # Combine metrics into quality score
            # Normalize and weight the metrics
            focus_score = min(laplacian_var / 1000.0, 1.0)  # Normalize focus
            contrast_score = min(contrast / 128.0, 1.0)  # Normalize contrast
            brightness_score = 1.0 - abs(brightness - 128) / 128.0  # Optimal brightness around 128
            
            quality_score = (focus_score * 0.5 + contrast_score * 0.3 + brightness_score * 0.2)
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating frame quality: {e}")
            return 0.5
    
    def _analyze_temporal_patterns(self, key_frames: List[VideoFrame], metadata: VideoMetadata) -> Dict[str, Any]:
        """Analyze temporal patterns in the video"""
        if not key_frames:
            return {'motion_patterns': [], 'changes_detected': []}
        
        motion_patterns = []
        changes_detected = []
        
        # Analyze motion patterns
        motion_scores = [frame.motion_score or 0 for frame in key_frames[1:]]
        if motion_scores:
            avg_motion = np.mean(motion_scores)
            motion_patterns.append(f"Average motion level: {avg_motion:.2f}")
            
            if avg_motion > 0.5:
                motion_patterns.append("High motion activity detected")
            elif avg_motion > 0.2:
                motion_patterns.append("Moderate motion activity detected")
            else:
                motion_patterns.append("Low motion activity detected")
        
        # Detect significant changes
        for i, frame in enumerate(key_frames):
            if frame.motion_score and frame.motion_score > self.extraction_settings['motion_threshold']:
                severity = 'high' if frame.motion_score > 0.6 else 'medium'
                changes_detected.append({
                    'timestamp': frame.timestamp,
                    'description': f'Significant motion detected (score: {frame.motion_score:.2f})',
                    'severity': severity
                })
        
        return {
            'motion_patterns': motion_patterns,
            'changes_detected': changes_detected,
            'total_key_frames': len(key_frames),
            'duration_analyzed': metadata.duration
        }
    
    def _calculate_video_quality(self, key_frames: List[VideoFrame], metadata: VideoMetadata) -> Dict[str, float]:
        """Calculate overall video quality metrics"""
        if not key_frames:
            return {'overall_quality': 0.0, 'average_frame_quality': 0.0}
        
        quality_scores = [frame.quality_score or 0 for frame in key_frames]
        motion_scores = [frame.motion_score or 0 for frame in key_frames if frame.motion_score]
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        avg_motion = np.mean(motion_scores) if motion_scores else 0.0
        
        # Calculate resolution quality
        width, height = metadata.dimensions
        resolution_score = min((width * height) / (1920 * 1080), 1.0)  # Normalize to 1080p
        
        # Calculate frame rate quality
        fps_score = min(metadata.fps / 30.0, 1.0)  # Normalize to 30fps
        
        # Overall quality combines multiple factors
        overall_quality = (
            avg_quality * 0.4 +
            resolution_score * 0.3 +
            fps_score * 0.2 +
            min(avg_motion, 1.0) * 0.1
        )
        
        return {
            'overall_quality': overall_quality,
            'average_frame_quality': avg_quality,
            'average_motion': avg_motion,
            'resolution_score': resolution_score,
            'fps_score': fps_score,
            'frames_analyzed': len(key_frames)
        }
    
    def _analyze_medical_content(self, key_frames: List[VideoFrame], analysis_type: str, 
                               clinical_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze medical content based on video type"""
        
        findings = {
            'abnormalities': [],
            'normal_findings': [],
            'analysis_type': analysis_type,
            'confidence_level': 'moderate'  # This would be enhanced with actual AI models
        }
        
        # Basic analysis based on video type
        if analysis_type == 'ultrasound':
            findings['normal_findings'].append('Video frames suitable for ultrasound analysis')
            findings['analysis_notes'] = 'Ultrasound video processed - recommend radiologist review'
            
        elif analysis_type == 'endoscopy':
            findings['normal_findings'].append('Endoscopic video frames extracted successfully')
            findings['analysis_notes'] = 'Endoscopy video processed - recommend gastroenterologist review'
            
        elif analysis_type == 'xray_motion':
            findings['normal_findings'].append('Fluoroscopy motion study frames captured')
            findings['analysis_notes'] = 'Motion study processed - recommend radiologist review'
            
        elif analysis_type == 'mri_sequence':
            findings['normal_findings'].append('MRI sequence frames extracted for analysis')
            findings['analysis_notes'] = 'MRI sequence processed - recommend radiologist review'
            
        else:  # general
            findings['normal_findings'].append('Medical video frames extracted successfully')
            findings['analysis_notes'] = 'General medical video processed - recommend specialist review'
        
        # Add frame-based observations
        high_quality_frames = len([f for f in key_frames if (f.quality_score or 0) > 0.7])
        findings['quality_assessment'] = f'{high_quality_frames}/{len(key_frames)} frames are high quality'
        
        return findings
    
    def save_key_frames(self, processed_video: ProcessedVideo, output_dir: str) -> List[str]:
        """Save extracted key frames as individual images"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        base_name = os.path.splitext(processed_video.metadata.filename)[0]
        
        for i, frame in enumerate(processed_video.key_frames):
            timestamp_str = f"{frame.timestamp:.2f}s"
            filename = f"{base_name}_frame_{i:03d}_{timestamp_str.replace('.', '_')}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            try:
                # Convert BGR to RGB for PIL
                rgb_frame = cv2.cvtColor(frame.image_data, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_image.save(filepath, 'JPEG', quality=95)
                saved_paths.append(filepath)
                
            except Exception as e:
                logger.error(f"Error saving frame {i}: {e}")
        
        return saved_paths
    
    def get_frame_at_timestamp(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """Extract a specific frame at given timestamp"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            cap.release()
            
            return frame if ret else None
            
        except Exception as e:
            logger.error(f"Error extracting frame at timestamp {timestamp}: {e}")
            return None

# Convenience functions
def is_video_file(filepath: str) -> bool:
    """Check if file is a supported video format"""
    processor = MedicalVideoProcessor()
    return processor.is_video_file(filepath)

def process_medical_video(video_path: str, analysis_type: str = 'general', 
                         clinical_context: Optional[Dict[str, Any]] = None) -> ProcessedVideo:
    """Process a medical video file"""
    processor = MedicalVideoProcessor()
    return processor.process_video(video_path, analysis_type, clinical_context)

def extract_video_frames(video_path: str, max_frames: int = 10) -> List[Dict[str, Any]]:
    """Extract key frames from video and return as list of frame data"""
    processor = MedicalVideoProcessor()
    
    # Temporarily adjust settings
    original_max = processor.extraction_settings['max_key_frames']
    processor.extraction_settings['max_key_frames'] = max_frames
    
    try:
        processed_video = processor.process_video(video_path)
        frames_data = []
        
        for frame in processed_video.key_frames:
            # Convert frame to base64 for API response
            rgb_frame = cv2.cvtColor(frame.image_data, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            frames_data.append({
                'frame_number': frame.frame_number,
                'timestamp': frame.timestamp,
                'quality_score': frame.quality_score,
                'motion_score': frame.motion_score,
                'is_key_frame': frame.is_key_frame,
                'dimensions': pil_image.size
            })
        
        return frames_data
        
    finally:
        # Restore original settings
        processor.extraction_settings['max_key_frames'] = original_max
