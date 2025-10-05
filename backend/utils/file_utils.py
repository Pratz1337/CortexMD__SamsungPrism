"""
File Utility Module for CortexMD
Handles file type detection and validation
"""

import os
from typing import Optional, Tuple

# Define supported file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.3gp', '.webm', '.m4v'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
DOCUMENT_EXTENSIONS = {'.json', '.pdf', '.txt', '.xml'}
DICOM_EXTENSIONS = {'.dcm', '.dicom'}

def get_file_type(filepath: str) -> Tuple[str, str]:
    """
    Determine the type of file based on its extension.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Tuple of (file_type, extension) where file_type is one of:
        'video', 'image', 'audio', 'document', 'dicom', 'unknown'
    """
    if not filepath:
        return 'unknown', ''
    
    extension = os.path.splitext(filepath)[1].lower()
    
    if extension in VIDEO_EXTENSIONS:
        return 'video', extension
    elif extension in IMAGE_EXTENSIONS:
        return 'image', extension
    elif extension in AUDIO_EXTENSIONS:
        return 'audio', extension
    elif extension in DOCUMENT_EXTENSIONS:
        return 'document', extension
    elif extension in DICOM_EXTENSIONS:
        return 'dicom', extension
    else:
        return 'unknown', extension

def is_video_file(filepath: str) -> bool:
    """Check if a file is a video format."""
    file_type, _ = get_file_type(filepath)
    return file_type == 'video'

def is_image_file(filepath: str) -> bool:
    """Check if a file is an image format."""
    file_type, _ = get_file_type(filepath)
    return file_type == 'image'

def is_audio_file(filepath: str) -> bool:
    """Check if a file is an audio format."""
    file_type, _ = get_file_type(filepath)
    return file_type == 'audio'

def is_medical_image(filepath: str) -> bool:
    """Check if a file is a medical image (including DICOM)."""
    file_type, _ = get_file_type(filepath)
    return file_type in ['image', 'dicom']

def validate_file_for_processing(filepath: str, expected_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate if a file is suitable for a specific type of processing.
    
    Args:
        filepath: Path to the file
        expected_type: Expected file type ('video', 'image', 'audio', 'document', 'dicom')
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    
    file_type, extension = get_file_type(filepath)
    
    if file_type == 'unknown':
        return False, f"Unsupported file format: {extension}"
    
    if file_type != expected_type:
        return False, f"Expected {expected_type} file, but got {file_type} file ({extension})"
    
    return True, None

def separate_files_by_type(filepaths: list) -> dict:
    """
    Separate a list of file paths by their types.
    
    Args:
        filepaths: List of file paths
        
    Returns:
        Dictionary with keys: 'videos', 'images', 'audio', 'documents', 'dicom', 'unknown'
    """
    separated = {
        'videos': [],
        'images': [],
        'audio': [],
        'documents': [],
        'dicom': [],
        'unknown': []
    }
    
    for filepath in filepaths:
        file_type, _ = get_file_type(filepath)
        if file_type == 'video':
            separated['videos'].append(filepath)
        elif file_type == 'image':
            separated['images'].append(filepath)
        elif file_type == 'audio':
            separated['audio'].append(filepath)
        elif file_type == 'document':
            separated['documents'].append(filepath)
        elif file_type == 'dicom':
            separated['dicom'].append(filepath)
        else:
            separated['unknown'].append(filepath)
    
    return separated
