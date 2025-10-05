import { api } from '@/lib/api';

export interface VideoFrame {
  timestamp: number;
  frameNumber: number;
  analysis: {
    detected_features: string[];
    confidence: number;
    roi_coordinates?: { x: number; y: number; width: number; height: number }[];
  };
}

export interface VideoAnalysisResult {
  video_id: string;
  duration: number;
  fps: number;
  total_frames: number;
  key_frames: VideoFrame[];
  temporal_analysis: {
    motion_patterns: string[];
    changes_detected: Array<{
      timestamp: number;
      description: string;
      severity: 'low' | 'medium' | 'high';
    }>;
  };
  medical_findings: {
    abnormalities: Array<{
      type: string;
      confidence: number;
      frame_range: [number, number];
      description: string;
    }>;
    normal_findings: string[];
  };
  xai_explanation: {
    attention_maps: Array<{
      frame: number;
      heatmap_url: string;
      regions_of_interest: Array<{
        label: string;
        importance: number;
        coordinates: { x: number; y: number; width: number; height: number };
      }>;
    }>;
    decision_path: string[];
    feature_importance: Record<string, number>;
  };
}

export class VideoAnalysisService {
  static async analyzeVideo(
    file: File,
    patientId: string,
    analysisType: 'ultrasound' | 'endoscopy' | 'xray_motion' | 'mri_sequence' | 'general'
  ): Promise<VideoAnalysisResult> {
    const formData = new FormData();
    formData.append('video', file);
    formData.append('patient_id', patientId);
    formData.append('analysis_type', analysisType);

    const response = await api.post('/api/video/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000, // 5 minutes for video processing
      onUploadProgress: (progressEvent) => {
        const progress = progressEvent.total
          ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
          : 0;
        console.log('Upload progress:', progress);
      },
    });

    return response.data;
  }

  static async extractKeyFrames(
    file: File,
    options?: {
      maxFrames?: number;
      motionThreshold?: number;
      includeFirst?: boolean;
      includeLast?: boolean;
    }
  ): Promise<Array<{ frame: Blob; timestamp: number; index: number }>> {
    const formData = new FormData();
    formData.append('video', file);
    if (options?.maxFrames) formData.append('max_frames', options.maxFrames.toString());
    if (options?.motionThreshold) formData.append('motion_threshold', options.motionThreshold.toString());
    if (options?.includeFirst !== undefined) formData.append('include_first', options.includeFirst.toString());
    if (options?.includeLast !== undefined) formData.append('include_last', options.includeLast.toString());

    const response = await api.post('/api/video/extract-frames', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      responseType: 'blob',
    });

    // Parse the multipart response
    const frames = await this.parseMultipartFrames(response.data);
    return frames;
  }

  static async compareVideoFrames(
    videoFile: File,
    referenceImages: File[],
    comparisonType: 'similarity' | 'progression' | 'anomaly'
  ): Promise<{
    comparisons: Array<{
      frame_timestamp: number;
      reference_image: string;
      similarity_score: number;
      differences: string[];
      xai_explanation: string;
    }>;
    overall_assessment: string;
  }> {
    const formData = new FormData();
    formData.append('video', videoFile);
    referenceImages.forEach((img, index) => {
      formData.append(`reference_${index}`, img);
    });
    formData.append('comparison_type', comparisonType);

    const response = await api.post('/api/video/compare', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    return response.data;
  }

  static async generateVideoReport(
    analysisResult: VideoAnalysisResult,
    includeFrames: boolean = true
  ): Promise<{
    report_url: string;
    summary: string;
    key_findings: string[];
  }> {
    const response = await api.post('/api/video/generate-report', {
      analysis_result: analysisResult,
      include_frames: includeFrames,
    });

    return response.data;
  }

  static async getVideoAnnotations(
    videoId: string
  ): Promise<Array<{
    timestamp: number;
    annotation: string;
    author: string;
    created_at: string;
  }>> {
    const response = await api.get(`/api/video/${videoId}/annotations`);
    return response.data;
  }

  static async addVideoAnnotation(
    videoId: string,
    timestamp: number,
    annotation: string
  ): Promise<void> {
    await api.post(`/api/video/${videoId}/annotations`, {
      timestamp,
      annotation,
    });
  }

  private static async parseMultipartFrames(blob: Blob): Promise<Array<{ frame: Blob; timestamp: number; index: number }>> {
    // Implementation to parse multipart response containing multiple frames
    // This is a simplified version - actual implementation would parse multipart boundaries
    const frames: Array<{ frame: Blob; timestamp: number; index: number }> = [];
    
    // For now, return a placeholder
    return frames;
  }
}
