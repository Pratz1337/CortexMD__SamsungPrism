// Camera utility functions for AR scanner

export const getCameraDevices = async (): Promise<MediaDeviceInfo[]> => {
  try {
    if (!navigator.mediaDevices?.enumerateDevices) {
      return [];
    }
    
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter(device => device.kind === 'videoinput');
  } catch (error) {
    console.error('🎥 Failed to enumerate camera devices:', error);
    return [];
  }
};

export const testCameraAccess = async (): Promise<{
  supported: boolean;
  devices: MediaDeviceInfo[];
  error?: string;
}> => {
  try {
    // Check basic API support
    if (!navigator.mediaDevices?.getUserMedia) {
      return {
        supported: false,
        devices: [],
        error: 'Camera API not supported in this browser'
      };
    }

    // Get available devices
    const devices = await getCameraDevices();
    
    if (devices.length === 0) {
      return {
        supported: false,
        devices: [],
        error: 'No camera devices found'
      };
    }

    // Test basic camera access
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      // Clean up immediately
      stream.getTracks().forEach(track => track.stop());
      
      return {
        supported: true,
        devices,
      };
    } catch (accessError: any) {
      return {
        supported: false,
        devices,
        error: `Camera access failed: ${accessError.message}`
      };
    }
  } catch (error: any) {
    return {
      supported: false,
      devices: [],
      error: `Camera test failed: ${error.message}`
    };
  }
};

export const getOptimalConstraints = (preferBackCamera: boolean = false) => {
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  
  return {
    video: {
      width: { ideal: isMobile ? 1920 : 1280, min: 640, max: 1920 },
      height: { ideal: isMobile ? 1080 : 720, min: 480, max: 1080 },
      facingMode: preferBackCamera || isMobile ? 'environment' : 'user',
      aspectRatio: { ideal: 16/9 },
      frameRate: { ideal: 30, min: 15 }
    },
    audio: false
  };
};

export const detectBrowserIssues = () => {
  const userAgent = navigator.userAgent;
  const issues: string[] = [];

  // Check for known browser issues
  if (userAgent.includes('Safari') && !userAgent.includes('Chrome')) {
    issues.push('Safari may require HTTPS for camera access');
  }
  
  if (userAgent.includes('Firefox')) {
    issues.push('Firefox may have autoplay restrictions');
  }
  
  if (userAgent.includes('Chrome') && location.protocol !== 'https:' && location.hostname !== 'localhost') {
    issues.push('Chrome requires HTTPS for camera access in production');
  }

  // Check for mobile-specific issues
  if (/iPhone|iPad/.test(userAgent)) {
    issues.push('iOS devices may require user interaction to start camera');
  }

  return issues;
};

export const logCameraState = (video: HTMLVideoElement, stream?: MediaStream) => {
  console.log('🎥 Camera State Debug:', {
    // Video element state
    videoElement: {
      readyState: video.readyState,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      paused: video.paused,
      ended: video.ended,
      muted: video.muted,
      autoplay: video.autoplay,
      playsInline: video.playsInline,
      srcObject: !!video.srcObject
    },
    // Stream state
    stream: stream ? {
      id: stream.id,
      active: stream.active,
      tracks: stream.getTracks().map(track => ({
        kind: track.kind,
        enabled: track.enabled,
        readyState: track.readyState,
        label: track.label
      }))
    } : null,
    // Browser info
    browser: {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      vendor: navigator.vendor,
      protocol: location.protocol,
      hostname: location.hostname
    }
  });
};
