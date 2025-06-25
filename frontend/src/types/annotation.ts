
/**
 * Core annotation types for the SAM2 video annotation system
 */

export interface VideoMetadata {
  name: string;
  size: number;
  duration?: number;
  format: string;
  patientName?: string;
  doctorName?: string;
  surgeryOutcome?: string;
  surgeryDate?: string;
  patientAge?: number;
  complications?: string;
  notes?: string;
}

export interface SessionData {
  sessionId: string;
  frames: string[];
  procedure: string;
  s3Link?: string;
  metadata?: VideoMetadata;
}

export interface AnnotationObject {
  id: string;
  name: string;
  color: string;
  annotations: Annotation[];
}

export interface Annotation {
  id: string;
  object_id: string;
  object_name: string;
  object_color: string;
  type: 'positive' | 'negative';
  x: number;
  y: number;
  timestamp: number;
  frame_index: number;
  created_at: string;
}

export interface AnnotationPoint {
  frameIndex: number;
  timestamp: number;
  x: number;
  y: number;
  objectId: string;
  objectName: string;
  objectColor: string;
  type: 'positive' | 'negative';
}

export interface BackendAnnotationRequest {
  sessionId: string;
  frameIndex: number;
  objectId: number;
  labels: number[];
  points: number[][];
  clearOldPoints: boolean;
  resetState: boolean;
}

export interface PropagateRequest {
  sessionId: string;
  start_frame_index: number;
}

export interface GenerateVideoRequest {
  sessionId: string;
  effect: string;
}

export interface MaskData {
  frameIndex: number;
  results: Array<{
    objectId: number;
    mask: any;
  }>;
}

/**
 * Preview image data for real-time mask display
 */
export interface PreviewImageData {
  frameIndex: number;
  base64Image: string;
  timestamp: number;
}
