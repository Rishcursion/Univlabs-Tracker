/**
 * API service layer for SAM2 annotation backend interactions (Updated for new backend)
 */
import { SessionData } from '@/types/annotation';

const API_BASE_URL = 'http://localhost:8000';

// --- Type Definitions for API communication ---

export interface BackendAnnotationRequest {
  sessionId: string;
  frameIndex: number;
  objectId: number;
  labels: number[];
  points: number[][];
  color?: string; // Add optional color property
}

export interface AnnotationResponse {
  previewImage: string; // base64 encoded image
}

export interface ProcessVideoRequest {
    sessionId: string;
}

/**
 * Creates a new annotation session with file upload
 */
export const createAnnotationSessionWithFile = async (file: File): Promise<SessionData> => {
  const formData = new FormData();
  formData.append('video_file', file);

  const response = await fetch(`${API_BASE_URL}/create_session_upload/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(`Failed to create session: ${errorData.detail || response.statusText}`);
  }

  const data = await response.json();
  return {
    sessionId: data.session_id,
    frameCount: data.frame_count,
    fps: data.fps,
    procedure: '',
  };
};

/**
 * Sends annotation points to the backend for processing
 */
export const sendAnnotationPoints = async (request: BackendAnnotationRequest): Promise<AnnotationResponse> => {
  const response = await fetch(`${API_BASE_URL}/add_new_points/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(`Failed to send annotation: ${errorData.detail || response.statusText}`);
  }
  return response.json();
};

/**
 * Triggers the backend to process the video and generate the final output
 */
export const processVideo = async (request: ProcessVideoRequest): Promise<Blob> => {
  const response = await fetch(`${API_BASE_URL}/process_video/${request.sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}), 
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(`Failed to process video: ${errorData.detail || response.statusText}`);
  }
  return response.blob();
};


/**
 * Deletes an annotation session and cleans up resources
 */
export const deleteAnnotationSession = async (sessionId: string): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/delete_session/${sessionId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Failed to delete session: ${response.statusText}`);
  }
};
