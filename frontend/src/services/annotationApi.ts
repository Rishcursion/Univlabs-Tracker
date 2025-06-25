/**
 * API service layer for SAM2 annotation backend interactions
 */

import { 
  SessionData, 
  BackendAnnotationRequest, 
  PropagateRequest, 
  GenerateVideoRequest,
  MaskData 
} from '@/types/annotation';

const API_BASE_URL = 'http://localhost:8000';

/**
 * Response format for add_new_points endpoint with preview image
 */
export interface AnnotationResponse {
  addPoints: {
    frameIndex: number;
    rleMaskList: any[];
    previewImage: string; // base64 encoded image
  };
}

/**
 * Creates a new annotation session with file upload
 * @param file - Video file to upload and process
 * @returns Promise containing session data
 */
export const createAnnotationSessionWithFile = async (file: File): Promise<SessionData> => {
  const formData = new FormData();
  formData.append('video_file', file);

  const response = await fetch(`${API_BASE_URL}/create_session_upload/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Failed to create session: ${response.statusText}`);
  }

  const data = await response.json();
  return {
    sessionId: data.session_id,
    frames: data.frames,
    procedure: '',
  };
};

/**
 * Creates a new annotation session with S3 link (legacy support)
 * @param s3Link - S3 URL of the video to process
 * @returns Promise containing session data
 */
export const createAnnotationSession = async (s3Link: string): Promise<SessionData> => {
  const response = await fetch(`${API_BASE_URL}/create_session/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ s3_link: s3Link }),
  });

  if (!response.ok) {
    throw new Error(`Failed to create session: ${response.statusText}`);
  }

  const data = await response.json();
  return {
    sessionId: data.session_id,
    frames: data.frames,
    procedure: '',
    s3Link: s3Link
  };
};

/**
 * Sends annotation points to the backend for processing
 * @param request - Annotation request data
 * @returns Promise containing the processed annotation response with preview image
 */
export const sendAnnotationPoints = async (request: BackendAnnotationRequest): Promise<AnnotationResponse> => {
  const response = await fetch(`${API_BASE_URL}/add_new_points/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Failed to send annotation: ${response.statusText}`);
  }

  return await response.json();
};

/**
 * Propagates annotations across video frames
 * @param request - Propagation request data
 * @returns Promise containing the response stream
 */
export const propagateAnnotations = async (request: PropagateRequest): Promise<Response> => {
  const response = await fetch(`${API_BASE_URL}/propagate_in_video`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Failed to propagate annotations: ${response.statusText}`);
  }

  return response;
};

/**
 * Generates a video with applied annotations
 * @param request - Video generation request data
 * @returns Promise containing the video blob
 */
export const generateAnnotatedVideo = async (request: GenerateVideoRequest): Promise<Blob> => {
  const response = await fetch(`${API_BASE_URL}/generate_video`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Failed to generate video: ${response.statusText}`);
  }

  return await response.blob();
};

/**
 * Deletes an annotation session and cleans up resources
 * @param sessionId - ID of the session to delete
 */
export const deleteAnnotationSession = async (sessionId: string): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/delete_session/${sessionId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Failed to delete session: ${response.statusText}`);
  }
};
