/**
 * Main annotation canvas component with video player and real-time mask preview
 */

import React, { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { VideoPlayer, VideoPlayerRef } from '@/components/VideoPlayer';
import { AnnotationsList } from '@/components/AnnotationsList';
import { sendAnnotationPoints, AnnotationResponse } from '@/services/annotationApi';
import { usePreviewImages } from '@/hooks/usePreviewImages';
import { 
  Annotation, 
  AnnotationObject, 
  SessionData, 
  BackendAnnotationRequest 
} from '@/types/annotation';

interface AnnotationCanvasProps {
  video: File;
  selectedObject: string | null;
  annotationMode: 'positive' | 'negative';
  objects: AnnotationObject[];
  annotations: Annotation[];
  onAnnotationsChange: (annotations: Annotation[]) => void;
  sessionData: SessionData;
}

/**
 * Main component for video annotation with canvas overlay and real-time preview
 */
export const AnnotationCanvas: React.FC<AnnotationCanvasProps> = ({
  video,
  selectedObject,
  annotationMode,
  objects,
  annotations,
  onAnnotationsChange,
  sessionData
}) => {
  const videoPlayerRef = useRef<VideoPlayerRef>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Preview image management
  const { previewImages, setPreviewImage, getPreviewImage } = usePreviewImages();

  // Default object for when no objects are created
  const defaultObject: AnnotationObject = {
    id: '0',
    name: 'Default Object',
    color: '#3b82f6',
    annotations: []
  };

  /**
   * Checks if there are any positive annotations for the current object
   */
  const hasPositiveAnnotations = (): boolean => {
    const targetObjectId = selectedObject || defaultObject.id;
    return annotations.some(ann => ann.object_id === targetObjectId && ann.type === 'positive');
  };

  /**
   * Determines if negative annotations can be added
   */
  const canAddNegativeAnnotation = (): boolean => {
    return annotationMode === 'positive' || hasPositiveAnnotations();
  };

  /**
   * Sends annotation data to the FastAPI backend with real-time preview
   */
  const sendAnnotationToBackend = async (
    points: number[][], 
    labels: number[], 
    objectId: string
  ): Promise<void> => {
    if (!sessionData?.sessionId) {
      console.error('No session ID available');
      return;
    }

    try {
      setIsProcessing(true);
      
      const request: BackendAnnotationRequest = {
        sessionId: sessionData.sessionId,
        frameIndex: currentFrame,
        objectId: parseInt(objectId) || 1,
        labels: labels,
        points: points,
        clearOldPoints: true,
        resetState: false
      };

      const result: AnnotationResponse = await sendAnnotationPoints(request);
      console.log('Backend annotation response:', result);
      
      // Store the preview image for real-time display
      if (result.addPoints?.previewImage) {
        setPreviewImage(currentFrame, result.addPoints.previewImage);
      }
      
    } catch (error) {
      console.error('Error sending annotation to backend:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  /**
   * Handles video time updates
   */
  const handleTimeUpdate = (time: number, frameIndex: number): void => {
    setCurrentTime(time);
    setCurrentFrame(frameIndex);
  };

  /**
   * Handles canvas click events for annotation creation with debouncing
   */
  const handleCanvasClick = async (x: number, y: number): Promise<void> => {
    if (!canAddNegativeAnnotation() || isProcessing) {
      return;
    }

    const targetObjectId = selectedObject || defaultObject.id;
    const targetObject = objects.find(obj => obj.id === targetObjectId) || defaultObject;

    const newAnnotation: Annotation = {
      id: Date.now().toString(),
      object_id: targetObjectId,
      object_name: targetObject.name,
      object_color: targetObject.color,
      type: annotationMode,
      x: x,
      y: y,
      timestamp: currentTime,
      frame_index: currentFrame,
      created_at: new Date().toISOString()
    };

    // Update local state immediately for responsive UI
    const updatedAnnotations = [...annotations, newAnnotation];
    onAnnotationsChange(updatedAnnotations);

    // Prepare data for backend
    const objectAnnotations = updatedAnnotations.filter(ann => 
      ann.object_id === targetObjectId && 
      Math.abs(ann.frame_index - currentFrame) <= 1
    );

    const points = objectAnnotations.map(ann => [ann.x, ann.y]);
    const labels = objectAnnotations.map(ann => ann.type === 'positive' ? 1 : 0);

    // Send to backend for real-time preview
    await sendAnnotationToBackend(points, labels, targetObjectId);
  };

  /**
   * Deletes an annotation
   */
  const deleteAnnotation = (annotationId: string): void => {
    const updatedAnnotations = annotations.filter(ann => ann.id !== annotationId);
    onAnnotationsChange(updatedAnnotations);
  };

  /**
   * Seeks video to annotation timestamp
   */
  const seekToAnnotation = (timestamp: number): void => {
    videoPlayerRef.current?.seek(timestamp);
  };

  /**
   * Clears all annotations
   */
  const clearAllAnnotations = (): void => {
    onAnnotationsChange([]);
  };

  const getCrosshairColor = (): string => {
    return annotationMode === 'positive' ? '#22c55e' : '#ef4444';
  };

  const currentPreview = getPreviewImage(currentFrame);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Video Player Section */}
      <div className="lg:col-span-2">
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="flex items-center justify-between text-white">
              <span>Annotation Canvas</span>
              <div className="text-sm text-slate-400">
                Frame: {currentFrame} | Coordinates: ({mousePos.x}, {mousePos.y})
                {currentPreview && <span className="ml-2 text-green-400">• Preview Active</span>}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <VideoPlayer
              ref={videoPlayerRef}
              video={video}
              annotations={annotations}
              currentFrame={currentFrame}
              onTimeUpdate={handleTimeUpdate}
              onCanvasClick={handleCanvasClick}
              showCrosshair={canAddNegativeAnnotation() && !isProcessing}
              crosshairColor={getCrosshairColor()}
              mousePosition={mousePos}
              previewImage={currentPreview}
              isProcessing={isProcessing}
            />

            {/* Validation Messages */}
            {annotationMode === 'negative' && !hasPositiveAnnotations() && (
              <div className="text-red-400 text-sm p-3 bg-red-400/10 rounded border border-red-400/20">
                Cannot add negative annotations without positive annotations first. 
                Please add positive points before negative ones.
              </div>
            )}

            {sessionData && (
              <div className="text-green-400 text-sm p-3 bg-green-400/10 rounded border border-green-400/20">
                Session ID: {sessionData.sessionId} | Backend connected
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Annotations List Section */}
      <div className="lg:col-span-1">
        <AnnotationsList
          annotations={annotations}
          objects={objects}
          onDeleteAnnotation={deleteAnnotation}
          onSeekToAnnotation={seekToAnnotation}
          onClearAllAnnotations={clearAllAnnotations}
        />
      </div>
    </div>
  );
};
