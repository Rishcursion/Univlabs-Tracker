
/**
 * Custom hook for managing annotation state and operations
 */

import { useState, useCallback } from 'react';
import { Annotation, AnnotationObject } from '@/types/annotation';

interface UseAnnotationsReturn {
  annotations: Annotation[];
  objects: AnnotationObject[];
  selectedObject: string | null;
  addAnnotation: (annotation: Annotation) => void;
  removeAnnotation: (annotationId: string) => void;
  clearAllAnnotations: () => void;
  addObject: (object: AnnotationObject) => void;
  removeObject: (objectId: string) => void;
  selectObject: (objectId: string | null) => void;
  updateObjects: (objects: AnnotationObject[]) => void;
}

/**
 * Hook for managing annotation and object state
 * @returns Object containing annotation state and methods
 */
export const useAnnotations = (): UseAnnotationsReturn => {
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [objects, setObjects] = useState<AnnotationObject[]>([]);
  const [selectedObject, setSelectedObject] = useState<string | null>(null);

  const addAnnotation = useCallback((annotation: Annotation) => {
    setAnnotations(prev => [...prev, annotation]);
  }, []);

  const removeAnnotation = useCallback((annotationId: string) => {
    setAnnotations(prev => prev.filter(annotation => annotation.id !== annotationId));
  }, []);

  const clearAllAnnotations = useCallback(() => {
    setAnnotations([]);
  }, []);

  const addObject = useCallback((object: AnnotationObject) => {
    setObjects(prev => [...prev, object]);
  }, []);

  const removeObject = useCallback((objectId: string) => {
    setObjects(prev => prev.filter(obj => obj.id !== objectId));
    if (selectedObject === objectId) {
      setSelectedObject(null);
    }
  }, [selectedObject]);

  const selectObject = useCallback((objectId: string | null) => {
    setSelectedObject(objectId);
  }, []);

  const updateObjects = useCallback((newObjects: AnnotationObject[]) => {
    setObjects(newObjects);
  }, []);

  return {
    annotations,
    objects,
    selectedObject,
    addAnnotation,
    removeAnnotation,
    clearAllAnnotations,
    addObject,
    removeObject,
    selectObject,
    updateObjects,
  };
};
