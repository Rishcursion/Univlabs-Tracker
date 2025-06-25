
/**
 * Custom hook for managing real-time mask preview images
 */

import { useState, useCallback } from 'react';
import { PreviewImageData } from '@/types/annotation';

interface UsePreviewImagesReturn {
  previewImages: Map<number, PreviewImageData>;
  setPreviewImage: (frameIndex: number, base64Image: string) => void;
  getPreviewImage: (frameIndex: number) => PreviewImageData | null;
  clearPreviewImage: (frameIndex: number) => void;
  clearAllPreviews: () => void;
}

/**
 * Hook for managing preview image cache and state
 */
export const usePreviewImages = (): UsePreviewImagesReturn => {
  const [previewImages, setPreviewImages] = useState<Map<number, PreviewImageData>>(new Map());

  const setPreviewImage = useCallback((frameIndex: number, base64Image: string) => {
    setPreviewImages(prev => {
      const newMap = new Map(prev);
      newMap.set(frameIndex, {
        frameIndex,
        base64Image,
        timestamp: Date.now()
      });
      return newMap;
    });
  }, []);

  const getPreviewImage = useCallback((frameIndex: number): PreviewImageData | null => {
    return previewImages.get(frameIndex) || null;
  }, [previewImages]);

  const clearPreviewImage = useCallback((frameIndex: number) => {
    setPreviewImages(prev => {
      const newMap = new Map(prev);
      newMap.delete(frameIndex);
      return newMap;
    });
  }, []);

  const clearAllPreviews = useCallback(() => {
    setPreviewImages(new Map());
  }, []);

  return {
    previewImages,
    setPreviewImage,
    getPreviewImage,
    clearPreviewImage,
    clearAllPreviews,
  };
};
