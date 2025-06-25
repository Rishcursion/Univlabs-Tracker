
/**
 * Component for displaying and managing the list of annotations
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Trash2 } from 'lucide-react';
import { Annotation, AnnotationObject } from '@/types/annotation';

interface AnnotationsListProps {
  annotations: Annotation[];
  objects: AnnotationObject[];
  onDeleteAnnotation: (annotationId: string) => void;
  onSeekToAnnotation: (timestamp: number) => void;
  onClearAllAnnotations: () => void;
}

/**
 * Component for displaying and managing annotation list
 */
export const AnnotationsList: React.FC<AnnotationsListProps> = ({
  annotations,
  objects,
  onDeleteAnnotation,
  onSeekToAnnotation,
  onClearAllAnnotations
}) => {
  const defaultObject: AnnotationObject = {
    id: '0',
    name: 'Default Object',
    color: '#3b82f6',
    annotations: []
  };

  const formatTime = (time: number): string => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <Card className="bg-slate-800 border-slate-700">
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-white">
          <span>Annotations ({annotations.length})</span>
          {annotations.length > 0 && (
            <Button
              onClick={onClearAllAnnotations}
              size="sm"
              variant="destructive"
              className="text-xs"
            >
              Clear All
            </Button>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {annotations.length === 0 ? (
          <div className="text-slate-400 text-sm text-center py-4">
            No annotations yet. Click on the video to add points.
          </div>
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {annotations.map((annotation, index) => {
              const obj = objects.find(obj => obj.id === annotation.object_id) || defaultObject;
              return (
                <div
                  key={annotation.id || index}
                  className="bg-slate-700 p-3 rounded-lg"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-3 h-3 rounded-full ${
                          annotation.type === 'positive' ? 'bg-green-500' : 'bg-red-500'
                        }`}
                      />
                      <span className="text-sm text-white">
                        {annotation.type === 'positive' ? 'Positive' : 'Negative'}
                      </span>
                    </div>
                    <Button
                      onClick={() => onDeleteAnnotation(annotation.id || index.toString())}
                      size="sm"
                      variant="ghost"
                      className="text-red-400 hover:text-red-300 p-1"
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                  
                  <div className="text-xs text-slate-400 mb-2">
                    <div>Object: {obj.name}</div>
                    <div>Position: ({annotation.x}, {annotation.y})</div>
                    <div>Frame: {annotation.frame_index}</div>
                    <div>Time: {formatTime(annotation.timestamp)}</div>
                  </div>
                  
                  <Button
                    onClick={() => onSeekToAnnotation(annotation.timestamp)}
                    size="sm"
                    variant="outline"
                    className="w-full bg-slate-600 border-slate-500 text-white hover:bg-slate-500 text-xs"
                  >
                    Seek to Time
                  </Button>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
};
