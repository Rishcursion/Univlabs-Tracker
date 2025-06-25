/**
 * Video player component with annotation overlay capabilities and real-time mask preview
 */

import React, { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Play, Pause, SkipBack, SkipForward, Loader2 } from 'lucide-react';
import { Annotation, PreviewImageData } from '@/types/annotation';

interface VideoPlayerProps {
  video: File;
  annotations: Annotation[];
  currentFrame: number;
  onTimeUpdate?: (currentTime: number, frameIndex: number) => void;
  onCanvasClick?: (x: number, y: number) => void;
  showCrosshair?: boolean;
  crosshairColor?: string;
  mousePosition?: { x: number; y: number };
  previewImage?: PreviewImageData | null;
  isProcessing?: boolean;
}

export interface VideoPlayerRef {
  play: () => void;
  pause: () => void;
  seek: (time: number) => void;
  getCurrentTime: () => number;
  getDuration: () => number;
}

/**
 * Video player component with annotation canvas overlay and real-time preview
 */
export const VideoPlayer = forwardRef<VideoPlayerRef, VideoPlayerProps>(({
  video,
  annotations,
  currentFrame,
  onTimeUpdate,
  onCanvasClick,
  showCrosshair = false,
  crosshairColor = '#3b82f6',
  mousePosition = { x: 0, y: 0 },
  previewImage,
  isProcessing = false
}, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [canvasMousePos, setCanvasMousePos] = useState({ x: 0, y: 0 });

  useImperativeHandle(ref, () => ({
    play: () => videoRef.current?.play(),
    pause: () => videoRef.current?.pause(),
    seek: (time: number) => {
      if (videoRef.current) {
        videoRef.current.currentTime = time;
      }
    },
    getCurrentTime: () => videoRef.current?.currentTime || 0,
    getDuration: () => videoRef.current?.duration || 0,
  }));

  useEffect(() => {
    if (videoRef.current && video) {
      const videoUrl = URL.createObjectURL(video);
      videoRef.current.src = videoUrl;
      
      return () => URL.revokeObjectURL(videoUrl);
    }
  }, [video]);

  useEffect(() => {
    const videoElement = videoRef.current;
    if (!videoElement) return;

    const handleLoadedMetadata = () => {
      setDuration(videoElement.duration);
      drawAnnotations();
    };

    const handleTimeUpdate = () => {
      const time = videoElement.currentTime;
      const frameIndex = Math.floor(time * 24); // Assuming 24 fps
      setCurrentTime(time);
      onTimeUpdate?.(time, frameIndex);
      drawAnnotations();
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    videoElement.addEventListener('loadedmetadata', handleLoadedMetadata);
    videoElement.addEventListener('timeupdate', handleTimeUpdate);
    videoElement.addEventListener('play', handlePlay);
    videoElement.addEventListener('pause', handlePause);

    return () => {
      videoElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
      videoElement.removeEventListener('timeupdate', handleTimeUpdate);
      videoElement.removeEventListener('play', handlePlay);
      videoElement.removeEventListener('pause', handlePause);
    };
  }, [annotations, onTimeUpdate, currentFrame]);

  // Force redraw when annotations change
  useEffect(() => {
    drawAnnotations();
  }, [annotations, currentFrame]);

  // Draw preview image on separate canvas
  useEffect(() => {
    const previewCanvas = previewCanvasRef.current;
    const video = videoRef.current;
    if (!previewCanvas || !video) return;

    const ctx = previewCanvas.getContext('2d');
    if (!ctx) return;

    previewCanvas.width = video.videoWidth || 640;
    previewCanvas.height = video.videoHeight || 480;
    ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);

    if (previewImage && previewImage.frameIndex === currentFrame) {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);
      };
      img.src = `data:image/png;base64,${previewImage.base64Image}`;
    }
  }, [previewImage, currentFrame]);

  const drawAnnotations = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw annotations for current frame (smaller points)
    const currentFrameAnnotations = annotations.filter(ann => 
      Math.abs(ann.frame_index - currentFrame) <= 1
    );

    currentFrameAnnotations.forEach((annotation) => {
      const radius = 4; // Smaller radius
      
      // Draw circle background
      ctx.fillStyle = annotation.object_color;
      ctx.beginPath();
      ctx.arc(annotation.x, annotation.y, radius, 0, 2 * Math.PI);
      ctx.fill();

      // Draw white border
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Draw plus or minus sign
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.lineCap = 'round';
      
      if (annotation.type === 'positive') {
        // Draw plus sign
        ctx.beginPath();
        ctx.moveTo(annotation.x - 2, annotation.y);
        ctx.lineTo(annotation.x + 2, annotation.y);
        ctx.moveTo(annotation.x, annotation.y - 2);
        ctx.lineTo(annotation.x, annotation.y + 2);
        ctx.stroke();
      } else {
        // Draw minus sign
        ctx.beginPath();
        ctx.moveTo(annotation.x - 2, annotation.y);
        ctx.lineTo(annotation.x + 2, annotation.y);
        ctx.stroke();
      }
    });

    // Draw crosshair lines if enabled and not playing
    if (!isPlaying && showCrosshair) {
      ctx.strokeStyle = crosshairColor;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      
      // Vertical line
      ctx.beginPath();
      ctx.moveTo(canvasMousePos.x, 0);
      ctx.lineTo(canvasMousePos.x, canvas.height);
      ctx.stroke();
      
      // Horizontal line
      ctx.beginPath();
      ctx.moveTo(0, canvasMousePos.y);
      ctx.lineTo(canvas.width, canvasMousePos.y);
      ctx.stroke();
      
      ctx.setLineDash([]);
    }
  };

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (isPlaying || !onCanvasClick) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    onCanvasClick(Math.round(x), Math.round(y));
  };

  const handleCanvasMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    setCanvasMousePos({ x: Math.round(x), y: Math.round(y) });
    drawAnnotations(); // Redraw to update crosshair
  };

  const togglePlayPause = () => {
    if (!videoRef.current) return;
    
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play();
    }
  };

  const skipTime = (seconds: number) => {
    if (!videoRef.current) return;
    
    const newTime = Math.max(0, Math.min(duration, videoRef.current.currentTime + seconds));
    videoRef.current.currentTime = newTime;
  };

  const handleSliderChange = (value: number[]) => {
    if (!videoRef.current || !duration) return;
    
    const newTime = (value[0] / 100) * duration;
    videoRef.current.currentTime = newTime;
  };

  const formatTime = (time: number): string => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-4">
      <div 
        ref={containerRef}
        className="relative bg-black rounded-lg overflow-hidden"
        style={{ cursor: !isPlaying && onCanvasClick ? 'crosshair' : 'default' }}
      >
        <video
          ref={videoRef}
          className="w-full h-auto"
          onClick={(e) => e.preventDefault()}
        />
        
        {/* Preview image overlay */}
        <canvas
          ref={previewCanvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          style={{ 
            opacity: previewImage && previewImage.frameIndex === currentFrame ? 1 : 0,
            transition: 'opacity 0.2s ease-in-out'
          }}
        />
        
        {/* Annotation points overlay */}
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-auto"
          onClick={handleCanvasClick}
          onMouseMove={handleCanvasMouseMove}
        />
        
        {/* Processing indicator */}
        {isProcessing && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
            <div className="bg-slate-800 rounded-lg p-4 flex items-center gap-3">
              <Loader2 className="h-5 w-5 animate-spin text-blue-400" />
              <span className="text-white text-sm">Processing mask...</span>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Button
            onClick={() => skipTime(-10)}
            size="sm"
            variant="outline"
            className="bg-slate-700 border-slate-600 text-white hover:bg-slate-600"
          >
            <SkipBack className="h-4 w-4" />
          </Button>
          <Button
            onClick={togglePlayPause}
            size="sm"
            className="bg-blue-600 hover:bg-blue-700"
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
          <Button
            onClick={() => skipTime(10)}
            size="sm"
            variant="outline"
            className="bg-slate-700 border-slate-600 text-white hover:bg-slate-600"
          >
            <SkipForward className="h-4 w-4" />
          </Button>
        </div>

        <div className="text-slate-400 text-sm hidden sm:block">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>
      </div>

      <div className="space-y-2">
        <Slider
          value={[duration ? (currentTime / duration) * 100 : 0]}
          onValueChange={handleSliderChange}
          max={100}
          step={0.1}
          className="w-full cursor-pointer"
        />
        <div className="text-slate-400 text-xs text-center sm:hidden">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>
      </div>
    </div>
  );
});

VideoPlayer.displayName = 'VideoPlayer';
