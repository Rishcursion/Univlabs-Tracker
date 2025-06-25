/**
 * Preview player component for SAM2 processing and video generation
 */

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Eye, EyeOff, Play, Download, CheckCircle } from 'lucide-react';
import { 
  propagateAnnotations, 
  generateAnnotatedVideo,
  deleteAnnotationSession 
} from '@/services/annotationApi';
import { SessionData, MaskData } from '@/types/annotation';
import { useToast } from '@/hooks/use-toast';

interface PreviewPlayerProps {
  show: boolean;
  onToggleShow: (show: boolean) => void;
  sessionData: SessionData;
}

/**
 * Component for processing annotations and generating preview videos
 */
export const PreviewPlayer: React.FC<PreviewPlayerProps> = ({
  show,
  onToggleShow,
  sessionData
}) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [maskData, setMaskData] = useState<MaskData | null>(null);
  const [videoBlob, setVideoBlob] = useState<string | null>(null);
  const [isPropagationComplete, setIsPropagationComplete] = useState(false);
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false);
  const { toast } = useToast();
  
  /**
   * Deletes the current session and cleans up resources
   */
  const handleDeleteSession = async (): Promise<void> => {
    if (!sessionData?.sessionId) return;

    try {
      await deleteAnnotationSession(sessionData.sessionId);
      
      // Reset local state
      setMaskData(null);
      setVideoBlob(null);
      setIsPropagationComplete(false);
      
      toast({
        title: "Session Deleted",
        description: "Session and associated files have been cleaned up.",
      });

      // Optionally reload the page or redirect
      window.location.reload();
      
    } catch (error) {
      console.error('Error deleting session:', error);
      toast({
        title: "Delete Failed",
        description: "Failed to delete session. Please try again.",
        variant: "destructive"
      });
    }
  };

  /**
   * Propagates annotations across video frames using SAM2
   */
  const handlePropagateAnnotations = async (): Promise<void> => {
    if (!sessionData?.sessionId) return;

    setIsProcessing(true);
    setIsPropagationComplete(false);
    setVideoBlob(null); // Clear any previous video
    
    try {
      const response = await propagateAnnotations({
        sessionId: sessionData.sessionId,
        start_frame_index: 0
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let frameCount = 0;

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const frames = chunk.split('frameseparator').filter(f => f.trim());
          
          for (const frame of frames) {
            try {
              const frameData = JSON.parse(frame);
              console.log('Received frame data:', frameData);
              setMaskData(frameData);
              frameCount++;
            } catch (e) {
              console.error('Error parsing frame data:', e);
            }
          }
        }
      }

      // Mark propagation as complete
      setIsPropagationComplete(true);
      toast({
        title: "Propagation Complete",
        description: `Successfully processed ${frameCount} frames. You can now generate the video.`,
      });

    } catch (error) {
      console.error('Error propagating annotations:', error);
      toast({
        title: "Propagation Failed",
        description: "Failed to propagate annotations. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsProcessing(false);
    }
  };

  /**
   * Generates annotated video with mask effects
   */
  const handleGenerateVideo = async (): Promise<void> => {
    if (!sessionData?.sessionId || !isPropagationComplete) return;

    setIsGeneratingVideo(true);
    try {
      const blob = await generateAnnotatedVideo({
        sessionId: sessionData.sessionId,
        effect: 'mask'
      });

      const videoUrl = URL.createObjectURL(blob);
      setVideoBlob(videoUrl);

      toast({
        title: "Video Generated",
        description: "Your annotated video is ready for download!",
      });

    } catch (error) {
      console.error('Error generating video:', error);
      toast({
        title: "Video Generation Failed",
        description: "Failed to generate video. Please ensure propagation was completed first.",
        variant: "destructive"
      });
    } finally {
      setIsGeneratingVideo(false);
    }
  };

  /**
   * Downloads the generated video
   */
  const downloadVideo = (): void => {
    if (!videoBlob) return;

    const a = document.createElement('a');
    a.href = videoBlob;
    a.download = 'annotated_video.webm';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  if (!show) {
    return (
      <div className="flex items-center justify-center p-4">
        <Button
          onClick={() => onToggleShow(true)}
          variant="outline"
          className="bg-slate-700 border-slate-600 text-white hover:bg-slate-600"
        >
          <Eye className="h-4 w-4 mr-2" />
          Show Preview Player
        </Button>
      </div>
    );
  }

  return (
    <Card className="bg-slate-800 border-slate-700 animate-fade-in">
      <CardHeader>
        <CardTitle className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-white">
          <span className="text-lg">SAM2 Processing Preview</span>
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-400">Show Preview</span>
              <Switch
                checked={show}
                onCheckedChange={onToggleShow}
              />
            </div>
            <div className="flex items-center gap-2">
              <Button
                onClick={handleDeleteSession}
                size="sm"
                variant="destructive"
                className="text-xs"
                disabled={!sessionData?.sessionId}
              >
                Delete Session
              </Button>
              <Button
                onClick={() => onToggleShow(false)}
                size="sm"
                variant="ghost"
                className="text-slate-400 hover:text-white"
              >
                <EyeOff className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Workflow status */}
        <div className="bg-slate-700 rounded-lg p-4 space-y-3">
          <h4 className="text-white font-medium">Processing Workflow</h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-slate-300">Session Created</span>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isPropagationComplete ? 'bg-green-500' : 'bg-slate-500'}`}></div>
              <span className={isPropagationComplete ? 'text-green-400' : 'text-slate-400'}>
                Annotations Propagated {isPropagationComplete && <CheckCircle className="inline h-3 w-3 ml-1" />}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${videoBlob ? 'bg-green-500' : 'bg-slate-500'}`}></div>
              <span className={videoBlob ? 'text-green-400' : 'text-slate-400'}>
                Video Generated {videoBlob && <CheckCircle className="inline h-3 w-3 ml-1" />}
              </span>
            </div>
          </div>
        </div>

        {/* Control buttons */}
        <div className="flex flex-col sm:flex-row gap-3">
          <Button
            onClick={handlePropagateAnnotations}
            disabled={isProcessing || !sessionData?.sessionId}
            className="bg-green-600 hover:bg-green-700 flex-1"
          >
            {isProcessing ? 'Propagating...' : 'Propagate Annotations'}
          </Button>
          
          <Button
            onClick={handleGenerateVideo}
            disabled={!isPropagationComplete || isGeneratingVideo || !sessionData?.sessionId}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:text-slate-400 flex-1"
            title={!isPropagationComplete ? "Complete propagation first" : "Generate video with masks"}
          >
            {isGeneratingVideo ? 'Generating...' : 'Generate Video'}
          </Button>

          {videoBlob && (
            <Button
              onClick={downloadVideo}
              className="bg-purple-600 hover:bg-purple-700 flex-1"
            >
              <Download className="h-4 w-4 mr-2" />
              Download
            </Button>
          )}
        </div>

        {/* Preview area */}
        <div className="bg-black rounded-lg overflow-hidden min-h-[200px] sm:min-h-[300px] flex items-center justify-center">
          {isProcessing ? (
            <div className="text-center p-6">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500 mx-auto mb-4"></div>
              <p className="text-slate-400">Propagating annotations with SAM2...</p>
              <p className="text-xs text-slate-500 mt-2">This may take a few minutes</p>
            </div>
          ) : isGeneratingVideo ? (
            <div className="text-center p-6">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
              <p className="text-slate-400">Generating annotated video...</p>
            </div>
          ) : videoBlob ? (
            <video
              src={videoBlob}
              controls
              className="max-w-full max-h-full"
            >
              Your browser does not support the video tag.
            </video>
          ) : maskData ? (
            <div className="text-center text-slate-300 p-6">
              <div className="w-16 h-16 bg-slate-700 rounded-lg flex items-center justify-center mx-auto mb-4">
                <CheckCircle className="h-8 w-8 text-green-400" />
              </div>
              <p className="mb-2">Propagation completed successfully</p>
              <p className="text-sm text-slate-500 mb-4">
                Frame: {maskData.frameIndex}, Objects: {maskData.results?.length || 0}
              </p>
              <p className="text-xs text-blue-400">
                Click "Generate Video" to create the final result
              </p>
            </div>
          ) : (
            <div className="text-center text-slate-400 p-6">
              <div className="w-16 h-16 bg-slate-700 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Play className="h-8 w-8" />
              </div>
              <p className="mb-4">Add annotations and propagate to begin processing</p>
              <div className="text-sm text-slate-500 space-y-1">
                <p>1. Add annotation points to your video</p>
                <p>2. Click "Propagate Annotations"</p>
                <p>3. Generate the final video</p>
              </div>
            </div>
          )}
        </div>

        {/* Session status */}
        <div className="text-xs text-slate-500 text-center">
          Session: {sessionData?.sessionId ? 
            <span className="text-green-400">Connected ({sessionData.sessionId.slice(0, 8)}...)</span> : 
            <span className="text-red-400">Not Connected</span>
          }
        </div>
      </CardContent>
    </Card>
  );
};
