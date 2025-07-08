/**
 * Preview player component for SAM2 processing and video generation (Updated)
 */
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Eye, Play, Download, Loader2 } from 'lucide-react';
import { processVideo, deleteAnnotationSession, ProcessVideoRequest } from '@/services/annotationApi';
import { SessionData } from '@/types/annotation';
import { useToast } from '@/hooks/use-toast';

interface PreviewPlayerProps {
  show: boolean;
  onToggleShow: (show: boolean) => void;
  sessionData: SessionData;
}

export const PreviewPlayer: React.FC<PreviewPlayerProps> = ({
  show,
  onToggleShow,
  sessionData
}) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [videoBlobUrl, setVideoBlobUrl] = useState<string | null>(null);
  const { toast } = useToast();
  
  const handleDeleteSession = async (): Promise<void> => {
    if (!sessionData?.sessionId) return;
    try {
      await deleteAnnotationSession(sessionData.sessionId);
      setVideoBlobUrl(null);
      toast({ title: "Session Deleted", description: "Session and associated files have been cleaned up." });
      window.location.reload();
    } catch (error) {
      toast({ title: "Delete Failed", description: "Failed to delete session.", variant: "destructive" });
    }
  };

  const handleProcessVideo = async (): Promise<void> => {
    if (!sessionData?.sessionId) return;
    setIsProcessing(true);
    setVideoBlobUrl(null);
    try {
      const request: ProcessVideoRequest = { sessionId: sessionData.sessionId };
      const blob = await processVideo(request);
      const videoUrl = URL.createObjectURL(blob);
      setVideoBlobUrl(videoUrl);
      toast({ title: "Video Processing Complete", description: "Your annotated video is ready." });
    } catch (error) {
      toast({ title: "Processing Failed", description: (error as Error).message, variant: "destructive" });
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadVideo = (): void => {
    if (!videoBlobUrl) return;
    const a = document.createElement('a');
    a.href = videoBlobUrl;
    a.download = 'annotated_video.mp4'; // Correct extension
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  if (!show) {
    return (
      <div className="flex items-center justify-center p-4">
        <Button onClick={() => onToggleShow(true)} variant="outline" className="bg-slate-700 hover:bg-slate-600">
          <Eye className="h-4 w-4 mr-2" />
          Show Processing Player
        </Button>
      </div>
    );
  }

  return (
    <Card className="bg-slate-800 border-slate-700 animate-fade-in">
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-white">
          <span>Processing & Preview</span>
          <Button onClick={handleDeleteSession} size="sm" variant="destructive" disabled={!sessionData?.sessionId}>
            Delete Session
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex flex-col sm:flex-row gap-3">
          <Button onClick={handleProcessVideo} disabled={isProcessing || !sessionData?.sessionId} className="bg-blue-600 hover:bg-blue-700 flex-1">
            {isProcessing ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Processing...</> : 'Process Video'}
          </Button>
          {videoBlobUrl && (
            <Button onClick={downloadVideo} className="bg-purple-600 hover:bg-purple-700 flex-1">
              <Download className="h-4 w-4 mr-2" />
              Download
            </Button>
          )}
        </div>

        <div className="bg-black rounded-lg overflow-hidden min-h-[300px] flex items-center justify-center">
          {isProcessing ? (
            <div className="text-center p-6">
              <Loader2 className="h-12 w-12 text-blue-500 animate-spin mx-auto mb-4" />
              <p className="text-slate-400">Processing video with SAM2...</p>
              <p className="text-xs text-slate-500 mt-2">This may take several minutes for long videos.</p>
            </div>
          ) : videoBlobUrl ? (
            <video src={videoBlobUrl} controls className="w-full h-full" />
          ) : (
            <div className="text-center text-slate-400 p-6">
              <Play className="h-12 w-12 text-slate-500 mx-auto mb-4" />
              <p>Add annotations and click "Process Video"</p>
            </div>
          )}
        </div>

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
