/**
 * Video upload component with file upload functionality
 */
import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Upload, Video, FileText } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { createAnnotationSessionWithFile } from '@/services/annotationApi';
import { SessionData } from '@/types/annotation';

interface VideoUploadProps {
  onVideoUpload: (file: File, sessionData: SessionData) => void;
}

export const VideoUpload: React.FC<VideoUploadProps> = ({ onVideoUpload }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [procedure, setProcedure] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const { toast } = useToast();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
    } else {
      toast({ title: "Invalid file type", variant: "destructive" });
    }
  }, [toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/*': ['.mp4', '.mov', '.webm'] },
    maxFiles: 1
  });

  const handleUpload = async () => {
    if (!procedure || !selectedFile) {
      toast({ title: "Missing information", description: "Please select a file and procedure.", variant: "destructive" });
      return;
    }
    setIsProcessing(true);
    try {
      const sessionData = await createAnnotationSessionWithFile(selectedFile);
      onVideoUpload(selectedFile, { ...sessionData, procedure });
      toast({ title: "Session created successfully", description: `Session ID: ${sessionData.sessionId}` });
    } catch (error) {
      toast({ title: "Session creation failed", description: (error as Error).message, variant: "destructive" });
    } finally {
      setIsProcessing(false);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-white">
            <Video className="h-5 w-5" /> Video Selection
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div {...getRootProps()} className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${isDragActive ? 'border-blue-500 bg-blue-500/10' : 'border-slate-600 hover:border-slate-500'}`}>
            <input {...getInputProps()} />
            <Upload className="h-12 w-12 text-slate-400 mx-auto mb-4" />
            <p className="text-slate-300 text-lg mb-2">Drag & drop video, or click to select</p>
          </div>

          {selectedFile && (
            <div className="bg-slate-700 rounded-lg p-4 animate-scale-in flex items-center gap-3">
              <FileText className="h-8 w-8 text-blue-400" />
              <div>
                <p className="text-white font-medium">{selectedFile.name}</p>
                <p className="text-slate-400 text-sm">{formatFileSize(selectedFile.size)}</p>
              </div>
            </div>
          )}

          <div>
            <Label className="text-white font-medium mb-2">Procedure Type</Label>
            <Select value={procedure} onValueChange={setProcedure}>
              <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                <SelectValue placeholder="Select a surgical procedure" />
              </SelectTrigger>
              <SelectContent className="bg-slate-700 border-slate-600">
                <SelectItem value="laparoscopic_cholecystectomy">Laparoscopic Cholecystectomy</SelectItem>
                <SelectItem value="appendectomy">Appendectomy</SelectItem>
                <SelectItem value="hernia_repair">Hernia Repair</SelectItem>
                <SelectItem value="other">Other</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>
      
      {/* Optional: Medical Metadata Card can be added back here if needed */}

      <Button onClick={handleUpload} disabled={!procedure || !selectedFile || isProcessing} className="w-full bg-blue-600 hover:bg-blue-700 text-white" size="lg">
        {isProcessing ? 'Creating Session...' : 'Create Annotation Session'}
      </Button>
    </div>
  );
};
