
/**
 * Video upload component with file upload and optional S3 integration
 */

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Upload, Video, FileText, User, Calendar, Stethoscope, Link } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { createAnnotationSession, createAnnotationSessionWithFile } from '@/services/annotationApi';
import { SessionData, VideoMetadata } from '@/types/annotation';

interface VideoUploadProps {
  onVideoUpload: (file: File, sessionData: SessionData) => void;
}

interface MetadataForm {
  patientName: string;
  doctorName: string;
  surgeryOutcome: string;
  surgeryDate: string;
  patientAge: string;
  complications: string;
  notes: string;
}

/**
 * Component for uploading videos and creating annotation sessions
 */
export const VideoUpload: React.FC<VideoUploadProps> = ({ onVideoUpload }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [procedure, setProcedure] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [s3Link, setS3Link] = useState<string>('');
  const [uploadMethod, setUploadMethod] = useState<'file' | 's3'>('file');
  const [metadata, setMetadata] = useState<MetadataForm>({
    patientName: '',
    doctorName: '',
    surgeryOutcome: '',
    surgeryDate: '',
    patientAge: '',
    complications: '',
    notes: ''
  });
  const { toast } = useToast();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
    } else {
      toast({
        title: "Invalid file type",
        description: "Please upload a valid video file (MP4, AVI, MOV, etc.)",
        variant: "destructive"
      });
    }
  }, [toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    },
    maxFiles: 1
  });

  const handleMetadataChange = (field: keyof MetadataForm, value: string) => {
    setMetadata(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleUpload = async () => {
    if (!procedure) {
      toast({
        title: "Missing information",
        description: "Please select a procedure type",
        variant: "destructive"
      });
      return;
    }

    // Validate based on upload method
    if (uploadMethod === 'file' && !selectedFile) {
      toast({
        title: "Missing file",
        description: "Please select a video file to upload",
        variant: "destructive"
      });
      return;
    }

    if (uploadMethod === 's3' && !s3Link.trim()) {
      toast({
        title: "Missing S3 link",
        description: "Please provide an S3 link to process the video",
        variant: "destructive"
      });
      return;
    }

    setIsProcessing(true);

    try {
      let sessionData: SessionData;

      if (uploadMethod === 'file' && selectedFile) {
        console.log('Creating session with file upload:', selectedFile.name);
        sessionData = await createAnnotationSessionWithFile(selectedFile);
      } else if (uploadMethod === 's3') {
        console.log('Creating session with S3 link:', s3Link);
        sessionData = await createAnnotationSession(s3Link.trim());
      } else {
        throw new Error('Invalid upload configuration');
      }
      
      // Compile video metadata
      const videoMetadata: VideoMetadata = {
        name: selectedFile?.name || 'S3 Video',
        size: selectedFile?.size || 0,
        format: selectedFile?.type || 'video/*',
        patientName: metadata.patientName || undefined,
        doctorName: metadata.doctorName || undefined,
        surgeryOutcome: metadata.surgeryOutcome || undefined,
        surgeryDate: metadata.surgeryDate || undefined,
        patientAge: metadata.patientAge ? parseInt(metadata.patientAge) : undefined,
        complications: metadata.complications || undefined,
        notes: metadata.notes || undefined,
      };

      // Add procedure and metadata to session data
      const completeSessionData: SessionData = {
        ...sessionData,
        procedure: procedure,
        metadata: videoMetadata,
        s3Link: uploadMethod === 's3' ? s3Link : undefined
      };

      // Use selectedFile or create a dummy file for S3 uploads
      const fileForCallback = selectedFile || new File([''], 'S3 Video', { type: 'video/mp4' });
      onVideoUpload(fileForCallback, completeSessionData);
      
      toast({
        title: "Session created successfully",
        description: `Session ID: ${sessionData.sessionId}`
      });

    } catch (error) {
      console.error('Session creation error:', error);
      toast({
        title: "Session creation failed",
        description: "Failed to create annotation session with backend",
        variant: "destructive"
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const formatFileSize = (bytes: number): string => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Byte';
    const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)).toString());
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-white mb-4">Upload Surgical Video</h2>
        <p className="text-slate-400 text-lg">
          Begin your annotation workflow by uploading a surgical video file
        </p>
      </div>

      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-white">
            <Video className="h-5 w-5" />
            Video Selection
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Upload method tabs */}
          <Tabs value={uploadMethod} onValueChange={(value) => setUploadMethod(value as 'file' | 's3')}>
            <TabsList className="grid w-full grid-cols-2 bg-slate-700">
              <TabsTrigger value="file" className="data-[state=active]:bg-slate-600">
                <Upload className="h-4 w-4 mr-2" />
                File Upload
              </TabsTrigger>
              <TabsTrigger value="s3" className="data-[state=active]:bg-slate-600">
                <Link className="h-4 w-4 mr-2" />
                S3 Link
              </TabsTrigger>
            </TabsList>

            <TabsContent value="file" className="space-y-4">
              {/* File drop zone */}
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200 ${
                  isDragActive
                    ? 'border-blue-500 bg-blue-500/10'
                    : 'border-slate-600 hover:border-slate-500 hover:bg-slate-750'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="h-12 w-12 text-slate-400 mx-auto mb-4" />
                {isDragActive ? (
                  <p className="text-blue-400 text-lg">Drop the video file here...</p>
                ) : (
                  <div>
                    <p className="text-slate-300 text-lg mb-2">
                      Drag & drop a video file here, or click to select
                    </p>
                    <p className="text-slate-500 text-sm">
                      Supported formats: MP4, AVI, MOV, MKV, WebM
                    </p>
                  </div>
                )}
              </div>

              {/* Selected file display */}
              {selectedFile && (
                <div className="bg-slate-700 rounded-lg p-4 animate-scale-in">
                  <div className="flex items-center gap-3">
                    <FileText className="h-8 w-8 text-blue-400" />
                    <div className="flex-1">
                      <p className="text-white font-medium">{selectedFile.name}</p>
                      <p className="text-slate-400 text-sm">{formatFileSize(selectedFile.size)}</p>
                    </div>
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="s3" className="space-y-4">
              <div>
                <Label className="text-white font-medium mb-2">S3 Video Link</Label>
                <Input
                  type="url"
                  placeholder="https://your-bucket.s3.amazonaws.com/video.mp4"
                  value={s3Link}
                  onChange={(e) => setS3Link(e.target.value)}
                  className="bg-slate-700 border-slate-600 text-white"
                />
                <p className="text-slate-400 text-sm mt-1">
                  Provide the S3 link to your video file for backend processing
                </p>
              </div>
            </TabsContent>
          </Tabs>

          {/* Procedure selection */}
          <div>
            <Label className="text-white font-medium mb-2">Procedure Type</Label>
            <Select value={procedure} onValueChange={setProcedure}>
              <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                <SelectValue placeholder="Select a surgical procedure" />
              </SelectTrigger>
              <SelectContent className="bg-slate-700 border-slate-600">
                <SelectItem value="laparoscopic_cholecystectomy">
                  Laparoscopic Cholecystectomy
                </SelectItem>
                <SelectItem value="appendectomy">
                  Appendectomy
                </SelectItem>
                <SelectItem value="hernia_repair">
                  Hernia Repair
                </SelectItem>
                <SelectItem value="gallbladder_removal">
                  Gallbladder Removal
                </SelectItem>
                <SelectItem value="gastric_bypass">
                  Gastric Bypass
                </SelectItem>
                <SelectItem value="other">
                  Other
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Medical Metadata Card */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-white">
            <Stethoscope className="h-5 w-5" />
            Medical Information (Optional)
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Patient Name */}
            <div>
              <Label className="text-white font-medium mb-2">Patient Name</Label>
              <Input
                placeholder="Enter patient name"
                value={metadata.patientName}
                onChange={(e) => handleMetadataChange('patientName', e.target.value)}
                className="bg-slate-700 border-slate-600 text-white"
              />
            </div>

            {/* Doctor Name */}
            <div>
              <Label className="text-white font-medium mb-2">Surgeon Name</Label>
              <Input
                placeholder="Enter surgeon name"
                value={metadata.doctorName}
                onChange={(e) => handleMetadataChange('doctorName', e.target.value)}
                className="bg-slate-700 border-slate-600 text-white"
              />
            </div>

            {/* Patient Age */}
            <div>
              <Label className="text-white font-medium mb-2">Patient Age</Label>
              <Input
                type="number"
                placeholder="Enter patient age"
                value={metadata.patientAge}
                onChange={(e) => handleMetadataChange('patientAge', e.target.value)}
                className="bg-slate-700 border-slate-600 text-white"
              />
            </div>

            {/* Surgery Date */}
            <div>
              <Label className="text-white font-medium mb-2">Surgery Date</Label>
              <Input
                type="date"
                value={metadata.surgeryDate}
                onChange={(e) => handleMetadataChange('surgeryDate', e.target.value)}
                className="bg-slate-700 border-slate-600 text-white"
              />
            </div>
          </div>

          {/* Surgery Outcome */}
          <div>
            <Label className="text-white font-medium mb-2">Surgery Outcome</Label>
            <Select 
              value={metadata.surgeryOutcome} 
              onValueChange={(value) => handleMetadataChange('surgeryOutcome', value)}
            >
              <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                <SelectValue placeholder="Select surgery outcome" />
              </SelectTrigger>
              <SelectContent className="bg-slate-700 border-slate-600">
                <SelectItem value="successful">Successful</SelectItem>
                <SelectItem value="partially_successful">Partially Successful</SelectItem>
                <SelectItem value="complications">Had Complications</SelectItem>
                <SelectItem value="unsuccessful">Unsuccessful</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Complications */}
          <div>
            <Label className="text-white font-medium mb-2">Complications</Label>
            <Input
              placeholder="Describe any complications (if any)"
              value={metadata.complications}
              onChange={(e) => handleMetadataChange('complications', e.target.value)}
              className="bg-slate-700 border-slate-600 text-white"
            />
          </div>

          {/* Notes */}
          <div>
            <Label className="text-white font-medium mb-2">Additional Notes</Label>
            <Textarea
              placeholder="Enter any additional notes or observations"
              value={metadata.notes}
              onChange={(e) => handleMetadataChange('notes', e.target.value)}
              className="bg-slate-700 border-slate-600 text-white min-h-[100px]"
            />
          </div>
        </CardContent>
      </Card>

      {/* Upload button */}
      <Button
        onClick={handleUpload}
        disabled={
          !procedure || 
          isProcessing || 
          (uploadMethod === 'file' && !selectedFile) ||
          (uploadMethod === 's3' && !s3Link.trim())
        }
        className="w-full bg-blue-600 hover:bg-blue-700 text-white"
        size="lg"
      >
        {isProcessing ? 'Creating Session...' : 'Create Annotation Session'}
      </Button>
    </div>
  );
};
