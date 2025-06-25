
import { useState } from 'react';
import { VideoUpload } from '@/components/VideoUpload';
import { AnnotationCanvas } from '@/components/AnnotationCanvas';
import { ObjectManager } from '@/components/ObjectManager';
import { PreviewPlayer } from '@/components/PreviewPlayer';
import { Header } from '@/components/Header';
import { Footer } from '@/components/Footer';
import { usePreviewImages } from '@/hooks/usePreviewImages';

const Index = () => {
  const [uploadedVideo, setUploadedVideo] = useState<File | null>(null);
  const [sessionData, setSessionData] = useState<any>(null);
  const [objects, setObjects] = useState<any[]>([]);
  const [selectedObject, setSelectedObject] = useState<string | null>(null);
  const [annotationMode, setAnnotationMode] = useState<'positive' | 'negative'>('positive');
  const [showPreview, setShowPreview] = useState(true);
  const [annotations, setAnnotations] = useState<any[]>([]);
  const { clearAllPreviews } = usePreviewImages();

  const handleVideoUpload = (file: File, sessionInfo: any) => {
    setUploadedVideo(file);
    setSessionData(sessionInfo);
    console.log('Video uploaded with session data:', sessionInfo);
  };

  const handleObjectDeleted = (objectId: string) => {
    // Remove all annotations for the deleted object
    const updatedAnnotations = annotations.filter(ann => ann.object_id !== objectId);
    setAnnotations(updatedAnnotations);
    
    // Clear preview images to reset frame annotations
    clearAllPreviews();
    
    console.log(`Deleted object ${objectId} and cleared related annotations`);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      <Header />
      
      <main className="container mx-auto px-4 sm:px-6 py-4 sm:py-8">
        {!uploadedVideo ? (
          <div className="max-w-2xl mx-auto">
            <VideoUpload 
              onVideoUpload={handleVideoUpload}
            />
          </div>
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-4 gap-4 sm:gap-6 lg:gap-8">
            {/* Left Column - Controls */}
            <div className="xl:col-span-1 space-y-4 sm:space-y-6">
              <ObjectManager
                objects={objects}
                selectedObject={selectedObject}
                onObjectsChange={setObjects}
                onObjectSelect={setSelectedObject}
                annotationMode={annotationMode}
                onAnnotationModeChange={setAnnotationMode}
                onObjectDeleted={handleObjectDeleted}
              />
            </div>

            {/* Right Column - Video and Preview */}
            <div className="xl:col-span-3 space-y-4 sm:space-y-6">
              <AnnotationCanvas
                video={uploadedVideo}
                selectedObject={selectedObject}
                annotationMode={annotationMode}
                objects={objects}
                annotations={annotations}
                onAnnotationsChange={setAnnotations}
                sessionData={sessionData}
              />

              <PreviewPlayer
                show={showPreview}
                onToggleShow={setShowPreview}
                sessionData={sessionData}
              />
            </div>
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
};

export default Index;
