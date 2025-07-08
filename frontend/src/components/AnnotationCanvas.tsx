/**
 * Main annotation canvas component with video player and real-time mask preview
 */
import React, { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { VideoPlayer, VideoPlayerRef } from '@/components/VideoPlayer';
import { AnnotationsList } from '@/components/AnnotationsList';
import { sendAnnotationPoints, AnnotationResponse, BackendAnnotationRequest } from '@/services/annotationApi';
import { usePreviewImages } from '@/hooks/usePreviewImages';
import { Annotation, AnnotationObject, SessionData } from '@/types/annotation';

interface AnnotationCanvasProps {
	video: File;
	selectedObject: string | null;
	annotationMode: 'positive' | 'negative';
	objects: AnnotationObject[];
	annotations: Annotation[];
	onAnnotationsChange: (annotations: Annotation[]) => void;
	sessionData: SessionData;
}

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
	const [isProcessing, setIsProcessing] = useState(false);
	const { previewImages, setPreviewImage, getPreviewImage } = usePreviewImages();
	const defaultObject: AnnotationObject = { id: '0', name: 'Default', color: '#3b82f6', annotations: [] };

	const hasPositiveAnnotations = (): boolean => {
		const targetObjectId = selectedObject || defaultObject.id;
		return annotations.some(ann => ann.object_id === targetObjectId && ann.type === 'positive');
	};

	const sendAnnotationToBackend = async (
		points: number[][],
		labels: number[],
		objectId: string,
		color: string, // Pass color
	): Promise<void> => {
		if (!sessionData?.sessionId) return;
		setIsProcessing(true);
		try {
			const request: BackendAnnotationRequest = {
				sessionId: sessionData.sessionId,
				frameIndex: currentFrame,
				objectId: parseInt(objectId, 10) || 1,
				labels,
				points,
				color, // Include color in request
			};
			const result = await sendAnnotationPoints(request);
			if (result.previewImage) {
				setPreviewImage(currentFrame, result.previewImage);
			}
		} catch (error) {
			console.error('Error sending annotation:', error);
		} finally {
			setIsProcessing(false);
		}
	};

	const handleCanvasClick = async (x: number, y: number): Promise<void> => {
		if ((annotationMode === 'negative' && !hasPositiveAnnotations()) || isProcessing) return;

		const targetObjectId = selectedObject || defaultObject.id;
		const targetObject = objects.find(obj => obj.id === targetObjectId) || defaultObject;

		const newAnnotation: Annotation = {
			id: Date.now().toString(),
			object_id: targetObjectId, object_name: targetObject.name, object_color: targetObject.color,
			type: annotationMode, x, y, timestamp: currentTime, frame_index: currentFrame,
			created_at: new Date().toISOString()
		};
		const updatedAnnotations = [...annotations, newAnnotation];
		onAnnotationsChange(updatedAnnotations);

		// Prepare data for backend
		const currentObjectAnnotations = updatedAnnotations.filter(ann => ann.object_id === targetObjectId && Math.abs(ann.frame_index - currentFrame) < 2);
		const points = currentObjectAnnotations.map(ann => [ann.x, ann.y]);
		const labels = currentObjectAnnotations.map(ann => (ann.type === 'positive' ? 1 : 0));

		await sendAnnotationToBackend(points, labels, targetObjectId, targetObject.color);
	};

	const handleTimeUpdate = (time: number, frameIndex: number): void => {
		setCurrentTime(time);
		setCurrentFrame(frameIndex);
	};

	const currentPreview = getPreviewImage(currentFrame);

	return (
		<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
			<div className="lg:col-span-2">
				<Card className="bg-slate-800 border-slate-700">
					<CardHeader>
						<CardTitle className="flex items-center justify-between text-white">
							<span>Annotation Canvas</span>
							<div className="text-sm text-slate-400">
								Frame: {currentFrame}
								{currentPreview && <span className="ml-2 text-green-400">• Preview Active</span>}
							</div>
						</CardTitle>
					</CardHeader>
					<CardContent>
						<VideoPlayer
							ref={videoPlayerRef}
							video={video}
							annotations={annotations}
							currentFrame={currentFrame}
							onTimeUpdate={handleTimeUpdate}
							onCanvasClick={handleCanvasClick}
							showCrosshair={!isProcessing}
							crosshairColor={annotationMode === 'positive' ? '#22c55e' : '#ef4444'}
							previewImage={currentPreview}
							isProcessing={isProcessing}
						/>
					</CardContent>
				</Card>
			</div>
			<div className="lg:col-span-1">
				<AnnotationsList
					annotations={annotations}
					objects={objects}
					onDeleteAnnotation={(id) => onAnnotationsChange(annotations.filter(a => a.id !== id))}
					onSeekToAnnotation={(time) => videoPlayerRef.current?.seek(time)}
					onClearAllAnnotations={() => onAnnotationsChange([])}
				/>
			</div>
		</div>
	);
};
