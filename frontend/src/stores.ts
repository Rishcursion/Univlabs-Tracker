import { writable } from 'svelte/store';
import type { VideoData, AnnotationMask, AnnotationState } from './Interfaces'; // Adjust path if needed

// Store for the currently loaded video data
export const videoData = writable<VideoData | null>(null);

// Store for all available annotation masks
export const availableMasks = writable<AnnotationMask[]>([]);

// Store for the overall annotation state
export const annotationState = writable<AnnotationState>({
	selectedMask: null,
	currentFrame: 0, // This will be updated by the video player
	isAnnotating: false,
	annotationTool: 'polygon',
	annotationMode: 'drawing', // Add this property
	drawingPoints: [], // Add this property
	currentBox: null // Add this property
});

// Function to add a video to the store
export function addVideo(video: VideoData) {
	videoData.set(video);
	// You might want to clear existing masks if a new video is loaded
	// availableMasks.set([]);
}

// Function to clear all annotations from all masks
export function clearAllAnnotations() {
	availableMasks.update((masks) => masks.map((mask) => ({ ...mask, annotations: [] })));
	console.log('All annotations cleared.');
}

// Function to update currentFrame in annotationState
export function updateCurrentFrame(frame: number) {
	annotationState.update((state) => ({ ...state, currentFrame: frame }));
}
