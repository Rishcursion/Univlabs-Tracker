// Mock API functions for annotation management

export interface AnnotationData {
	id: number;
	type: 'positive_point' | 'negative_point' | 'box';
	x: number;
	y: number;
	width?: number;
	height?: number;
	timestamp: number;
	videoId: string;
	sessionId: string;
}

export interface SAM2ProcessRequest {
	videoId: string;
	annotations: AnnotationData[];
	procedure: string;
	currentFrame: number;
}

export interface SAM2ProcessResponse {
	success: boolean;
	maskUrl?: string;
	processId: string;
	status: 'processing' | 'completed' | 'failed';
}

// Mock API base URL - in production this would be your actual backend
const API_BASE_URL = 'http://localhost:8000/api';

// Simulate network delay
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function sendAnnotation(
	annotation: AnnotationData
): Promise<{ success: boolean; id?: string }> {
	try {
		console.log('Sending annotation to backend:', annotation);

		// Simulate API call delay
		await delay(200 + Math.random() * 300);

		// Mock successful response
		const response = {
			success: true,
			id: `ann_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
			timestamp: new Date().toISOString()
		};

		console.log('Annotation sent successfully:', response);
		return response;
	} catch (error) {
		console.error('Failed to send annotation:', error);
		return { success: false };
	}
}

export async function processSAM2(request: SAM2ProcessRequest): Promise<SAM2ProcessResponse> {
	try {
		console.log('🚀 Starting SAM2 processing:', request);

		// Simulate processing delay
		await delay(1000 + Math.random() * 2000);

		// Mock successful response
		const response: SAM2ProcessResponse = {
			success: true,
			maskUrl: `${API_BASE_URL}/masks/${request.videoId}_${Date.now()}.png`,
			processId: `proc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
			status: 'processing'
		};

		console.log('SAM2 processing started:', response);
		return response;
	} catch (error) {
		console.error('SAM2 processing failed:', error);
		return {
			success: false,
			processId: '',
			status: 'failed'
		};
	}
}

export async function getProcessingStatus(processId: string): Promise<SAM2ProcessResponse> {
	try {
		console.log('Checking processing status for:', processId);

		// Simulate API call delay
		await delay(500);

		// Mock completed response
		const response: SAM2ProcessResponse = {
			success: true,
			maskUrl: `${API_BASE_URL}/masks/completed_${processId}.png`,
			processId,
			status: 'completed'
		};

		console.log('Processing status retrieved:', response);
		return response;
	} catch (error) {
		console.error('Failed to get processing status:', error);
		return {
			success: false,
			processId,
			status: 'failed'
		};
	}
}

export async function downloadMasks(videoId: string, annotations: AnnotationData[]): Promise<Blob> {
	try {
		console.log('Preparing mask download for:', videoId);

		// Create mock mask data
		const maskData = {
			videoId,
			annotations,
			timestamp: new Date().toISOString(),
			sam2_prompts: {
				positive_points: annotations
					.filter((ann) => ann.type === 'positive_point')
					.map((ann) => [ann.x, ann.y]),
				negative_points: annotations
					.filter((ann) => ann.type === 'negative_point')
					.map((ann) => [ann.x, ann.y]),
				boxes: annotations
					.filter((ann) => ann.type === 'box')
					.map((ann) => [ann.x, ann.y, ann.x + (ann.width || 0), ann.y + (ann.height || 0)])
			}
		};

		// Simulate processing delay
		await delay(800);

		const blob = new Blob([JSON.stringify(maskData, null, 2)], {
			type: 'application/json'
		});

		console.log('Mask data prepared for download');
		return blob;
	} catch (error) {
		console.error('Failed to prepare mask download:', error);
		throw error;
	}
}

// Real-time annotation sync (WebSocket simulation)
export class AnnotationSync {
	private listeners: ((annotation: AnnotationData) => void)[] = [];
	private connected = false;

	connect(sessionId: string) {
		console.log('Connecting to annotation sync for session:', sessionId);
		this.connected = true;

		// Simulate connection delay
		setTimeout(() => {
			console.log('Connected to annotation sync');
		}, 500);
	}

	disconnect() {
		console.log('Disconnecting from annotation sync');
		this.connected = false;
		this.listeners = [];
	}

	onAnnotation(callback: (annotation: AnnotationData) => void) {
		this.listeners.push(callback);
	}

	// Simulate receiving annotations from other users
	private simulateIncomingAnnotation() {
		if (!this.connected) return;

		const mockAnnotation: AnnotationData = {
			id: Date.now(),
			type: 'positive_point',
			x: Math.random() * 800,
			y: Math.random() * 600,
			timestamp: Math.random() * 100,
			videoId: 'current-video',
			sessionId: 'other-user-session'
		};

		this.listeners.forEach((callback) => callback(mockAnnotation));
	}
}
