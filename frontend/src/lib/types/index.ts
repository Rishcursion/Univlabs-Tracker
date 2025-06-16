export interface Annotation {
	id: number;
	type: 'point' | 'rectangle' | 'polygon';
	coordinates: number[];
	timestamp: number;
	label?: string;
	confidence?: number;
}

export interface MaskData {
	video: string;
	procedure: string;
	timestamp: string;
	annotations: Annotation[];
	currentFrame: number;
}

export interface ProcedureConfig {
	id: string;
	name: string;
	description: string;
	duration: string;
	complexity: string;
	presetMasks: PresetMask[];
	available: boolean;
}

export interface PresetMask {
	id: string;
	name: string;
	color: string;
	description?: string;
}
