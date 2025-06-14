<script lang="ts">
	import {onMount} from 'svelte';
	import VideoUpload from '$lib/components/VideoUpload.svelte';
	import ProcedureSelector from '$lib/components/ProcedureSelector.svelte';
	import VideoAnnotation from '$lib/components/VideoAnnotation.svelte';
	import MaskOptions from '$lib/components/MaskOptions.svelte';

	let currentStep: 'upload' | 'procedure' | 'annotation' = 'upload';
	let uploadedVideo: File | null = null;
	let selectedProcedure: string = '';
	let videoUrl: string = '';

	function handleVideoUpload(event: CustomEvent<File>) {
		uploadedVideo = event.detail;
		videoUrl = URL.createObjectURL(uploadedVideo);
		currentStep = 'procedure';
	}

	function handleProcedureSelect(event: CustomEvent<string>) {
		selectedProcedure = event.detail;
		currentStep = 'annotation';
	}

	function resetToUpload() {
		currentStep = 'upload';
		uploadedVideo = null;
		selectedProcedure = '';
		if (videoUrl) {
			URL.revokeObjectURL(videoUrl);
			videoUrl = '';
		}
	}
</script>

<div class="max-w-7xl mx-auto">
	<!-- Enhanced Progress Indicator -->
	<div class="mb-12">
		<div class="flex items-center justify-center space-x-8">
			<div class="flex items-center">
				<div
					class="progress-step {currentStep === 'upload' ? 'active' : uploadedVideo ? 'completed' : 'inactive'}">
					1
				</div>
				<span class="ml-3 text-sm font-semibold text-medical-700">Upload Video</span>
			</div>

			<div class="w-16 h-px bg-gradient-to-r from-medical-300 to-medical-400 relative">
				<div
					class="absolute inset-0 bg-gradient-to-r from-primary-500 to-primary-600 transition-all duration-500 {uploadedVideo ? 'opacity-100' : 'opacity-0'}">
				</div>
			</div>

			<div class="flex items-center">
				<div
					class="progress-step {currentStep === 'procedure' ? 'active' : selectedProcedure ? 'completed' : 'inactive'}">
					2
				</div>
				<span class="ml-3 text-sm font-semibold text-medical-700">Select Procedure</span>
			</div>

			<div class="w-16 h-px bg-gradient-to-r from-medical-300 to-medical-400 relative">
				<div
					class="absolute inset-0 bg-gradient-to-r from-primary-500 to-primary-600 transition-all duration-500 {selectedProcedure ? 'opacity-100' : 'opacity-0'}">
				</div>
			</div>

			<div class="flex items-center">
				<div class="progress-step {currentStep === 'annotation' ? 'active' : 'inactive'}">
					3
				</div>
				<span class="ml-3 text-sm font-semibold text-medical-700">Annotate & Process</span>
			</div>
		</div>
	</div>

	<!-- Content based on current step -->
	{#if currentStep === 'upload'}
	<div class="animate-fade-in">
		<VideoUpload on:video-uploaded={handleVideoUpload} />
	</div>
	{:else if currentStep === 'procedure'}
	<div class="animate-slide-up">
		<ProcedureSelector {uploadedVideo} on:procedure-selected={handleProcedureSelect} on:back={resetToUpload} />
	</div>
	{:else if currentStep === 'annotation'}
	<div class="animate-slide-up">
		<VideoAnnotation {videoUrl} {selectedProcedure} {uploadedVideo} on:back={()=> currentStep = 'procedure'}
			/>
	</div>
	{/if}
</div>
