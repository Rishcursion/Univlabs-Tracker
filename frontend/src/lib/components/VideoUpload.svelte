<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { Upload, Video, FileVideo, CircleCheck as CheckCircle } from '@lucide/svelte';
	
	const dispatch = createEventDispatcher<{
		'video-uploaded': File;
	}>();
	
	let isDragOver = false;
	let fileInput: HTMLInputElement;
	let isUploading = false;
	let uploadProgress = 0;
	
	function handleDragOver(event: DragEvent) {
		event.preventDefault();
		isDragOver = true;
	}
	
	function handleDragLeave(event: DragEvent) {
		event.preventDefault();
		isDragOver = false;
	}
	
	function handleDrop(event: DragEvent) {
		event.preventDefault();
		isDragOver = false;
		
		const files = event.dataTransfer?.files;
		if (files && files.length > 0) {
			handleFileSelect(files[0]);
		}
	}
	
	function handleFileInput(event: Event) {
		const target = event.target as HTMLInputElement;
		const files = target.files;
		if (files && files.length > 0) {
			handleFileSelect(files[0]);
		}
	}
	
	async function handleFileSelect(file: File) {
		if (file.type.startsWith('video/')) {
			isUploading = true;
			uploadProgress = 0;
			
			// Simulate upload progress
			const interval = setInterval(() => {
				uploadProgress += Math.random() * 20;
				if (uploadProgress >= 100) {
					uploadProgress = 100;
					clearInterval(interval);
					setTimeout(() => {
						isUploading = false;
						dispatch('video-uploaded', file);
					}, 500);
				}
			}, 150);
		} else {
			alert('Please select a valid video file.');
		}
	}
	
	function triggerFileInput() {
		if (!isUploading) {
			fileInput.click();
		}
	}
	
	function formatFileSize(bytes: number): string {
		if (bytes === 0) return '0 Bytes';
		const k = 1024;
		const sizes = ['Bytes', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	}
</script>

<div class="card max-w-3xl mx-auto">
	<div class="text-center mb-8">
		<div class="relative inline-block">
			<div class="w-16 h-16 bg-gradient-to-br from-primary-500 to-primary-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg">
				<Video class="w-8 h-8 text-white" />
			</div>
			<div class="absolute -top-2 -right-2 w-6 h-6 bg-secondary-500 rounded-full flex items-center justify-center">
				<Upload class="w-3 h-3 text-white" />
			</div>
		</div>
		<h2 class="text-3xl font-bold text-medical-900 mb-3">Upload Surgical Video</h2>
		<p class="text-medical-600 text-lg leading-relaxed max-w-2xl mx-auto">
			Upload a surgical video file to begin the annotation process with SAM2. 
			Our system supports high-quality video formats for precise segmentation.
		</p>
	</div>
	
	{#if isUploading}
		<div class="mb-8 animate-fade-in">
			<div class="bg-primary-50 rounded-xl p-6 border border-primary-200">
				<div class="flex items-center space-x-4">
					<div class="flex-shrink-0">
						<div class="w-12 h-12 bg-primary-500 rounded-full flex items-center justify-center">
							<FileVideo class="w-6 h-6 text-white animate-pulse" />
						</div>
					</div>
					<div class="flex-1">
						<p class="text-primary-900 font-semibold mb-2">Uploading video...</p>
						<div class="w-full bg-primary-200 rounded-full h-3">
							<div 
								class="bg-gradient-to-r from-primary-500 to-primary-600 h-3 rounded-full transition-all duration-300 ease-out"
								style="width: {uploadProgress}%"
							></div>
						</div>
						<p class="text-primary-700 text-sm mt-2">{Math.round(uploadProgress)}% complete</p>
					</div>
				</div>
			</div>
		</div>
	{:else}
		<div 
			class="upload-zone {isDragOver ? 'dragover' : ''} {isUploading ? 'pointer-events-none opacity-50' : ''}"
			on:dragover={handleDragOver}
			on:dragleave={handleDragLeave}
			on:drop={handleDrop}
			role="button"
			tabindex="0"
			on:click={triggerFileInput}
			on:keydown={(e) => e.key === 'Enter' && triggerFileInput()}
		>
			<div class="relative">
				<Upload class="w-16 h-16 text-medical-400 mx-auto mb-6 transition-all duration-300 {isDragOver ? 'scale-110 text-primary-500' : ''}" />
				
				<div class="space-y-4">
					<div>
						<p class="text-xl font-semibold text-medical-700 mb-2">
							Drag and drop your video here
						</p>
						<p class="text-medical-500 mb-6">or</p>
					</div>
					
					<button type="button" class="btn-primary interactive-btn text-lg px-8 py-4">
						Choose Video File
					</button>
				</div>
				
				<div class="mt-8 pt-6 border-t border-medical-200">
					<div class="grid md:grid-cols-2 gap-4 text-sm text-medical-600">
						<div class="flex items-center space-x-2">
							<CheckCircle class="w-4 h-4 text-green-500" />
							<span>Supported: MP4, AVI, MOV, WebM</span>
						</div>
						<div class="flex items-center space-x-2">
							<CheckCircle class="w-4 h-4 text-green-500" />
							<span>Maximum size: 500MB</span>
						</div>
						<div class="flex items-center space-x-2">
							<CheckCircle class="w-4 h-4 text-green-500" />
							<span>HD quality recommended</span>
						</div>
						<div class="flex items-center space-x-2">
							<CheckCircle class="w-4 h-4 text-green-500" />
							<span>Secure & private processing</span>
						</div>
					</div>
				</div>
			</div>
		</div>
	{/if}
	
	<input
		bind:this={fileInput}
		type="file"
		accept="video/*"
		class="hidden"
		on:change={handleFileInput}
		disabled={isUploading}
	/>
</div>
