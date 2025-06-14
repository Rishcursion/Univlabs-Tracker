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

<div class="card mx-auto max-w-3xl">
	<div class="mb-8 text-center">
		<div class="relative inline-block">
			<div
				class="from-primary-500 to-primary-600 mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br shadow-lg"
			>
				<Video class="h-8 w-8 text-white" />
			</div>
			<div
				class="bg-secondary-500 absolute -top-2 -right-2 flex h-6 w-6 items-center justify-center rounded-full"
			>
				<Upload class="h-3 w-3 text-white" />
			</div>
		</div>
		<h2 class="text-medical-900 mb-3 text-3xl font-bold">Upload Surgical Video</h2>
		<p class="text-medical-600 mx-auto max-w-2xl text-lg leading-relaxed">
			Upload a surgical video file to begin the annotation process with SAM2. Our system supports
			high-quality video formats for precise segmentation.
		</p>
	</div>

	{#if isUploading}
		<div class="animate-fade-in mb-8">
			<div class="bg-primary-50 border-primary-200 rounded-xl border p-6">
				<div class="flex items-center space-x-4">
					<div class="flex-shrink-0">
						<div class="bg-primary-500 flex h-12 w-12 items-center justify-center rounded-full">
							<FileVideo class="h-6 w-6 animate-pulse text-white" />
						</div>
					</div>
					<div class="flex-1">
						<p class="text-primary-900 mb-2 font-semibold">Uploading video...</p>
						<div class="bg-primary-200 h-3 w-full rounded-full">
							<div
								class="from-primary-500 to-primary-600 h-3 rounded-full bg-gradient-to-r transition-all duration-300 ease-out"
								style="width: {uploadProgress}%"
							></div>
						</div>
						<p class="text-primary-700 mt-2 text-sm">{Math.round(uploadProgress)}% complete</p>
					</div>
				</div>
			</div>
		</div>
	{:else}
		<div
			class="upload-zone {isDragOver ? 'dragover' : ''} {isUploading
				? 'pointer-events-none opacity-50'
				: ''}"
			on:dragover={handleDragOver}
			on:dragleave={handleDragLeave}
			on:drop={handleDrop}
			role="button"
			tabindex="0"
			on:click={triggerFileInput}
			on:keydown={(e) => e.key === 'Enter' && triggerFileInput()}
		>
			<div class="relative">
				<Upload
					class="text-medical-400 mx-auto mb-6 h-16 w-16 transition-all duration-300 {isDragOver
						? 'text-primary-500 scale-110'
						: ''}"
				/>

				<div class="space-y-4">
					<div>
						<p class="text-medical-700 mb-2 text-xl font-semibold">Drag and drop your video here</p>
						<p class="text-medical-500 mb-6">or</p>
					</div>

					<button type="button" class="btn-primary px-8 py-4 text-lg"> Choose Video File </button>
				</div>

				<div class="border-medical-200 mt-8 border-t pt-6">
					<div class="text-medical-600 grid gap-4 text-sm md:grid-cols-2">
						<div class="flex items-center space-x-2">
							<CheckCircle class="h-4 w-4 text-green-500" />
							<span>Supported: MP4, AVI, MOV, WebM</span>
						</div>
						<div class="flex items-center space-x-2">
							<CheckCircle class="h-4 w-4 text-green-500" />
							<span>Maximum size: 500MB</span>
						</div>
						<div class="flex items-center space-x-2">
							<CheckCircle class="h-4 w-4 text-green-500" />
							<span>HD quality recommended</span>
						</div>
						<div class="flex items-center space-x-2">
							<CheckCircle class="h-4 w-4 text-green-500" />
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
