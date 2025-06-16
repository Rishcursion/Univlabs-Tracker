<script lang="ts">
    import { onMount, createEventDispatcher } from 'svelte';
    import {
        ArrowLeft,
        Play,
        Pause,
        SkipBack,
        SkipForward,
        Download,
        Trash2,
        MousePointer,
        Square,
        Plus,
        Minus,
        Eye,
        EyeOff,
        Layers,
        Sparkles,
    } from '@lucide/svelte';
    import MaskOptions from './MaskOptions.svelte';
    import AnnotationCanvas from './AnnotationCanvas.svelte';
    import pkg from 'file-saver';
    const saveAs = pkg;
    const dispatch = createEventDispatcher<{
        back: void;
    }>();

    export let videoUrl: string;
    export let selectedProcedure: string;
    export let uploadedVideo: File;
    let videoElement: HTMLVideoElement;
    let previewVideoElement: HTMLVideoElement;
    let annotationCanvas: AnnotationCanvas;
    let isPlaying = false;
    let currentTime = 0;
    let duration = 0;
    let annotations: any[] = [];
    let selectedMaskOption = 'custom';
    let annotationMode = false;
    let currentAnnotationType: 'positive' | 'negative' | 'box' = 'positive';
    let showPreview = true;
    let isProcessing = false;
    let processingProgress = 0;

    // Sync preview video with main video
    $: if (previewVideoElement && videoElement) {
        previewVideoElement.currentTime = currentTime;
        if (isPlaying) {
            previewVideoElement.play();
        } else {
            previewVideoElement.pause();
        }
    }

    function handleBack() {
        dispatch('back');
    }

    function togglePlay() {
        if (videoElement) {
            if (isPlaying) {
                videoElement.pause();
            } else {
                videoElement.play();
            }
        }
    }

    function seekBackward() {
        if (videoElement) {
            videoElement.currentTime = Math.max(0, videoElement.currentTime - 10);
        }
    }

    function seekForward() {
        if (videoElement) {
            videoElement.currentTime = Math.min(duration, videoElement.currentTime + 10);
        }
    }

    function handleTimeUpdate() {
        if (videoElement) {
            currentTime = videoElement.currentTime;
        }
    }

    function handleLoadedMetadata() {
        if (videoElement) {
            duration = videoElement.duration;
        }
    }

    function handlePlay() {
        isPlaying = true;
    }

    function handlePause() {
        isPlaying = false;
    }

    function handleSeek(event: Event) {
        const target = event.target as HTMLInputElement;
        if (videoElement) {
            videoElement.currentTime = Number(target.value);
        }
    }

    function formatTime(time: number): string {
        const minutes = Math.floor(time / 60);
        const seconds = Math.floor(time % 60);
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }

    function toggleAnnotationMode() {
        annotationMode = !annotationMode;
    }

    function setAnnotationType(type: 'positive' | 'negative' | 'box') {
        currentAnnotationType = type;
    }

    function handleMaskOptionChange(event: CustomEvent<string>) {
        selectedMaskOption = event.detail;
    }

    function handleAnnotationCreated(event: CustomEvent<any>) {
        const annotation = event.detail;
        annotations = [...annotations, annotation];
    }

    function togglePreview() {
        showPreview = !showPreview;
    }

    function processSAM2() {
        isProcessing = true;
        processingProgress = 0;

        // Simulate processing
        const interval = setInterval(() => {
            processingProgress += Math.random() * 15;
            if (processingProgress >= 100) {
                processingProgress = 100;
                clearInterval(interval);
                setTimeout(() => {
                    isProcessing = false;
                }, 500);
            }
        }, 200);
    }

    function downloadMasks() {
        try {
            // Create comprehensive annotation data
            const maskData = {
                video: {
                    name: uploadedVideo.name,
                    duration: duration,
                    currentFrame: currentTime,
                },
                procedure: selectedProcedure,
                timestamp: new Date().toISOString(),
                annotations: annotations.map((ann) => ({
                    id: ann.id,
                    type: ann.type,
                    coordinates:
                        ann.type === 'box' ? [ann.x, ann.y, ann.x + ann.width, ann.y + ann.height] : [ann.x, ann.y],
                    timestamp: ann.timestamp,
                    frame: Math.floor(ann.timestamp * 30), // Assuming 30fps
                })),
                sam2_prompts: {
                    positive_points: annotations
                        .filter((ann) => ann.type === 'positive_point')
                        .map((ann) => [ann.x, ann.y]),
                    negative_points: annotations
                        .filter((ann) => ann.type === 'negative_point')
                        .map((ann) => [ann.x, ann.y]),
                    boxes: annotations
                        .filter((ann) => ann.type === 'box')
                        .map((ann) => [ann.x, ann.y, ann.x + ann.width, ann.y + ann.height]),
                },
                metadata: {
                    total_annotations: annotations.length,
                    annotation_types: {
                        positive_points: annotations.filter((ann) => ann.type === 'positive_point').length,
                        negative_points: annotations.filter((ann) => ann.type === 'negative_point').length,
                        boxes: annotations.filter((ann) => ann.type === 'box').length,
                    },
                    mask_option: selectedMaskOption,
                },
            };

            // Create and download the file
            const blob = new Blob([JSON.stringify(maskData, null, 2)], {
                type: 'application/json',
            });

            const filename = `sam2_annotations_${uploadedVideo.name.replace(/\.[^/.]+$/, '')}_${Date.now()}.json`;
            saveAs(blob, filename);

            console.log('Annotations downloaded successfully');
        } catch (error) {
            console.error('Failed to download annotations:', error);
            alert('Failed to download annotations. Please try again.');
        }
    }

    function clearAnnotations() {
        annotations = [];
        annotationCanvas?.clearAnnotations();
    }
    function deleteAnnotation(annotationId: number) {
        // Remove annotation from the list
        annotations = annotations.filter((ann) => ann.id !== annotationId);

        // Update the canvas to reflect the change
        if (annotationCanvas) {
            annotationCanvas.removeAnnotation(annotationId);
        }
    }

    function seekToAnnotation(timestamp: number) {
        if (videoElement) {
            videoElement.currentTime = timestamp;
        }
    }
</script>

<div class="mx-auto max-w-7xl space-y-8">
    <!-- Header with Actions -->
    <div class="flex items-center justify-between">
        <button
            on:click={handleBack}
            class="interactive-btn text-medical-600 hover:text-medical-800 flex items-center transition-all duration-200"
        >
            <ArrowLeft class="mr-2 h-5 w-5" />
            Back to Procedure Selection
        </button>

        <div class="flex items-center space-x-8">
            <button
                on:click={downloadMasks}
                class="btn-outline interactive-btn bg-primary-500 flex items-center rounded-md p-2"
            >
                <Download class="mr-2 h-4 w-4" />
                Download Annotations
            </button>
            <button on:click={clearAnnotations} class="interactive-btn flex items-center rounded-md bg-red-400 p-2">
                <Trash2 class="mr-2 h-4 w-4" />
                Clear All
            </button>
        </div>
    </div>

    <!-- Processing Progress -->
    {#if isProcessing}
        <div class="card-compact">
            <div class="flex items-center space-x-4">
                <div class="flex-shrink-0">
                    <div class="bg-primary-500 flex h-8 w-8 items-center justify-center rounded-full">
                        <Sparkles class="h-4 w-4 animate-spin text-white" />
                    </div>
                </div>
                <div class="flex-1">
                    <p class="text-medical-900 text-sm font-medium">Processing with SAM2...</p>
                    <div class="bg-medical-200 mt-2 h-2 w-full rounded-full">
                        <div
                            class="bg-primary-500 h-2 rounded-full transition-all duration-300 ease-out"
                            style="width: {processingProgress}%"
                        ></div>
                    </div>
                    <p class="text-medical-600 mt-1 text-xs">{Math.round(processingProgress)}% complete</p>
                </div>
            </div>
        </div>
    {/if}

    <div class="grid gap-8 lg:grid-cols-4">
        <!-- Video Player and Annotation Canvas -->
        <div class="space-y-6 lg:col-span-3">
            <!-- Main Video Section -->
            <div class="card">
                <div class="mb-6">
                    <div class="flex items-center justify-between">
                        <div>
                            <h3 class="text-medical-900 mb-2 text-xl font-bold">Video Annotation</h3>
                            <p class="text-medical-600 text-sm">
                                {uploadedVideo.name} • {selectedProcedure
                                    .replace('-', ' ')
                                    .replace(/\b\w/g, (l) => l.toUpperCase())}
                            </p>
                        </div>
                        <div class="flex items-center space-x-2">
                            <button
                                on:click={togglePreview}
                                class="interactive-btn bg-medical-100 hover:bg-medical-200 text-medical-700 flex items-center rounded-lg px-3 py-2 text-sm transition-colors"
                            >
                                {#if showPreview}
                                    <EyeOff class="mr-2 h-4 w-4" />
                                    Hide Preview
                                {:else}
                                    <Eye class="mr-2 h-4 w-4" />
                                    Show Preview
                                {/if}
                            </button>
                        </div>
                    </div>
                </div>

                <div class="video-container relative mb-6">
                    <video
                        bind:this={videoElement}
                        src={videoUrl}
                        class="h-auto w-full"
                        on:timeupdate={handleTimeUpdate}
                        on:loadedmetadata={handleLoadedMetadata}
                        on:play={handlePlay}
                        on:pause={handlePause}
                        controls={false}
                    ></video>

                    {#if annotationMode}
                        <AnnotationCanvas
                            bind:this={annotationCanvas}
                            {videoElement}
                            annotationMode={currentAnnotationType}
                            on:annotation-created={handleAnnotationCreated}
                        />
                    {/if}
                </div>

                <!-- Video Controls -->
                <div class="control-panel space-y-6">
                    <!-- Timeline -->
                    <div class="flex items-center space-x-4">
                        <span class="text-medical-600 w-16 text-sm font-medium">{formatTime(currentTime)}</span>
                        <div class="relative flex-1">
                            <input
                                type="range"
                                min="0"
                                max={duration || 0}
                                value={currentTime}
                                on:input={handleSeek}
                                class="bg-medical-200 slider h-2 w-full cursor-pointer appearance-none rounded-lg"
                            />
                        </div>
                        <span class="text-medical-600 w-16 text-sm font-medium">{formatTime(duration)}</span>
                    </div>

                    <!-- Playback Controls -->
                    <div class="flex items-center justify-center space-x-6">
                        <button
                            on:click={seekBackward}
                            class="interactive-btn text-medical-600 hover:text-medical-800 hover:bg-medical-100 rounded-full p-3 transition-all duration-200"
                        >
                            <SkipBack class="h-5 w-5" />
                        </button>

                        <button
                            on:click={togglePlay}
                            class="interactive-btn bg-primary-500 hover:bg-primary-600 rounded-full p-4 text-white shadow-lg transition-all duration-200"
                        >
                            {#if isPlaying}
                                <Pause class="h-6 w-6" />
                            {:else}
                                <Play class="ml-1 h-6 w-6" />
                            {/if}
                        </button>

                        <button
                            on:click={seekForward}
                            class="interactive-btn text-medical-600 hover:text-medical-800 hover:bg-medical-100 rounded-full p-3 transition-all duration-200"
                        >
                            <SkipForward class="h-5 w-5" />
                        </button>
                    </div>

                    <!-- Annotation Controls -->
                    <div class="border-medical-200 flex items-center justify-center space-x-4 border-t pt-6">
                        <button
                            on:click={toggleAnnotationMode}
                            class="interactive-btn flex items-center px-6 py-3 {annotationMode
                                ? 'bg-primary-500 text-white shadow-lg'
                                : 'bg-medical-100 text-medical-700 hover:bg-medical-200'} rounded-xl font-medium transition-all duration-200"
                        >
                            {#if annotationMode}
                                <Square class="mr-2 h-4 w-4" />
                                Exit Annotation
                            {:else}
                                <MousePointer class="mr-2 h-4 w-4" />
                                Start Annotation
                            {/if}
                        </button>

                        {#if annotationMode}
                            <div class="animate-fade-in flex items-center space-x-2">
                                <button
                                    on:click={() => setAnnotationType('positive')}
                                    class="interactive-btn flex items-center px-4 py-3 {currentAnnotationType ===
                                    'positive'
                                        ? 'bg-green-500 text-white shadow-md'
                                        : 'bg-green-100 text-green-700 hover:bg-green-200'}
								rounded-xl text-sm font-medium transition-all duration-200"
                                >
                                    <Plus class="mr-1 h-3 w-3" />
                                    Positive
                                </button>

                                <button
                                    on:click={() => setAnnotationType('negative')}
                                    class="interactive-btn flex items-center px-4 py-3 {currentAnnotationType ===
                                    'negative'
                                        ? 'bg-red-500 text-white shadow-md'
                                        : 'bg-red-100 text-red-700 hover:bg-red-200'}
								rounded-xl text-sm font-medium transition-all duration-200"
                                >
                                    <Minus class="mr-1 h-3 w-3" />
                                    Negative
                                </button>

                                <button
                                    on:click={() => setAnnotationType('box')}
                                    class="interactive-btn flex items-center px-4 py-3 {currentAnnotationType === 'box'
                                        ? 'bg-blue-500 text-white shadow-md'
                                        : 'bg-blue-100 text-blue-700 hover:bg-blue-200'}
								rounded-xl text-sm font-medium transition-all duration-200"
                                >
                                    <Square class="mr-1 h-3 w-3" />
                                    Box
                                </button>
                            </div>
                        {/if}
                    </div>
                </div>
            </div>

            <!-- SAM2 Preview Section -->
            {#if showPreview}
                <div class="card animate-slide-up">
                    <div class="mb-6">
                        <div class="flex items-center space-x-3">
                            <div
                                class="bg-secondary-100 text-secondary-600 flex h-8 w-8 items-center justify-center rounded-lg"
                            >
                                <Layers class="h-4 w-4" />
                            </div>
                            <div>
                                <h3 class="text-medical-900 text-lg font-semibold">SAM2 Output Preview</h3>
                                <p class="text-medical-600 text-sm">Real-time segmentation results</p>
                            </div>
                        </div>
                    </div>

                    <div class="video-container">
                        <video
                            bind:this={previewVideoElement}
                            src={videoUrl}
                            class="h-auto w-full opacity-50"
                            controls={false}
                            muted
                        >
                            <track kind="captions" />
                        </video>
                    </div>
                </div>
            {/if}
        </div>

        <!-- Sidebar Controls -->
        <div class="space-y-6">
            <MaskOptions
                {selectedProcedure}
                bind:selectedOption={selectedMaskOption}
                on:mask-option-changed={handleMaskOptionChange}
            />

            <!-- Annotation Instructions -->
            {#if annotationMode}
                <div class="card-compact animate-fade-in">
                    <h4 class="text-medical-900 mb-4 flex items-center font-semibold">
                        <MousePointer class="mr-2 h-4 w-4" />
                        Annotation Guide
                    </h4>
                    <div class="space-y-3 text-sm">
                        <div class="flex items-start rounded-lg border-l-4 border-green-500 bg-green-50 p-3">
                            <div
                                class="mt-0.5 mr-3 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-green-500"
                            >
                                <Plus class="h-3 w-3 text-white" />
                            </div>
                            <div>
                                <p class="font-medium text-green-800">Positive Points</p>
                                <p class="text-green-700">Click to mark areas to include in the mask</p>
                            </div>
                        </div>

                        <div class="flex items-start rounded-lg border-l-4 border-red-500 bg-red-50 p-3">
                            <div
                                class="mt-0.5 mr-3 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-red-500"
                            >
                                <Minus class="h-3 w-3 text-white" />
                            </div>
                            <div>
                                <p class="font-medium text-red-800">Negative Points</p>
                                <p class="text-red-700">Click to mark areas to exclude from the mask</p>
                            </div>
                        </div>

                        <div class="flex items-start rounded-lg border-l-4 border-blue-500 bg-blue-50 p-3">
                            <div class="mt-1 mr-3 h-5 w-5 flex-shrink-0 rounded bg-blue-500"></div>
                            <div>
                                <p class="font-medium text-blue-800">Bounding Box</p>
                                <p class="text-blue-700">Drag to create a box around the target object</p>
                            </div>
                        </div>
                    </div>
                </div>
            {/if}

            <!-- Annotations List -->
            {#if annotations.length > 0}
                <div class="card-compact animate-slide-up">
                    <h4 class="text-medical-900 mb-4 flex items-center justify-between font-semibold">
                        <span>Annotations</span>
                        <span class="bg-primary-100 text-primary-700 rounded-full px-2 py-1 text-sm"
                            >{annotations.length}</span
                        >
                    </h4>
                    <div class="custom-scrollbar max-h-64 space-y-2 overflow-y-auto">
                        {#each annotations as annotation, index}
                            <div
                                class="bg-medical-50 hover:bg-medical-100 flex items-center justify-between rounded-lg border p-3 text-sm transition-colors"
                            >
                                <div class="flex items-center">
                                    {#if annotation.type === 'positive_point'}
                                        <div
                                            class="mr-3 flex h-4 w-4 items-center justify-center rounded-full bg-green-500"
                                        >
                                            <Plus class="h-2 w-2 text-white" />
                                        </div>
                                        <span class="font-medium">Positive Point</span>
                                    {:else if annotation.type === 'negative_point'}
                                        <div
                                            class="mr-3 flex h-4 w-4 items-center justify-center rounded-full bg-red-500"
                                        >
                                            <Minus class="h-2 w-2 text-white" />
                                        </div>
                                        <span class="font-medium">Negative Point</span>
                                    {:else if annotation.type === 'box'}
                                        <div class="mr-3 h-4 w-4 rounded bg-blue-500"></div>
                                        <span class="font-medium">Bounding Box</span>
                                    {/if}
                                </div>
                                <span class="text-medical-500 text-xs">{formatTime(annotation.timestamp || 0)}</span>
                            </div>
                        {/each}
                    </div>
                </div>
            {/if}
        </div>
    </div>
</div>

<style>
    .slider {
        background: linear-gradient(
            to right,
            #3b82f6 0%,
            #3b82f6 var(--progress, 0%),
            #e2e8f0 var(--progress, 0%),
            #e2e8f0 100%
        );
    }

    .slider::-webkit-slider-thumb {
        appearance: none;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #3b82f6;
        cursor: pointer;
        border: 2px solid white;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }

    .slider::-webkit-slider-thumb:hover {
        transform: scale(1.2);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    .slider::-moz-range-thumb {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #3b82f6;
        cursor: pointer;
        border: 2px solid white;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    }

    .interactive-btn {
        transition: all 0.15s ease-in-out;
    }

    .interactive-btn:hover {
        transform: scale(1.05);
    }

    .interactive-btn:active {
        transform: scale(0.95);
    }

    .interactive-btn:disabled {
        transform: none;
        opacity: 0.6;
        cursor: not-allowed;
    }
</style>
