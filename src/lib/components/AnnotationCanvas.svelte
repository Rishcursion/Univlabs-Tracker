<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';

	const dispatch = createEventDispatcher<{
		'annotation-created': any;
	}>();

	export let videoElement: HTMLVideoElement;
	export let annotationMode: 'positive' | 'negative' | 'box' = 'positive';

	let canvas: HTMLCanvasElement;
	let ctx: CanvasRenderingContext2D;
	let isDrawing = false;
	let startX = 0;
	let startY = 0;
	let annotations: any[] = [];

	onMount(() => {
		if (canvas) {
			ctx = canvas.getContext('2d')!;
			resizeCanvas();

			// Resize canvas when video dimensions change
			const resizeObserver = new ResizeObserver(resizeCanvas);
			resizeObserver.observe(videoElement);

			return () => {
				resizeObserver.disconnect();
			};
		}
	});

	function resizeCanvas() {
		if (canvas && videoElement) {
			const rect = videoElement.getBoundingClientRect();
			canvas.width = rect.width;
			canvas.height = rect.height;
			canvas.style.width = rect.width + 'px';
			canvas.style.height = rect.height + 'px';
			redrawAnnotations();
		}
	}

	// Mock API call to send annotation to backend
	async function sendAnnotationToBackend(annotation: any) {
		try {
			console.log('Sending annotation to backend:', annotation);

			// Simulate API call
			const response = await fetch('/api/annotations', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					annotation,
					videoId: 'current-video-id',
					timestamp: Date.now(),
					sessionId: 'current-session-id'
				})
			});

			// Since this is a mock, we'll simulate the response
			if (!response.ok) {
				throw new Error('Failed to send annotation');
			}

			console.log('Annotation sent successfully');
		} catch (error) {
			console.error('Error sending annotation to backend:', error);
			// In a real app, you might want to show a toast notification or retry
		}
	}

	function handleMouseDown(event: MouseEvent) {
		if (annotationMode === 'box') {
			const rect = canvas.getBoundingClientRect();
			startX = event.clientX - rect.left;
			startY = event.clientY - rect.top;
			isDrawing = true;
		}
	}

	function handleMouseMove(event: MouseEvent) {
		if (!isDrawing || annotationMode !== 'box') return;

		const rect = canvas.getBoundingClientRect();
		const currentX = event.clientX - rect.left;
		const currentY = event.clientY - rect.top;

		// Clear canvas and redraw existing annotations
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		redrawAnnotations();

		// Draw current rectangle
		drawRect(startX, startY, currentX - startX, currentY - startY, '#3b82f6', false);
	}

	async function handleMouseUp(event: MouseEvent) {
		if (!isDrawing || annotationMode !== 'box') return;

		const rect = canvas.getBoundingClientRect();
		const endX = event.clientX - rect.left;
		const endY = event.clientY - rect.top;

		// Create box annotation
		const annotation = {
			type: 'box',
			x: Math.min(startX, endX),
			y: Math.min(startY, endY),
			width: Math.abs(endX - startX),
			height: Math.abs(endY - startY),
			timestamp: videoElement.currentTime,
			id: Date.now()
		};

		// Only add if rectangle has minimum size
		if (annotation.width > 10 && annotation.height > 10) {
			annotations = [...annotations, annotation];
			dispatch('annotation-created', annotation);

			// Send to backend
			await sendAnnotationToBackend(annotation);
		}

		isDrawing = false;
		redrawAnnotations();
	}

	async function handleClick(event: MouseEvent) {
		if (isDrawing || annotationMode === 'box') return;

		const rect = canvas.getBoundingClientRect();
		const x = event.clientX - rect.left;
		const y = event.clientY - rect.top;

		// Create point annotation (positive or negative)
		const annotation = {
			type: annotationMode === 'positive' ? 'positive_point' : 'negative_point',
			x,
			y,
			timestamp: videoElement.currentTime,
			id: Date.now()
		};

		annotations = [...annotations, annotation];
		dispatch('annotation-created', annotation);

		// Send to backend
		await sendAnnotationToBackend(annotation);

		redrawAnnotations();
	}

	function drawRect(
		x: number,
		y: number,
		width: number,
		height: number,
		color: string,
		filled: boolean = false
	) {
		ctx.strokeStyle = color;
		ctx.lineWidth = 2;

		if (filled) {
			ctx.fillStyle = color + '33'; // Add transparency
			ctx.fillRect(x, y, width, height);
		}

		ctx.strokeRect(x, y, width, height);
	}

	function drawPositivePoint(x: number, y: number) {
		// Draw green circle for positive points
		ctx.fillStyle = '#10b981';
		ctx.beginPath();
		ctx.arc(x, y, 8, 0, 2 * Math.PI);
		ctx.fill();

		// Draw white border
		ctx.strokeStyle = 'white';
		ctx.lineWidth = 2;
		ctx.beginPath();
		ctx.arc(x, y, 8, 0, 2 * Math.PI);
		ctx.stroke();

		// Draw plus sign
		ctx.strokeStyle = 'white';
		ctx.lineWidth = 2;
		ctx.beginPath();
		ctx.moveTo(x - 4, y);
		ctx.lineTo(x + 4, y);
		ctx.moveTo(x, y - 4);
		ctx.lineTo(x, y + 4);
		ctx.stroke();
	}

	function drawNegativePoint(x: number, y: number) {
		// Draw red circle for negative points
		ctx.fillStyle = '#ef4444';
		ctx.beginPath();
		ctx.arc(x, y, 8, 0, 2 * Math.PI);
		ctx.fill();

		// Draw white border
		ctx.strokeStyle = 'white';
		ctx.lineWidth = 2;
		ctx.beginPath();
		ctx.arc(x, y, 8, 0, 2 * Math.PI);
		ctx.stroke();

		// Draw minus sign
		ctx.strokeStyle = 'white';
		ctx.lineWidth = 2;
		ctx.beginPath();
		ctx.moveTo(x - 4, y);
		ctx.lineTo(x + 4, y);
		ctx.stroke();
	}

	function redrawAnnotations() {
		if (!ctx) return;

		ctx.clearRect(0, 0, canvas.width, canvas.height);

		annotations.forEach((annotation) => {
			if (annotation.type === 'box') {
				drawRect(annotation.x, annotation.y, annotation.width, annotation.height, '#3b82f6', true);
			} else if (annotation.type === 'positive_point') {
				drawPositivePoint(annotation.x, annotation.y);
			} else if (annotation.type === 'negative_point') {
				drawNegativePoint(annotation.x, annotation.y);
			}
		});
	}

	// Export function to clear annotations
	export function clearAnnotations() {
		annotations = [];
		redrawAnnotations();
	}

	// Export function to get annotations
	export function getAnnotations() {
		return annotations;
	}
</script>

<canvas
	bind:this={canvas}
	class="absolute inset-0 z-10 h-full w-full cursor-crosshair"
	on:mousedown={handleMouseDown}
	on:mousemove={handleMouseMove}
	on:mouseup={handleMouseUp}
	on:click={handleClick}
	style="pointer-events: auto;"
></canvas>
