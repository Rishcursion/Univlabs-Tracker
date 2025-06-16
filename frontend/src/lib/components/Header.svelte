<script lang="ts">
	import { Activity, Zap } from '@lucide/svelte';
</script>

<header class="border-medical-200 border-b bg-white shadow-sm">
	<div class="container mx-auto px-4 py-4">
		<div class="flex items-center justify-between">
			<div class="flex items-center space-x-3">
				<div class="bg-primary-500 flex h-10 w-10 items-center justify-center rounded-lg">
					<Activity class="h-6 w-6 text-white" />
				</div>
				<div>
					<h1 class="text-medical-900 text-xl font-bold">SAM2 Surgical Annotation</h1>
					<p class="text-medical-600 text-sm">Advanced Video Segmentation for Medical Procedures</p>
				</div>
			</div>

			<div class="text-medical-600 flex items-center space-x-2 text-sm">
				<Zap class="h-4 w-4" />
				<span>Powered by Segment Anything Model 2</span>
			</div>
		</div>
	</div>
</header>
