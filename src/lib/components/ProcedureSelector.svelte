<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { ArrowLeft, Scissors, Clock, Users } from '@lucide/svelte';

	const dispatch = createEventDispatcher<{
		'procedure-selected': string;
		back: void;
	}>();

	export let uploadedVideo: File;

	let selectedProcedure = '';

	const procedures = [
		{
			id: 'laparoscopic-cholecystectomy',
			name: 'Laparoscopic Cholecystectomy',
			description: 'Minimally invasive gallbladder removal surgery',
			duration: '30-60 minutes',
			complexity: 'Moderate',
			icon: Scissors,
			available: true
		},
		{
			id: 'appendectomy',
			name: 'Appendectomy',
			description: 'Surgical removal of the appendix',
			duration: '20-40 minutes',
			complexity: 'Low',
			icon: Scissors,
			available: false
		},
		{
			id: 'hernia-repair',
			name: 'Hernia Repair',
			description: 'Laparoscopic hernia repair procedure',
			duration: '45-90 minutes',
			complexity: 'Moderate',
			icon: Scissors,
			available: false
		}
	];

	function selectProcedure(procedureId: string) {
		selectedProcedure = procedureId;
	}

	function handleContinue() {
		if (selectedProcedure) {
			dispatch('procedure-selected', selectedProcedure);
		}
	}

	function handleBack() {
		dispatch('back');
	}
</script>

<div class="mx-auto max-w-4xl">
	<div class="mb-6 flex items-center">
		<button
			on:click={handleBack}
			class="text-medical-600 hover:text-medical-800 flex items-center transition-colors"
		>
			<ArrowLeft class="mr-2 h-5 w-5" />
			Back to Upload
		</button>
	</div>

	<div class="card">
		<div class="mb-6">
			<h2 class="text-medical-900 mb-2 text-2xl font-bold">Select Procedure Type</h2>
			<p class="text-medical-600">
				Choose the surgical procedure being performed in the uploaded video:
				<span class="text-medical-800 font-medium">{uploadedVideo.name}</span>
			</p>
		</div>

		<div class="mb-6 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
			{#each procedures as procedure}
				<div
					class="relative cursor-pointer rounded-lg border-2 p-4 transition-all duration-200 {procedure.available
						? selectedProcedure === procedure.id
							? 'border-primary-500 bg-primary-50'
							: 'border-medical-200 hover:border-primary-300 hover:bg-primary-25'
						: 'border-medical-200 bg-medical-50 cursor-not-allowed opacity-60'}"
					on:click={() => procedure.available && selectProcedure(procedure.id)}
					role="button"
					tabindex="0"
					on:keydown={(e) =>
						e.key === 'Enter' && procedure.available && selectProcedure(procedure.id)}
				>
					{#if !procedure.available}
						<div class="bg-medical-400 absolute top-2 right-2 rounded px-2 py-1 text-xs text-white">
							Coming Soon
						</div>
					{/if}

					<div class="mb-3 flex items-center">
						<div
							class="flex h-10 w-10 items-center justify-center {procedure.available
								? 'bg-primary-100 text-primary-600'
								: 'bg-medical-200 text-medical-500'} mr-3 rounded-lg"
						>
							<svelte:component this={procedure.icon} class="h-5 w-5" />
						</div>
						<h3 class="text-medical-900 font-semibold">{procedure.name}</h3>
					</div>

					<p class="text-medical-600 mb-3 text-sm">{procedure.description}</p>

					<div class="text-medical-500 flex items-center justify-between text-xs">
						<div class="flex items-center">
							<Clock class="mr-1 h-3 w-3" />
							{procedure.duration}
						</div>
						<div class="flex items-center">
							<Users class="mr-1 h-3 w-3" />
							{procedure.complexity}
						</div>
					</div>

					{#if selectedProcedure === procedure.id}
						<div
							class="bg-primary-500 absolute top-2 left-2 flex h-4 w-4 items-center justify-center rounded-full"
						>
							<div class="h-2 w-2 rounded-full bg-white"></div>
						</div>
					{/if}
				</div>
			{/each}
		</div>

		{#if selectedProcedure}
			<div class="flex justify-end">
				<button on:click={handleContinue} class="btn-primary"> Continue to Annotation </button>
			</div>
		{/if}
	</div>
</div>
