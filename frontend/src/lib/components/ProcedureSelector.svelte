<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { ArrowLeft, Scissors, Clock, Users } from '@lucide/svelte';
	
	const dispatch = createEventDispatcher<{
		'procedure-selected': string;
		'back': void;
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

<div class="max-w-4xl mx-auto">
	<div class="flex items-center mb-6">
		<button 
			on:click={handleBack}
			class="interactive-btn flex items-center text-medical-600 hover:text-medical-800 transition-colors"
		>
			<ArrowLeft class="w-5 h-5 mr-2" />
			Back to Upload
		</button>
	</div>
	
	<div class="card">
		<div class="mb-6">
			<h2 class="text-2xl font-bold text-medical-900 mb-2">Select Procedure Type</h2>
			<p class="text-medical-600">
				Choose the surgical procedure being performed in the uploaded video: 
				<span class="font-medium text-medical-800">{uploadedVideo.name}</span>
			</p>
		</div>
		
		<div class="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
			{#each procedures as procedure}
				<div 
					class="interactive-btn relative border-2 rounded-lg p-4 cursor-pointer transition-all duration-200 {
						procedure.available 
							? selectedProcedure === procedure.id 
								? 'border-primary-500 bg-primary-50' 
								: 'border-medical-200 hover:border-primary-300 hover:bg-primary-25'
							: 'border-medical-200 bg-medical-50 opacity-60 cursor-not-allowed'
					}"
					on:click={() => procedure.available && selectProcedure(procedure.id)}
					role="button"
					tabindex="0"
					on:keydown={(e) => e.key === 'Enter' && procedure.available && selectProcedure(procedure.id)}
				>
					{#if !procedure.available}
						<div class="absolute top-2 right-2 bg-medical-400 text-white text-xs px-2 py-1 rounded">
							Coming Soon
						</div>
					{/if}
					
					<div class="flex items-center mb-3">
						<div class="flex items-center justify-center w-10 h-10 {procedure.available ? 'bg-primary-100 text-primary-600' : 'bg-medical-200 text-medical-500'} rounded-lg mr-3">
							<svelte:component this={procedure.icon} class="w-5 h-5" />
						</div>
						<h3 class="font-semibold text-medical-900">{procedure.name}</h3>
					</div>
					
					<p class="text-sm text-medical-600 mb-3">{procedure.description}</p>
					
					<div class="flex items-center justify-between text-xs text-medical-500">
						<div class="flex items-center">
							<Clock class="w-3 h-3 mr-1" />
							{procedure.duration}
						</div>
						<div class="flex items-center">
							<Users class="w-3 h-3 mr-1" />
							{procedure.complexity}
						</div>
					</div>
					
					{#if selectedProcedure === procedure.id}
						<div class="absolute top-2 left-2 w-4 h-4 bg-primary-500 rounded-full flex items-center justify-center">
							<div class="w-2 h-2 bg-white rounded-full"></div>
						</div>
					{/if}
				</div>
			{/each}
		</div>
		
		{#if selectedProcedure}
			<div class="flex justify-end">
				<button on:click={handleContinue} class="btn-primary interactive-btn">
					Continue to Annotation
				</button>
			</div>
		{/if}
	</div>
</div>
