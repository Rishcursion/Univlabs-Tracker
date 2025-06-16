<script lang="ts">
    import { createEventDispatcher } from 'svelte';
    import { Layers, Wand as Wand2, Target } from '@lucide/svelte';

    const dispatch = createEventDispatcher<{
        'mask-option-changed': string;
    }>();

    export let selectedProcedure: string;
    export let selectedOption: string = 'custom';

    const presetMasks = {
        'laparoscopic-cholecystectomy': [
            { id: 'gallbladder', name: 'Gallbladder', color: '#ef4444' },
            { id: 'liver', name: 'Liver', color: '#f97316' },
            { id: 'cystic-artery', name: 'Cystic Artery', color: '#84cc16' },
            { id: 'cystic-duct', name: 'Cystic Duct', color: '#06b6d4' },
            { id: 'instruments', name: 'Surgical Instruments', color: '#8b5cf6' },
        ],
    };

    $: currentPresets = presetMasks[selectedProcedure as keyof typeof presetMasks] || [];

    function selectOption(option: string) {
        selectedOption = option;
        dispatch('mask-option-changed', option);
    }
</script>

<div class="card">
    <h3 class="text-medical-900 mb-4 text-lg font-semibold">SAM2 Prompting Options</h3>

    <div class="space-y-4">
        <!-- Custom Annotation -->
        <div
            class="interactive-btn cursor-pointer rounded-lg border-2 p-4 transition-all duration-200 {selectedOption ===
            'custom'
                ? 'border-primary-500 bg-primary-50'
                : 'border-medical-200 hover:border-primary-300'}"
            on:click={() => selectOption('custom')}
            role="button"
            tabindex="0"
            on:keydown={(e) => e.key === 'Enter' && selectOption('custom')}
        >
            <div class="mb-2 flex items-center">
                <div class="bg-primary-100 text-primary-600 mr-3 flex h-8 w-8 items-center justify-center rounded-lg">
                    <Wand2 class="h-4 w-4" />
                </div>
                <h4 class="text-medical-900 font-medium">Interactive Prompting</h4>
            </div>
            <p class="text-medical-600 text-sm">
                Use positive/negative points and bounding boxes to guide SAM2 segmentation
            </p>
        </div>

        <!-- Preset Masks -->
        {#if currentPresets.length > 0}
            <div
                class="interactive-btn cursor-pointer rounded-lg border-2 p-4 transition-all duration-200 {selectedOption ===
                'preset'
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-medical-200 hover:border-primary-300'}"
                on:click={() => selectOption('preset')}
                role="button"
                tabindex="0"
                on:keydown={(e) => e.key === 'Enter' && selectOption('preset')}
            >
                <div class="mb-2 flex items-center">
                    <div
                        class="bg-secondary-100 text-secondary-600 mr-3 flex h-8 w-8 items-center justify-center rounded-lg"
                    >
                        <Layers class="h-4 w-4" />
                    </div>
                    <h4 class="text-medical-900 font-medium">Anatomical Presets</h4>
                </div>
                <p class="text-medical-600 mb-3 text-sm">Pre-configured prompts for common anatomical structures</p>

                {#if selectedOption === 'preset'}
                    <div class="border-medical-200 mt-3 space-y-2 border-t pt-3">
                        {#each currentPresets as mask}
                            <div class="flex items-center rounded border bg-white p-2">
                                <div class="mr-3 h-4 w-4 rounded-full" style="background-color: {mask.color}"></div>
                                <span class="text-medical-800 text-sm font-medium">{mask.name}</span>
                            </div>
                        {/each}
                    </div>
                {/if}
            </div>
        {/if}

        <!-- Auto-Detection -->
        <div
            class="interactive-btn cursor-pointer rounded-lg border-2 p-4 transition-all duration-200 {selectedOption ===
            'auto'
                ? 'border-primary-500 bg-primary-50'
                : 'border-medical-200 hover:border-primary-300'}"
            on:click={() => selectOption('auto')}
            role="button"
            tabindex="0"
            on:keydown={(e) => e.key === 'Enter' && selectOption('auto')}
        >
            <div class="mb-2 flex items-center">
                <div class="mr-3 flex h-8 w-8 items-center justify-center rounded-lg bg-green-100 text-green-600">
                    <Target class="h-4 w-4" />
                </div>
                <h4 class="text-medical-900 font-medium">Automatic Segmentation</h4>
            </div>
            <p class="text-medical-600 text-sm">Let SAM2 automatically detect and segment all objects in the frame</p>
        </div>
    </div>

    <!-- Options Details -->
    {#if selectedOption === 'custom'}
        <div class="mt-4 rounded-lg bg-blue-50 p-3">
            <p class="text-sm text-blue-800">
                <strong>SAM2 Interactive Mode:</strong> Use positive points (green +) to indicate areas to include, negative
                points (red -) to exclude areas, and bounding boxes to roughly outline objects.
            </p>
        </div>
    {:else if selectedOption === 'preset'}
        <div class="mt-4 rounded-lg bg-teal-50 p-3">
            <p class="text-sm text-teal-800">
                <strong>Anatomical Templates:</strong> Pre-configured prompts optimized for common surgical structures in
                laparoscopic procedures.
            </p>
        </div>
    {:else if selectedOption === 'auto'}
        <div class="mt-4 rounded-lg bg-green-50 p-3">
            <p class="text-sm text-green-800">
                <strong>Everything Mode:</strong> SAM2 will automatically generate masks for all detectable objects without
                manual prompting.
            </p>
        </div>
    {/if}
</div>
