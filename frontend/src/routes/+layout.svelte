<script lang="ts">
	import '../app.css';
	import { page } from '$app/stores';
	let { children } = $props();
	import Header from '$lib/components/Header.svelte';
</script>

<div class="bg-medical-50 min-h-screen">
	<Header />
	<main class="container mx-auto px-4 py-8">{@render children()}</main>
</div>
