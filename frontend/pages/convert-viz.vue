<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
      <h1 class="font-bold text-lg text-dark">Convert & Visualize Masks</h1>
      <p class="text-sm text-dark/50">Generate YOLO-seg polygon masks, then visualize samples</p>
    </div>

    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
      <div class="flex items-center gap-3 mb-4">
        <span class="w-7 h-7 rounded-full bg-tertiary text-white flex items-center justify-center text-xs font-bold">2</span>
        <h2 class="font-bold text-dark">Convert to YOLO-seg & Visualize</h2>
        <div class="ml-auto flex gap-2">
          <button :disabled="convertRunning" class="px-4 py-1.5 text-xs font-bold rounded-lg bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runConvert">
            {{ convertRunning ? 'Converting...' : 'Convert Masks' }}
          </button>
          <button :disabled="vizRunning" class="px-4 py-1.5 text-xs font-bold rounded-lg bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runViz">
            {{ vizRunning ? 'Loading...' : 'Visualize Samples' }}
          </button>
        </div>
      </div>
      <div v-if="convertResult || vizResult" class="bg-gray-50 rounded-lg p-3 text-xs font-mono max-h-48 overflow-auto">
        <pre v-if="convertResult" class="text-green-700">{{ convertResult }}</pre>
        <pre v-if="vizResult" class="text-green-700">{{ vizResult }}</pre>
      </div>
      <div v-if="vizImages.length" class="mt-3 grid grid-cols-6 gap-2">
        <img v-for="url in vizImages" :key="url" :src="`${apiBase}${url}`" class="w-full aspect-square object-cover rounded border" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase
const { show: showError } = useToast()
const convertResult = ref(null); const convertRunning = ref(false)
const vizResult = ref(null); const vizRunning = ref(false); const vizImages = ref([])

async function runConvert() {
  convertRunning.value = true; convertResult.value = null
  try {
    const res = await $fetch("/api/kaggle/convert", { baseURL: apiBase, method: "POST" }) as any
    convertResult.value = `Skipped ${res.skipped_existing} existing, generated ${res.generated} new`
  } catch (e: any) { showError(e?.data?.detail || e?.message) } finally { convertRunning.value = false }
}
async function runViz() {
  vizRunning.value = true; vizResult.value = null; vizImages.value = []
  try {
    const res = await $fetch("/api/kaggle/viz", { baseURL: apiBase }) as any
    if (res.images?.length) { vizImages.value = res.images; vizResult.value = `${res.images.length} samples with mask overlays` }
    else { vizResult.value = "No viz available" }
  } catch (e: any) { showError(e?.data?.detail || e?.message) } finally { vizRunning.value = false }
}
</script>
