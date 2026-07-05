<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
      <h1 class="font-bold text-lg text-dark">Dataset Preparation & Profiling</h1>
      <p class="text-sm text-dark/50">Load dataset, then run profiling to see class distribution</p>
    </div>

    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
      <div class="flex items-center gap-3 mb-4">
        <span class="w-7 h-7 rounded-full bg-tertiary text-white flex items-center justify-center text-xs font-bold">1</span>
        <h2 class="font-bold text-dark">Load Dataset & Profiling</h2>
        <div class="ml-auto flex gap-2">
          <button :disabled="loadRunning" class="px-4 py-1.5 text-xs font-bold rounded-lg bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runLoad">
            {{ loadRunning ? 'Loading...' : 'Load Dataset' }}
          </button>
          <button :disabled="profileRunning" class="px-4 py-1.5 text-xs font-bold rounded-lg bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runProfile">
            {{ profileRunning ? 'Scanning...' : 'Dataset Profiling' }}
          </button>
        </div>
      </div>
      <div v-if="loadResult || profileResult" class="bg-gray-50 rounded-lg p-3 text-xs font-mono max-h-48 overflow-auto">
        <pre v-if="loadResult" class="text-green-700">{{ loadResult }}</pre>
        <pre v-if="profileResult" class="text-green-700">{{ profileResult }}</pre>
      </div>
      <div v-if="profileImages.length" class="mt-3 grid grid-cols-6 gap-2">
        <img v-for="url in profileImages" :key="url" :src="`${apiBase}${url}`" class="w-full aspect-square object-cover rounded border" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase
const { show: showError } = useToast()
const loadResult = ref(null); const loadRunning = ref(false)
const profileResult = ref(null); const profileRunning = ref(false); const profileImages = ref([])

async function runLoad() {
  loadRunning.value = true; loadResult.value = null
  try {
    const res = await $fetch("/api/kaggle/download?source=local", { baseURL: apiBase, method: "POST" }) as any
    loadResult.value = `Loaded ${res.total_images} images, ${res.classes} classes. Edge: ${res.edge_masks}, Fallback: ${res.fallback_masks}`
  } catch (e: any) { showError(e?.data?.detail || e?.message) } finally { loadRunning.value = false }
}
async function runProfile() {
  profileRunning.value = true; profileResult.value = null; profileImages.value = []
  try {
    const res = await $fetch("/api/kaggle/explore", { baseURL: apiBase }) as any
    let out = `${res.total_images} images, ${res.num_classes} classes\n\n`
    const maxC = Math.max(...res.distribution.map((d: any) => d.count))
    for (const d of res.distribution) {
      const bar = '█'.repeat(Math.round(d.count / maxC * 25))
      out += `  [${d.class_id}] ${d.name.padEnd(25)} ${String(d.count).padStart(4)} ${bar}\n`
    }
    profileResult.value = out
    profileImages.value = res.distribution.filter((d: any) => d.sample_url).map((d: any) => d.sample_url)
  } catch (e: any) { showError(e?.data?.detail || e?.message) } finally { profileRunning.value = false }
}
</script>
