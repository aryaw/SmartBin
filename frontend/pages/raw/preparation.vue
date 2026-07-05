<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
      <h1 class="font-bold text-lg text-dark">Preparation</h1>
      <p class="text-sm text-dark/50">Convert to YOLO-seg format and visualize polygon masks</p>
    </div>

    <div class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
      <table class="w-full text-xs">
        <thead class="bg-gray-50 border-b">
          <tr>
            <th class="text-left px-4 py-3 font-semibold text-dark/60">Process</th>
            <th class="text-left px-4 py-3 font-semibold text-dark/60">Description</th>
            <th class="text-center px-4 py-3 font-semibold text-dark/60">Status</th>
            <th class="text-left px-4 py-3 font-semibold text-dark/60">Summary</th>
            <th class="text-center px-4 py-3 font-semibold text-dark/60">Action</th>
          </tr>
        </thead>
        <tbody class="divide-y">
          <tr>
            <td class="px-4 py-3 font-medium">Convert to YOLO-seg</td>
            <td class="px-4 py-3 text-dark/60">Generate train/val/test splits, polygon labels, and data.yaml</td>
            <td class="px-4 py-3 text-center">
              <span v-if="convertStatus === 'done'" class="text-green-600 bg-green-50 px-2 py-0.5 rounded">Done</span>
              <span v-else-if="convertStatus === 'running'" class="text-blue-600 bg-blue-50 px-2 py-0.5 rounded">Running</span>
              <span v-else class="text-gray-400 bg-gray-50 px-2 py-0.5 rounded">Not Started</span>
            </td>
            <td class="px-4 py-3 text-dark/60">{{ convertSummary }}</td>
            <td class="px-4 py-3 text-center">
              <button :disabled="convertStatus === 'running'" class="px-3 py-1 text-xs font-bold rounded bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runConvert">Convert Dataset</button>
            </td>
          </tr>
          <tr>
            <td class="px-4 py-3 font-medium">Visualize Polygon Masks</td>
            <td class="px-4 py-3 text-dark/60">Generate sample overlays for polygon inspection</td>
            <td class="px-4 py-3 text-center">
              <span v-if="vizStatus === 'done'" class="text-green-600 bg-green-50 px-2 py-0.5 rounded">Done</span>
              <span v-else-if="vizStatus === 'running'" class="text-blue-600 bg-blue-50 px-2 py-0.5 rounded">Running</span>
              <span v-else class="text-gray-400 bg-gray-50 px-2 py-0.5 rounded">Not Started</span>
            </td>
            <td class="px-4 py-3 text-dark/60">{{ vizSummary }}</td>
            <td class="px-4 py-3 text-center">
              <div class="flex items-center justify-center gap-1">
                <button :disabled="vizStatus === 'running'" class="px-3 py-1 text-xs font-bold rounded bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runViz">Generate Preview</button>
                <button v-if="vizImages.length" class="px-3 py-1 text-xs font-bold rounded bg-gray-100 text-dark/50 hover:bg-gray-200" @click="showPreview = !showPreview">View Preview</button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="showPreview && vizImages.length" class="grid grid-cols-6 gap-2">
      <img v-for="url in vizImages" :key="url" :src="`${apiBase}${url}`" class="w-full aspect-square object-cover rounded border" />
    </div>
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase
const { show: showError } = useToast()
const convertStatus = ref('not_started'); const convertSummary = ref('')
const vizStatus = ref('not_started'); const vizSummary = ref(''); const vizImages = ref([]); const showPreview = ref(false)

async function runConvert() {
  convertStatus.value = 'running'
  try {
    const res = await $fetch("/api/kaggle/convert", { baseURL: apiBase, method: "POST" }) as any
    convertSummary.value = `Skipped ${res.skipped_existing} existing, generated ${res.generated} new`
    convertStatus.value = 'done'
  } catch (e: any) { showError(e?.data?.detail || e?.message); convertStatus.value = 'not_started' }
}
async function runViz() {
  vizStatus.value = 'running'; vizImages.value = []
  try {
    const res = await $fetch("/api/kaggle/viz", { baseURL: apiBase }) as any
    if (res.images?.length) { vizImages.value = res.images; vizSummary.value = `${res.images.length} sample previews`; vizStatus.value = 'done' }
    else { vizSummary.value = 'No previews available'; vizStatus.value = 'done' }
  } catch (e: any) { showError(e?.data?.detail || e?.message); vizStatus.value = 'not_started' }
}
</script>
