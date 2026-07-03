<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex items-center justify-between">
      <div>
        <h1 class="font-bold text-lg text-dark">YOLO Segmentation (Train Inference)</h1>
        <p class="text-sm text-dark/50">{{ filteredItems.length }} predicted images</p>
      </div>
      <button :disabled="pipelineRunning"
        class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm transition-all flex items-center gap-2 disabled:opacity-50"
        @click="runPipeline">
        <svg v-if="pipelineRunning" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
        </svg>
        {{ pipelineRunning ? 'Training...' : 'Train YOLO Seg' }}
      </button>
    </div>

    <div v-if="loading" class="flex justify-center py-12">
      <svg class="animate-spin h-8 w-8 text-tertiary" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
    </div>

    <div v-else-if="!filteredItems.length" class="bg-white rounded-xl p-10 text-center border border-gray-100">
      <p class="text-dark/50 text-sm">No YOLO segmentation predictions. Click "Train YOLO Seg" above.</p>
    </div>

    <template v-else>
      <div class="space-y-4">
        <div v-for="img in paginatedItems" :key="img.filename"
          class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
          <div class="flex flex-col lg:flex-row">
            <div class="lg:w-72 p-3">
              <img :src="`${apiBase}${img.viz_url}`"
                class="w-full aspect-square object-cover rounded-lg border-2 border-purple-200 cursor-pointer hover:opacity-90 transition-opacity"
                @click="zoomUrl = `${apiBase}${img.viz_url}`" />
              <p class="text-xs font-medium text-dark mt-2 truncate">{{ img.filename }}</p>
              <p class="text-xs text-dark/40">{{ img.predictions.length }} object(s)</p>
            </div>
            <div class="flex-1 overflow-x-auto p-3">
              <table class="w-full text-xs">
                <thead>
                  <tr class="text-dark/50 uppercase tracking-wide border-b border-gray-100">
                    <th class="p-2 text-left font-medium">#</th>
                    <th class="p-2 text-left font-medium">Class</th>
                    <th class="p-2 text-left font-medium">Category</th>
                    <th class="p-2 text-right font-medium">Conf</th>
                    <th class="p-2 text-right font-medium">Points</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(pred, j) in img.predictions" :key="j" class="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                    <td class="p-2 text-dark/40 font-mono">{{ j + 1 }}</td>
                    <td class="p-2 font-medium text-dark">{{ pred.class_id }}</td>
                    <td class="p-2">
                      <span :class="pred.category === 'Organik' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'"
                        class="font-bold px-2 py-0.5 rounded text-xs">{{ pred.category }}</span>
                    </td>
                    <td class="p-2 text-right font-mono text-purple-600">{{ (pred.confidence * 100).toFixed(0) }}%</td>
                    <td class="p-2 text-right font-mono text-purple-600">{{ pred.segmentation?.length || 0 }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
      <Pagination :page="page" :total="filteredItems.length" :per="perPage" @update:page="page = $event" />
    </template>

    <ZoomModal :url="zoomUrl" @close="zoomUrl = null" />
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase

const data = ref<any>(null)
const loading = ref(true)
const pipelineRunning = ref(false)
const page = ref(1)
const perPage = 10
const zoomUrl = ref<string | null>(null)

const filteredItems = computed(() => (data.value?.train_seg || []).filter((img: any) => img.viz_url && img.predictions.length > 0))

const paginatedItems = computed(() => {
  const start = (page.value - 1) * perPage
  return filteredItems.value.slice(start, start + perPage)
})

async function load() {
  loading.value = true
  try {
    data.value = await $fetch("/api/dataset/grid", { baseURL: apiBase })
  } catch {} finally { loading.value = false }
}

async function runPipeline() {
  pipelineRunning.value = true
  try {
    await $fetch("/api/dataset/pipeline/yolo/seg", { baseURL: apiBase, method: "POST" })
    await load()
  } catch {} finally { pipelineRunning.value = false }
}

onMounted(load)
</script>
