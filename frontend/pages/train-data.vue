<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex items-center justify-between">
      <div>
        <h1 class="font-bold text-lg text-dark">Train Data</h1>
        <p class="text-sm text-dark/50">{{ items.length }} images</p>
      </div>
      <div class="flex items-center gap-2">
        <button :disabled="pipelineRunning"
          class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-4 py-2.5 rounded-xl shadow-sm transition-all flex items-center gap-2 disabled:opacity-50"
          @click="runCoco">
          <svg v-if="pipelineRunning && pipeType==='coco'" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
          </svg>
          {{ pipelineRunning && pipeType==='coco' ? 'Generating...' : 'Generate COCO Annotation' }}
        </button>
        <button :disabled="pipelineRunning"
          class="bg-secondary hover:bg-blue-600 text-white text-sm font-bold px-4 py-2.5 rounded-xl shadow-sm transition-all flex items-center gap-2 disabled:opacity-50"
          @click="runYolo">
          <svg v-if="pipelineRunning && pipeType==='yolo'" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
          </svg>
          {{ pipelineRunning && pipeType==='yolo' ? 'Generating...' : 'Generate YOLO Annotation' }}
        </button>
        <button :disabled="pipelineRunning"
          class="bg-purple-600 hover:bg-purple-700 text-white text-sm font-bold px-4 py-2.5 rounded-xl shadow-sm transition-all flex items-center gap-2 disabled:opacity-50"
          @click="runYoloSeg">
          <svg v-if="pipelineRunning && pipeType==='yolo-seg'" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
          </svg>
          {{ pipelineRunning && pipeType==='yolo-seg' ? 'Generating...' : 'Generate YOLO Segmentation' }}
        </button>
      </div>
    </div>

    <div v-if="loading" class="flex justify-center py-12">
      <svg class="animate-spin h-8 w-8 text-tertiary" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
    </div>

    <div v-else-if="!paginatedItems.length" class="bg-white rounded-xl p-10 text-center border border-gray-100">
      <p class="text-dark/50 text-sm">No train images. Run Split on Raw Data page first.</p>
    </div>

    <template v-else>
      <div class="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 gap-3">
        <div v-for="img in paginatedItems" :key="img.filename"
          class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden hover:shadow-md transition-shadow group cursor-pointer"
          @click="zoomUrl = `${apiBase}${img.url}`">
          <div class="aspect-square overflow-hidden bg-gray-50">
            <img :src="`${apiBase}${img.url}`" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
          </div>
          <div class="p-2">
            <p class="text-xs font-medium text-dark truncate">{{ img.filename }}</p>
            <p class="text-xs text-dark/40">{{ img.size_kb }} KB</p>
          </div>
        </div>
      </div>
      <Pagination :page="page" :total="items.length" :per="perPage" @update:page="page = $event" />
    </template>

    <ZoomModal :url="zoomUrl" @close="zoomUrl = null" />
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase

const data = ref<any>(null)
const loading = ref(true)
const page = ref(1)
const perPage = 20
const zoomUrl = ref<string | null>(null)
const pipelineRunning = ref(false)
const pipeType = ref('')

const items = computed(() => data.value?.train_plain || [])
const paginatedItems = computed(() => {
  const start = (page.value - 1) * perPage
  return items.value.slice(start, start + perPage)
})

async function runCoco() {
  pipelineRunning.value = true; pipeType.value = 'coco'
  try {
    await $fetch("/api/dataset/pipeline/coco", { baseURL: apiBase, method: "POST" })
    await load()
  } catch {} finally { pipelineRunning.value = false }
}

async function runYoloSeg() {
  pipelineRunning.value = true; pipeType.value = 'yolo-seg'
  try {
    await $fetch("/api/dataset/pipeline/yolo/seg", { baseURL: apiBase, method: "POST" })
    await load()
  } catch {} finally { pipelineRunning.value = false }
}

async function runYolo() {
  pipelineRunning.value = true; pipeType.value = 'yolo'
  try {
    await Promise.allSettled([
      $fetch("/api/dataset/pipeline/yolo", { baseURL: apiBase, method: "POST" }),
      $fetch("/api/dataset/pipeline/yolo/seg", { baseURL: apiBase, method: "POST" }),
    ])
    await load()
  } catch {} finally { pipelineRunning.value = false }
}

async function load() {
  loading.value = true
  try { data.value = await $fetch("/api/dataset/grid", { baseURL: apiBase })
  } catch {} finally { loading.value = false }
}

onMounted(load)
</script>
