<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex items-center justify-between">
      <div>
        <h1 class="font-bold text-lg text-dark">Validation Inference</h1>
        <p class="text-sm text-dark/50">{{ tab === 0 ? `${valImages.length} val images` : `${results.length} annotated images` }}</p>
      </div>
    </div>

    <div class="border-b border-gray-200">
      <nav class="flex gap-6">
        <button v-for="(t, i) in tabs" :key="i"
          class="pb-3 text-sm font-medium transition-colors border-b-2 -mb-px"
          :class="tab === i ? 'text-tertiary border-tertiary' : 'text-dark/50 border-transparent hover:text-dark/70'"
          @click="tab = i">
          {{ t }}
        </button>
      </nav>
    </div>

    <div v-if="tab === 0">
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex items-center justify-between mb-4">
        <div>
          <p class="text-sm text-dark/70">Run YOLO inference on validation split</p>
        </div>
        <button :disabled="inferring"
          class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm transition-all flex items-center gap-2 disabled:opacity-50"
          @click="runVal">
          <svg v-if="inferring" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
          </svg>
          {{ inferring ? 'Running...' : 'Run Validation' }}
        </button>
      </div>

      <div v-if="inferResult" class="bg-green-50 border border-green-200 rounded-xl p-4 text-sm text-green-700 mb-4">
        Val done: {{ inferResult.images_processed || inferResult.processed }} images, {{ inferResult.total_predictions || inferResult.predictions }} predictions
      </div>

      <div v-if="loading" class="flex justify-center py-12">
        <svg class="animate-spin h-8 w-8 text-tertiary" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>

      <div v-else-if="!paginatedVal.length" class="bg-white rounded-xl p-10 text-center border border-gray-100">
        <p class="text-dark/50 text-sm">No validation dataset. Split raw data first.</p>
      </div>

      <template v-else>
        <div class="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 gap-3">
          <div v-for="img in paginatedVal" :key="img.filename"
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
        <Pagination :page="page" :total="valImages.length" :per="perPage" @update:page="page = $event" />
      </template>
    </div>

    <div v-if="tab === 1">
      <div v-if="loading" class="flex justify-center py-12">
        <svg class="animate-spin h-8 w-8 text-tertiary" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>

      <div v-else-if="!paginatedRes.length" class="bg-white rounded-xl p-10 text-center border border-gray-100">
        <p class="text-dark/50 text-sm">No val results. Run validation first.</p>
      </div>

      <template v-else>
        <div class="space-y-4">
          <div v-for="img in paginatedRes" :key="img.filename"
            class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
            <div class="flex flex-col lg:flex-row">
              <div class="lg:w-72 p-3">
                <div v-if="img.viz_url" class="relative">
                  <img :src="`${apiBase}${img.viz_url}`"
                    class="w-full aspect-square object-cover rounded-lg border-2 border-blue-200 cursor-pointer hover:opacity-90 transition-opacity"
                    @click="zoomUrl = `${apiBase}${img.viz_url}`" />
                </div>
                <div v-else class="w-full aspect-square rounded-lg border-2 border-dashed border-gray-200 bg-gray-50 flex items-center justify-center">
                  <p class="text-xs text-dark/30">No detection</p>
                </div>
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
                      <th class="p-2 text-right font-medium">X1</th>
                      <th class="p-2 text-right font-medium">Y1</th>
                      <th class="p-2 text-right font-medium">X2</th>
                      <th class="p-2 text-right font-medium">Y2</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(pred, j) in img.predictions" :key="j"
                      class="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                      <td class="p-2 text-dark/40 font-mono">{{ j + 1 }}</td>
                      <td class="p-2 font-medium text-dark">{{ pred.class_id }}</td>
                      <td class="p-2">
                        <span :class="pred.category === 'Organik' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'"
                          class="font-bold px-2 py-0.5 rounded text-xs">{{ pred.category }}</span>
                      </td>
                      <td class="p-2 text-right font-mono text-dark/60">{{ (pred.confidence * 100).toFixed(0) }}%</td>
                      <td class="p-2 text-right font-mono text-orange-600">{{ pred.bbox[0].toFixed(0) }}</td>
                      <td class="p-2 text-right font-mono text-orange-600">{{ pred.bbox[1].toFixed(0) }}</td>
                      <td class="p-2 text-right font-mono text-orange-600">{{ pred.bbox[2].toFixed(0) }}</td>
                      <td class="p-2 text-right font-mono text-orange-600">{{ pred.bbox[3].toFixed(0) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
        <Pagination :page="page2" :total="results.length" :per="perPage" @update:page="page2 = $event" />
      </template>
    </div>

    <ZoomModal :url="zoomUrl" @close="zoomUrl = null" />
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase

const data = ref<any>(null)
const loading = ref(true)
const tab = ref(0)
const page = ref(1)
const page2 = ref(1)
const perPage = 10
const zoomUrl = ref<string | null>(null)
const inferring = ref(false)
const inferResult = ref<any>(null)

const tabs = ["Validation Dataset", "Validation Result"]

const valImages = computed(() => data.value?.val_plain || [])
const results = computed(() => data.value?.val_real || [])

const paginatedVal = computed(() => {
  const start = (page.value - 1) * perPage
  return valImages.value.slice(start, start + perPage)
})

const paginatedRes = computed(() => {
  const start = (page2.value - 1) * perPage
  return results.value.slice(start, start + perPage)
})

async function runVal() {
  inferring.value = true; inferResult.value = null
  try {
    inferResult.value = await $fetch("/api/dataset/pipeline/yolo/val", { baseURL: apiBase, method: "POST" })
    await load()
    tab.value = 1
  } catch {} finally { inferring.value = false }
}

async function load() {
  loading.value = true
  try { data.value = await $fetch("/api/dataset/grid", { baseURL: apiBase })
  } catch {} finally { loading.value = false }
}

onMounted(load)
</script>
