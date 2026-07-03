<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex items-center justify-between">
      <div>
        <h1 class="font-bold text-lg text-dark">Test Data</h1>
        <p class="text-sm text-dark/50">{{ tab === 0 ? `${items.length} images` : tab === 1 ? `${testResults.length} results` : '' }}</p>
      </div>
      <div class="flex items-center gap-2">
        <button v-if="tab === 1" :disabled="inferring"
          class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm transition-all flex items-center gap-2 disabled:opacity-50"
          @click="runTest">
          <svg v-if="inferring" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
          </svg>
          {{ inferring ? 'Running...' : 'Run Test' }}
        </button>
        <button v-if="tab === 2" :disabled="evaluating"
          class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm transition-all flex items-center gap-2 disabled:opacity-50"
          @click="runEval">
          <svg v-if="evaluating" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z" />
          </svg>
          {{ evaluating ? 'Evaluating...' : evalMetrics ? 'Refresh Evaluation' : 'Run Evaluation' }}
        </button>
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

    <div v-if="inferResult" class="bg-green-50 border border-green-200 rounded-xl p-4 text-sm text-green-700">
      Test done: {{ inferResult.images_processed || inferResult.processed }} images, {{ inferResult.total_predictions || inferResult.predictions }} predictions
    </div>

    <div v-if="evalError" class="bg-red-50 border border-red-200 rounded-xl p-4">
      <p class="text-red-600 text-sm">{{ evalError }}</p>
    </div>

    <!-- Tab 0: Test Images -->
    <template v-if="tab === 0">
      <div v-if="loading" class="flex justify-center py-12">
        <svg class="animate-spin h-8 w-8 text-tertiary" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>
      <div v-else-if="!paginatedItems.length" class="text-center py-12">
        <p class="text-dark/50 text-sm">No test images.</p>
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
            </div>
          </div>
        </div>
        <Pagination :page="page" :total="items.length" :per="perPage" @update:page="page = $event" />
      </template>
    </template>

    <!-- Tab 1: Test Inference Results -->
    <template v-if="tab === 1">
      <div v-if="loading" class="flex justify-center py-12">
        <svg class="animate-spin h-8 w-8 text-tertiary" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>
      <div v-else-if="!paginatedResults.length" class="text-center py-12">
        <p class="text-dark/50 text-sm">No test results. Run test first.</p>
      </div>
      <template v-else>
        <div class="space-y-4">
          <div v-for="img in paginatedResults" :key="img.filename"
            class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
            <div class="flex flex-col lg:flex-row">
              <div class="lg:w-72 p-3">
                <img v-if="img.viz_url" :src="`${apiBase}${img.viz_url}`"
                  class="w-full aspect-square object-cover rounded-lg border-2 border-blue-200 cursor-pointer hover:opacity-90 transition-opacity"
                  @click="zoomUrl = `${apiBase}${img.viz_url}`" />
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
        <Pagination :page="resPage" :total="testResults.length" :per="perPage" @update:page="resPage = $event" />
      </template>
    </template>

    <!-- Tab 2: Evaluation Metrics -->
    <template v-if="tab === 2">
      <div v-if="evalMetrics" class="space-y-6">
        <div class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
          <table class="w-full text-sm">
            <thead class="bg-gray-50 text-dark/60 text-xs uppercase tracking-wide">
              <tr>
                <th class="p-3 text-left font-medium">Split</th>
                <th class="p-3 text-right font-medium">mAP@0.5</th>
                <th class="p-3 text-right font-medium">mAP@0.5:0.95</th>
                <th class="p-3 text-right font-medium">Precision</th>
                <th class="p-3 text-right font-medium">Recall</th>
                <th class="p-3 text-right font-medium">F1-Score</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(m, split) in evalMetrics" :key="split" class="border-b border-gray-100 hover:bg-gray-50/50 transition-colors">
                <td class="p-3 font-medium text-dark capitalize">{{ split }}</td>
                <td class="p-3 text-right font-mono" :class="scoreColor(m.mAP50)">{{ m.mAP50.toFixed(1) }}%</td>
                <td class="p-3 text-right font-mono" :class="scoreColor(m.mAP50_95)">{{ m.mAP50_95.toFixed(1) }}%</td>
                <td class="p-3 text-right font-mono" :class="scoreColor(m.precision)">{{ m.precision.toFixed(1) }}%</td>
                <td class="p-3 text-right font-mono" :class="scoreColor(m.recall)">{{ m.recall.toFixed(1) }}%</td>
                <td class="p-3 text-right font-mono" :class="scoreColor(f1Score(m))">{{ f1Score(m).toFixed(1) }}%</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div v-for="(m, split) in evalMetrics" :key="'pc-'+split" class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
          <div class="bg-gray-50 border-b border-gray-100 px-4 py-2">
            <h4 class="text-xs font-bold text-dark/60 uppercase tracking-wide">{{ split }} · Per-Class mAP@0.5</h4>
          </div>
          <table class="w-full text-sm" v-if="m.per_class?.length">
            <thead class="text-dark/50 text-xs uppercase tracking-wide">
              <tr>
                <th class="p-2 pl-4 text-left font-medium">Class</th>
                <th class="p-2 text-right font-medium">mAP@0.5</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="pc in m.per_class" :key="pc.class_id" class="border-b border-gray-50 hover:bg-gray-50/50 transition-colors">
                <td class="p-2 pl-4 font-medium text-dark">{{ pc.name }}</td>
                <td class="p-2 text-right font-mono" :class="scoreColor(pc.mAP50)">{{ pc.mAP50.toFixed(1) }}%</td>
              </tr>
            </tbody>
          </table>
          <p v-else class="p-4 text-center text-dark/40 text-xs">No per-class data</p>
        </div>
      </div>

      <div v-else-if="!evaluating" class="bg-white rounded-xl p-10 text-center border border-gray-100">
        <p class="text-dark/50 text-sm">Click "Run Evaluation" to evaluate model performance.</p>
      </div>

      <div v-if="loading && evaluating" class="flex justify-center py-12">
        <svg class="animate-spin h-8 w-8 text-tertiary" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>
    </template>

    <ZoomModal :url="zoomUrl" @close="zoomUrl = null" />
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase
const { show: showError } = useToast()

const data = ref<any>(null)
const loading = ref(true)
const tab = ref(0)
const page = ref(1)
const resPage = ref(1)
const perPage = 20
const zoomUrl = ref<string | null>(null)
const inferring = ref(false)
const inferResult = ref<any>(null)
const evaluating = ref(false)
const evalMetrics = ref<any>(null)
const evalError = ref<string | null>(null)

const STORAGE_KEY = 'sb_eval_metrics'
const tabs = ["Test Images", "Test Inference Results", "Evaluation Metrics"]

const items = computed(() => data.value?.test || [])
const testResults = computed(() => data.value?.test_real || [])
const paginatedItems = computed(() => {
  const start = (page.value - 1) * perPage
  return items.value.slice(start, start + perPage)
})
const paginatedResults = computed(() => {
  const start = (resPage.value - 1) * perPage
  return testResults.value.slice(start, start + perPage)
})

function f1Score(m: any): number {
  const p = m.precision, r = m.recall
  return (p + r) > 0 ? 2 * p * r / (p + r) : 0
}

function scoreColor(val: number): string {
  if (val >= 80) return 'text-green-600'
  if (val >= 60) return 'text-yellow-600'
  return 'text-red-600'
}

async function runTest() {
  inferring.value = true; inferResult.value = null
  try {
    inferResult.value = await $fetch("/api/dataset/pipeline/yolo/test", { baseURL: apiBase, method: "POST" })
    await load()
  } catch (e: any) { showError(e?.data?.detail || e?.message || 'Test pipeline failed') } finally { inferring.value = false }
}

async function runEval() {
  evaluating.value = true; evalError.value = null; evalMetrics.value = null
  try {
    const res = await $fetch("/api/dataset/evaluate", { baseURL: apiBase, params: { split: "all" } })
    evalMetrics.value = res.metrics
    localStorage.setItem(STORAGE_KEY, JSON.stringify(res.metrics))
  } catch (e: any) {
    evalError.value = e.data?.detail || e.message || "Evaluation failed"
  } finally { evaluating.value = false }
}

async function load() {
  loading.value = true
  try { data.value = await $fetch("/api/dataset/grid", { baseURL: apiBase })
  } catch (e: any) { showError(e?.data?.detail || e?.message || 'Failed to load data') } finally { loading.value = false }
}

onMounted(async () => {
  try {
    const cached = localStorage.getItem(STORAGE_KEY)
    if (cached) evalMetrics.value = JSON.parse(cached)
  } catch {}
  await load()
})
</script>
