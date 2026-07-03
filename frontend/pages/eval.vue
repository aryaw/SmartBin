<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex items-center justify-between">
      <h1 class="font-bold text-lg text-dark">Evaluation Metrics</h1>
      <div class="flex items-center gap-3">
        <p class="text-sm text-dark/50">{{ valImages.length }} val images</p>
        <button :disabled="evaluating"
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

    <div v-if="evalError" class="bg-red-50 border border-red-200 rounded-xl p-4">
      <p class="text-red-600 text-sm">{{ evalError }}</p>
    </div>

    <!-- Metrics Table -->
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

    <div v-else-if="!evaluating && loading" class="bg-white rounded-xl p-10 text-center border border-gray-100">
      <p class="text-dark/50 text-sm">Click "Run Evaluation" to evaluate model performance.</p>
    </div>

    <!-- Validation Images -->
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
      <h2 class="font-bold text-dark/60 uppercase tracking-wide text-sm mb-4">Validation Images with Predictions</h2>

      <div v-if="loading" class="flex justify-center py-8">
        <svg class="animate-spin h-8 w-8 text-tertiary" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>

      <div v-else-if="!paginatedItems.length" class="text-center py-8">
        <p class="text-dark/50 text-sm">No validation images available.</p>
      </div>

      <template v-else>
        <div class="space-y-4">
          <div v-for="img in paginatedItems" :key="img.filename"
            class="border border-gray-100 rounded-xl overflow-hidden">
            <div class="flex flex-col lg:flex-row">
              <div class="lg:w-72 p-3">
                <img :src="`${apiBase}${img.viz_url}`"
                  class="w-full aspect-square object-cover rounded-lg border-2 border-blue-200 cursor-pointer hover:opacity-90 transition-opacity"
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
        <Pagination :page="page" :total="valImages.length" :per="perPage" @update:page="page = $event" />
      </template>
    </div>

    <ZoomModal :url="zoomUrl" @close="zoomUrl = null" />
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase

const evaluating = ref(false)
const evalMetrics = ref<any>(null)
const evalError = ref<string | null>(null)
const data = ref<any>(null)
const loading = ref(true)
const page = ref(1)
const perPage = 10
const zoomUrl = ref<string | null>(null)

const STORAGE_KEY = 'sb_eval_metrics'

const valImages = computed(() => (data.value?.val_real || []).filter((img: any) => img.viz_url && img.predictions.length > 0))

const paginatedItems = computed(() => {
  const start = (page.value - 1) * perPage
  return valImages.value.slice(start, start + perPage)
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

onMounted(async () => {
  try {
    const cached = localStorage.getItem(STORAGE_KEY)
    if (cached) evalMetrics.value = JSON.parse(cached)
  } catch {}
  try {
    data.value = await $fetch("/api/dataset/grid", { baseURL: apiBase })
  } catch {} finally { loading.value = false }
})
</script>
