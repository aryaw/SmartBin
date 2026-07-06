<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
      <h1 class="font-bold text-lg text-dark">Train, Results & Evaluation</h1>
      <p class="text-sm text-dark/50">Train YOLOv26m-seg, view curves, and evaluate metrics</p>
    </div>

    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
      <div class="flex items-center gap-3 mb-4">
        <span class="w-7 h-7 rounded-full bg-tertiary text-white flex items-center justify-center text-xs font-bold">3</span>
        <h2 class="font-bold text-dark">Train, Results & Evaluation</h2>
        <div class="ml-auto flex gap-2">
          <button :disabled="trainRunning" class="px-4 py-1.5 text-xs font-bold rounded-lg bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runTrain">
            {{ trainRunning ? 'Training...' : 'Train Model' }}
          </button>
          <button :disabled="resultsRunning" class="px-4 py-1.5 text-xs font-bold rounded-lg bg-gray-100 text-dark/50 hover:bg-gray-200 disabled:opacity-40" @click="runResults">
            {{ resultsRunning ? 'Loading...' : 'Show Curves' }}
          </button>
          <button :disabled="evalRunning" class="px-4 py-1.5 text-xs font-bold rounded-lg bg-gray-100 text-dark/50 hover:bg-gray-200 disabled:opacity-40" @click="runEval">
            {{ evalRunning ? 'Evaluating...' : 'Evaluate' }}
          </button>
        </div>
      </div>
      <div class="mb-3 flex gap-3 flex-wrap">
        <label v-for="(val, key) in trainConfig" :key="key" class="text-xs flex items-center gap-1">
          <span class="text-dark/60">{{ key }}:</span>
          <input v-model="trainConfig[key]" class="w-20 px-2 py-1 border rounded text-xs" />
        </label>
      </div>
      <div v-if="trainResult || resultsResult || evalResult" class="bg-gray-50 rounded-lg p-3 text-xs font-mono max-h-48 overflow-auto">
        <pre v-if="trainResult" class="text-green-700">{{ trainResult }}</pre>
        <pre v-if="resultsResult" class="text-green-700">{{ resultsResult }}</pre>
        <pre v-if="evalResult" class="text-green-700">{{ evalResult }}</pre>
      </div>
      <div v-if="resultsImages.length" class="mt-3 grid grid-cols-6 gap-2">
        <img v-for="url in resultsImages" :key="url" :src="`${apiBase}${url}`" class="w-full aspect-square object-cover rounded border" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase
const { show: showError } = useToast()
const trainConfig = ref({ batch: 16, lr0: 0.001 })
const trainResult = ref(null); const trainRunning = ref(false)
const resultsResult = ref(null); const resultsRunning = ref(false); const resultsImages = ref([])
const evalResult = ref(null); const evalRunning = ref(false)

async function runTrain() {
  trainRunning.value = true; trainResult.value = null
  try {
    const cfg = trainConfig.value
    const res = await $fetch(`/api/kaggle/train?batch=${cfg.batch}&lr0=${cfg.lr0}`, { baseURL: apiBase, method: "POST" }) as any
    trainResult.value = `Best model: ${res.best_model_path}\nmAP50: ${(res.map50 * 100).toFixed(1)}%\nSaved to: ${res.saved_to}`
  } catch (e: any) { showError(e?.data?.detail || e?.message) } finally { trainRunning.value = false }
}
async function runResults() {
  resultsRunning.value = true; resultsResult.value = null; resultsImages.value = []
  try {
    const res = await $fetch("/api/kaggle/results", { baseURL: apiBase }) as any
    if (res.images?.length) { resultsImages.value = res.images.map((i: any) => i.url); resultsResult.value = `${res.images.length} result images` }
    else { resultsResult.value = "No results" }
  } catch (e: any) { showError(e?.data?.detail || e?.message) } finally { resultsRunning.value = false }
}
async function runEval() {
  evalRunning.value = true; evalResult.value = null
  try {
    const res = await $fetch("/api/kaggle/evaluate", { baseURL: apiBase }) as any
    evalResult.value =
      `Box mAP@50: ${(res.box_mAP50 * 100).toFixed(1)}% | mAP@50-95: ${(res.box_mAP50_95 * 100).toFixed(1)}%\n` +
      `Box Precision: ${(res.box_precision * 100).toFixed(1)}% | Recall: ${(res.box_recall * 100).toFixed(1)}%\n\n` +
      `Mask mAP@50: ${(res.mask_mAP50 * 100).toFixed(1)}% | mAP@50-95: ${(res.mask_mAP50_95 * 100).toFixed(1)}%\n` +
      `Mask Precision: ${(res.mask_precision * 100).toFixed(1)}% | Recall: ${(res.mask_recall * 100).toFixed(1)}%`
  } catch (e: any) { showError(e?.data?.detail || e?.message) } finally { evalRunning.value = false }
}
</script>
