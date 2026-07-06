<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
      <h1 class="font-bold text-lg text-dark">Training</h1>
      <p class="text-sm text-dark/50">Train the waste segmentation model and evaluate results</p>
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
            <td class="px-4 py-3 font-medium">Train YOLOv26m</td>
            <td class="px-4 py-3 text-dark/60">Train the waste segmentation model on the processed dataset</td>
            <td class="px-4 py-3 text-center">
              <span v-if="trainStatus === 'done'" class="text-green-600 bg-green-50 px-2 py-0.5 rounded">Done</span>
              <span v-else-if="trainStatus === 'running'" class="text-blue-600 bg-blue-50 px-2 py-0.5 rounded">Running</span>
              <span v-else class="text-gray-400 bg-gray-50 px-2 py-0.5 rounded">Not Started</span>
            </td>
            <td class="px-4 py-3 text-dark/60">{{ trainSummary }}</td>
            <td class="px-4 py-3 text-center">
              <button :disabled="trainStatus === 'running'" class="px-3 py-1 text-xs font-bold rounded bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runTrain">Start Training</button>
            </td>
          </tr>
          <tr>
            <td class="px-4 py-3 font-medium">Training Results & Curves</td>
            <td class="px-4 py-3 text-dark/60">View training metrics, loss curves, and evaluation plots</td>
            <td class="px-4 py-3 text-center">
              <span v-if="resultsStatus === 'done'" class="text-green-600 bg-green-50 px-2 py-0.5 rounded">Done</span>
              <span v-else class="text-gray-400 bg-gray-50 px-2 py-0.5 rounded">Not Started</span>
            </td>
            <td class="px-4 py-3 text-dark/60">{{ resultsSummary }}</td>
            <td class="px-4 py-3 text-center">
              <button class="px-3 py-1 text-xs font-bold rounded bg-gray-100 text-dark/50 hover:bg-gray-200" @click="runResults">View Results</button>
            </td>
          </tr>
          <tr>
            <td class="px-4 py-3 font-medium">Validation & Test Evaluation</td>
            <td class="px-4 py-3 text-dark/60">Evaluate box and mask metrics on validation and test sets</td>
            <td class="px-4 py-3 text-center">
              <span v-if="evalStatus === 'done'" class="text-green-600 bg-green-50 px-2 py-0.5 rounded">Done</span>
              <span v-else-if="evalStatus === 'running'" class="text-blue-600 bg-blue-50 px-2 py-0.5 rounded">Running</span>
              <span v-else class="text-gray-400 bg-gray-50 px-2 py-0.5 rounded">Not Started</span>
            </td>
            <td class="px-4 py-3 text-dark/60">{{ evalSummary }}</td>
            <td class="px-4 py-3 text-center">
              <button :disabled="evalStatus === 'running'" class="px-3 py-1 text-xs font-bold rounded bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runEval">Run Evaluation</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="resultsImages.length" class="grid grid-cols-3 gap-2">
      <img v-for="url in resultsImages" :key="url" :src="`${apiBase}${url}`" class="w-full rounded border" />
    </div>
    <div v-if="evalReport" class="bg-gray-50 rounded-xl p-4 text-xs font-mono max-h-64 overflow-auto">
      <pre class="text-green-700">{{ evalReport }}</pre>
    </div>
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase
const { show: showError } = useToast()
const trainStatus = ref('not_started'); const trainSummary = ref('')
const resultsStatus = ref('not_started'); const resultsSummary = ref(''); const resultsImages = ref([])
const evalStatus = ref('not_started'); const evalSummary = ref(''); const evalReport = ref('')

onMounted(async () => {
  try {
    const s = await $fetch("/api/kaggle/train/status", { baseURL: apiBase }) as any
    if (s.running) trainStatus.value = 'running'
  } catch (_) {}
})

async function runTrain() {
  trainStatus.value = 'running'
  try {
    const res = await $fetch("/api/kaggle/train?batch=16&lr0=0.001", { baseURL: apiBase, method: "POST" }) as any
    trainSummary.value = `mAP50: ${(res.map50 * 100).toFixed(1)}%, model: ${res.saved_to}`
    trainStatus.value = 'done'
  } catch (e: any) { showError(e?.data?.detail || e?.message); trainStatus.value = 'not_started' }
}
async function runResults() {
  try {
    const res = await $fetch("/api/kaggle/results", { baseURL: apiBase }) as any
    if (res.images?.length) { resultsImages.value = res.images.map((i: any) => i.url); resultsSummary.value = `${res.images.length} plots found`; resultsStatus.value = 'done' }
    else { resultsSummary.value = 'No results available' }
  } catch (e: any) { showError(e?.data?.detail || e?.message) }
}
async function runEval() {
  evalStatus.value = 'running'; evalReport.value = ''
  try {
    const res = await $fetch("/api/kaggle/evaluate", { baseURL: apiBase }) as any
    evalReport.value =
      `Box mAP@50: ${(res.box_mAP50 * 100).toFixed(1)}% | mAP@50-95: ${(res.box_mAP50_95 * 100).toFixed(1)}%\n` +
      `Box Precision: ${(res.box_precision * 100).toFixed(1)}% | Recall: ${(res.box_recall * 100).toFixed(1)}%\n\n` +
      `Mask mAP@50: ${(res.mask_mAP50 * 100).toFixed(1)}% | mAP@50-95: ${(res.mask_mAP50_95 * 100).toFixed(1)}%\n` +
      `Mask Precision: ${(res.mask_precision * 100).toFixed(1)}% | Recall: ${(res.mask_recall * 100).toFixed(1)}%`
    evalSummary.value = `Box mAP50: ${(res.box_mAP50 * 100).toFixed(1)}%, Mask mAP50: ${(res.mask_mAP50 * 100).toFixed(1)}%`
    evalStatus.value = 'done'
  } catch (e: any) { showError(e?.data?.detail || e?.message); evalStatus.value = 'not_started' }
}
</script>
