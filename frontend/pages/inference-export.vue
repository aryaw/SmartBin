<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
      <h1 class="font-bold text-lg text-dark">Inference, Export & Final Verification</h1>
      <p class="text-sm text-dark/50">Upload image, batch inference, export model, verify results</p>
    </div>

    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
      <div class="flex items-center gap-3 mb-4">
        <span class="w-7 h-7 rounded-full bg-tertiary text-white flex items-center justify-center text-xs font-bold">4</span>
        <h2 class="font-bold text-dark">Inference, Export & Verification</h2>
        <div class="ml-auto flex gap-2 flex-wrap">
          <button class="px-4 py-1.5 text-xs font-bold rounded-lg bg-tertiary text-white hover:bg-blue-700" @click="runInference">Upload & Infer</button>
          <button :disabled="batchRunning" class="px-4 py-1.5 text-xs font-bold rounded-lg bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runBatch">
            {{ batchRunning ? 'Running...' : 'Batch Test' }}
          </button>
          <button :disabled="exportRunning" class="px-4 py-1.5 text-xs font-bold rounded-lg bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runExport">
            {{ exportRunning ? 'Exporting...' : 'Export Model' }}
          </button>
          <button :disabled="verifyRunning" class="px-4 py-1.5 text-xs font-bold rounded-lg bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runVerify">
            {{ verifyRunning ? 'Verifying...' : 'Final Verify' }}
          </button>
        </div>
      </div>
      <div v-if="inferenceResult || batchResult || exportResult || verifyResult || advice.length" class="bg-gray-50 rounded-lg p-3 text-xs font-mono max-h-64 overflow-auto">
        <pre v-if="inferenceResult" class="text-green-700">{{ inferenceResult }}</pre>
        <pre v-if="batchResult" class="text-green-700">{{ batchResult }}</pre>
        <pre v-if="exportResult" class="text-green-700">{{ exportResult }}</pre>
        <pre v-if="verifyResult" class="text-green-700">{{ verifyResult }}</pre>
        <div v-for="(a, i) in advice" :key="i" class="text-xs px-3 py-1 rounded-lg mt-1"
          :class="a.category === 'Organic' ? 'bg-green-50 text-green-700' : a.category === 'Recyclable' ? 'bg-blue-50 text-blue-700' : a.category === 'Hazardous' ? 'bg-red-50 text-red-700' : 'bg-gray-50 text-gray-700'">
          <strong>{{ a.class_name }}</strong> ({{ a.category }}): {{ a.advice }}
        </div>
      </div>
      <div v-if="inferenceImage" class="mt-3">
        <img :src="inferenceImage" class="max-w-lg rounded border" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase
const { show: showError } = useToast()
const inferenceResult = ref(null); const inferenceImage = ref(null); const advice = ref([])
const batchResult = ref(null); const batchRunning = ref(false)
const exportResult = ref(null); const exportRunning = ref(false)
const verifyResult = ref(null); const verifyRunning = ref(false)

async function runInference() {
  const input = document.createElement("input"); input.type = "file"; input.accept = "image/*"
  input.onchange = async () => {
    if (!input.files?.[0]) return
    inferenceResult.value = "Running..."; advice.value = []; inferenceImage.value = null
    const form = new FormData(); form.append("file", input.files[0])
    try {
      const cats = await $fetch("/api/kaggle/categories", { baseURL: apiBase }) as any
      const res = await $fetch("/api/kaggle/inference", { baseURL: apiBase, method: "POST", body: form }) as any
      const blob = res as Blob; inferenceImage.value = URL.createObjectURL(blob)
      inferenceResult.value = "Inference complete"
      const h = (res as any).headers?.["x-detections"]
      if (h) {
        const dets = JSON.parse(h)
        advice.value = dets.map((d: any) => {
          const mc = cats.sub_to_main?.[d.class] || "Unknown"
          return { class_name: d.class, category: mc, advice: cats.recycling_advice?.[mc] || "" }
        })
      }
    } catch (e: any) { showError(e?.data?.detail || e?.message) }
  }
  input.click()
}
async function runBatch() {
  batchRunning.value = true; batchResult.value = null
  try {
    const res = await $fetch("/api/kaggle/inference/batch", { baseURL: apiBase, method: "POST" }) as any
    let out = `Processed ${res.images_processed} images, ${res.total_detections} detections\n\nPer-Class:\n`
    for (const p of res.per_class) { out += `  ${p.class_name.padEnd(25)} ${String(p.count).padStart(3)}  ${(p.max_confidence * 100).toFixed(0)}%  ${p.advice}\n` }
    batchResult.value = out
  } catch (e: any) { showError(e?.data?.detail || e?.message) } finally { batchRunning.value = false }
}
async function runExport() {
  exportRunning.value = true; exportResult.value = null
  try {
    const res = await $fetch("/api/kaggle/export?format=onnx", { baseURL: apiBase, method: "POST" }) as any
    let out = ""
    for (const [fmt, info] of Object.entries(res.formats)) { const i = info as any; out += `  ${fmt}: ${i.size_mb ? i.size_mb + ' MB' : i.error || 'unknown'}\n` }
    exportResult.value = out
  } catch (e: any) { showError(e?.data?.detail || e?.message) } finally { exportRunning.value = false }
}
async function runVerify() {
  verifyRunning.value = true; verifyResult.value = null
  try {
    const res = await $fetch("/api/kaggle/verify", { baseURL: apiBase }) as any
    let out = `Final Verification\n  Box mAP@50: ${(res.summary.box_mAP50 * 100).toFixed(1)}%\n  Mask mAP@50: ${(res.summary.mask_mAP50 * 100).toFixed(1)}%\n\nPer-Class Mask AP@50:\n`
    for (const p of res.per_class) { const bar = '█'.repeat(Math.round(p.mask_ap50 * 40)); out += `  [${p.class_id}] ${p.name.padEnd(25)} ${(p.mask_ap50 * 100).toFixed(1)}% ${bar}\n` }
    verifyResult.value = out
  } catch (e: any) { showError(e?.data?.detail || e?.message) } finally { verifyRunning.value = false }
}
</script>
