<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
      <h1 class="font-bold text-lg text-dark">Deployment</h1>
      <p class="text-sm text-dark/50">Run inference, batch test, export model, and verify</p>
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
            <td class="px-4 py-3 font-medium">Inference</td>
            <td class="px-4 py-3 text-dark/60">Upload an image and get segmentation masks, labels, and advice</td>
            <td class="px-4 py-3 text-center">
              <span v-if="inferStatus === 'done'" class="text-green-600 bg-green-50 px-2 py-0.5 rounded">Done</span>
              <span v-else class="text-gray-400 bg-gray-50 px-2 py-0.5 rounded">Not Started</span>
            </td>
            <td class="px-4 py-3 text-dark/60">{{ inferSummary }}</td>
            <td class="px-4 py-3 text-center">
              <button class="px-3 py-1 text-xs font-bold rounded bg-tertiary text-white hover:bg-blue-700" @click="runInference">Analyze Image</button>
            </td>
          </tr>
          <tr>
            <td class="px-4 py-3 font-medium">Batch Inference</td>
            <td class="px-4 py-3 text-dark/60">Run inference on test set and generate advice report</td>
            <td class="px-4 py-3 text-center">
              <span v-if="batchStatus === 'done'" class="text-green-600 bg-green-50 px-2 py-0.5 rounded">Done</span>
              <span v-else-if="batchStatus === 'running'" class="text-blue-600 bg-blue-50 px-2 py-0.5 rounded">Running</span>
              <span v-else class="text-gray-400 bg-gray-50 px-2 py-0.5 rounded">Not Started</span>
            </td>
            <td class="px-4 py-3 text-dark/60">{{ batchSummary }}</td>
            <td class="px-4 py-3 text-center">
              <button :disabled="batchStatus === 'running'" class="px-3 py-1 text-xs font-bold rounded bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runBatch">Analyze Test Dataset</button>
            </td>
          </tr>
          <tr>
            <td class="px-4 py-3 font-medium">Export Model</td>
            <td class="px-4 py-3 text-dark/60">Export trained model to ONNX or TorchScript format</td>
            <td class="px-4 py-3 text-center">
              <span v-if="exportStatus === 'done'" class="text-green-600 bg-green-50 px-2 py-0.5 rounded">Done</span>
              <span v-else-if="exportStatus === 'running'" class="text-blue-600 bg-blue-50 px-2 py-0.5 rounded">Running</span>
              <span v-else class="text-gray-400 bg-gray-50 px-2 py-0.5 rounded">Not Started</span>
            </td>
            <td class="px-4 py-3 text-dark/60">{{ exportSummary }}</td>
            <td class="px-4 py-3 text-center">
              <button :disabled="exportStatus === 'running'" class="px-3 py-1 text-xs font-bold rounded bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runExport">Export Model</button>
            </td>
          </tr>
          <tr>
            <td class="px-4 py-3 font-medium">Final Verification</td>
            <td class="px-4 py-3 text-dark/60">Verify artifacts and generate per-class mask mAP report</td>
            <td class="px-4 py-3 text-center">
              <span v-if="verifyStatus === 'done'" class="text-green-600 bg-green-50 px-2 py-0.5 rounded">Done</span>
              <span v-else-if="verifyStatus === 'running'" class="text-blue-600 bg-blue-50 px-2 py-0.5 rounded">Running</span>
              <span v-else class="text-gray-400 bg-gray-50 px-2 py-0.5 rounded">Not Started</span>
            </td>
            <td class="px-4 py-3 text-dark/60">{{ verifySummary }}</td>
            <td class="px-4 py-3 text-center">
              <div class="flex items-center justify-center gap-1">
                <button :disabled="verifyStatus === 'running'" class="px-3 py-1 text-xs font-bold rounded bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runVerify">Run Final Verification</button>
                <button v-if="verifyReport" class="px-3 py-1 text-xs font-bold rounded bg-gray-100 text-dark/50 hover:bg-gray-200" @click="showVerify = !showVerify">View Report</button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="inferenceImage" class="max-w-lg mx-auto">
      <img :src="inferenceImage" class="w-full rounded border" />
    </div>
    <div v-if="advice.length" class="space-y-1">
      <div v-for="(a, i) in advice" :key="i" class="text-xs px-3 py-1.5 rounded-lg"
        :class="a.category === 'Organic' ? 'bg-green-50 text-green-700' : a.category === 'Recyclable' ? 'bg-blue-50 text-blue-700' : a.category === 'Hazardous' ? 'bg-red-50 text-red-700' : 'bg-gray-50 text-gray-700'">
        <strong>{{ a.class_name }}</strong> ({{ a.category }}): {{ a.advice }}
      </div>
    </div>
    <div v-if="batchReport" class="bg-gray-50 rounded-xl p-4 text-xs font-mono max-h-64 overflow-auto">
      <pre class="text-green-700">{{ batchReport }}</pre>
    </div>
    <div v-if="verifyReport && showVerify" class="bg-gray-50 rounded-xl p-4 text-xs font-mono max-h-96 overflow-auto">
      <pre class="text-green-700">{{ verifyReport }}</pre>
    </div>
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase
const { show: showError } = useToast()
const inferStatus = ref('not_started'); const inferSummary = ref(''); const inferenceImage = ref(null); const advice = ref([])
const batchStatus = ref('not_started'); const batchSummary = ref(''); const batchReport = ref('')
const exportStatus = ref('not_started'); const exportSummary = ref('')
const verifyStatus = ref('not_started'); const verifySummary = ref(''); const verifyReport = ref(''); const showVerify = ref(false)

async function runInference() {
  const input = document.createElement("input"); input.type = "file"; input.accept = "image/*"
  input.onchange = async () => {
    if (!input.files?.[0]) return
    const form = new FormData(); form.append("file", input.files[0])
    try {
      const cats = await $fetch("/api/kaggle/categories", { baseURL: apiBase }) as any
      const res = await $fetch("/api/kaggle/inference", { baseURL: apiBase, method: "POST", body: form }) as any
      const blob = res as Blob; inferenceImage.value = URL.createObjectURL(blob)
      const h = (res as any).headers?.["x-detections"]
      if (h) {
        const dets = JSON.parse(h)
        advice.value = dets.map((d: any) => {
          const mc = cats.sub_to_main?.[d.class] || "Unknown"
          return { class_name: d.class, category: mc, advice: cats.recycling_advice?.[mc] || "" }
        })
        inferSummary.value = `${dets.length} objects detected`
      }
      inferStatus.value = 'done'
    } catch (e: any) { showError(e?.data?.detail || e?.message) }
  }
  input.click()
}
async function runBatch() {
  batchStatus.value = 'running'; batchReport.value = ''
  try {
    const res = await $fetch("/api/kaggle/inference/batch", { baseURL: apiBase, method: "POST" }) as any
    let out = `Processed ${res.images_processed} images, ${res.total_detections} detections\n\nPer-Class:\n`
    for (const p of res.per_class) { out += `  ${p.class_name.padEnd(25)} ${String(p.count).padStart(3)}  ${(p.max_confidence * 100).toFixed(0)}%  ${p.advice}\n` }
    batchReport.value = out; batchSummary.value = `${res.images_processed} images, ${res.total_detections} detections`
    batchStatus.value = 'done'
  } catch (e: any) { showError(e?.data?.detail || e?.message); batchStatus.value = 'not_started' }
}
async function runExport() {
  exportStatus.value = 'running'
  try {
    const res = await $fetch("/api/kaggle/export?format=onnx", { baseURL: apiBase, method: "POST" }) as any
    let out = ""
    for (const [fmt, info] of Object.entries(res.formats)) { const i = info as any; out += `${fmt}: ${i.size_mb ? i.size_mb + ' MB' : i.error || 'unknown'}\n` }
    exportSummary.value = out.trim()
    exportStatus.value = 'done'
  } catch (e: any) { showError(e?.data?.detail || e?.message); exportStatus.value = 'not_started' }
}
async function runVerify() {
  verifyStatus.value = 'running'; verifyReport.value = ''
  try {
    const res = await $fetch("/api/kaggle/verify", { baseURL: apiBase }) as any
    let out = `Box mAP@50: ${(res.summary.box_mAP50 * 100).toFixed(1)}%\nMask mAP@50: ${(res.summary.mask_mAP50 * 100).toFixed(1)}%\n\nPer-Class Mask AP@50:\n`
    for (const p of res.per_class) { const bar = '█'.repeat(Math.round(p.mask_ap50 * 40)); out += `  [${p.class_id}] ${p.name.padEnd(25)} ${(p.mask_ap50 * 100).toFixed(1)}% ${bar}\n` }
    verifyReport.value = out; verifySummary.value = `Box mAP50: ${(res.summary.box_mAP50 * 100).toFixed(1)}%, Mask mAP50: ${(res.summary.mask_mAP50 * 100).toFixed(1)}%`
    verifyStatus.value = 'done'
  } catch (e: any) { showError(e?.data?.detail || e?.message); verifyStatus.value = 'not_started' }
}
</script>
