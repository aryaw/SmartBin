<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
      <h1 class="font-bold text-lg text-dark">Kaggle Waste Pipeline CMS</h1>
      <p class="text-sm text-dark/50">Step-by-step pipeline for waste dataset training</p>
    </div>

    <div class="grid gap-4">
      <div v-for="(step, i) in steps" :key="step.key"
        class="bg-white rounded-xl shadow-sm border border-gray-100 p-5"
        :class="{ 'opacity-50': step.locked }">
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold"
              :class="step.done ? 'bg-green-500 text-white' : step.running ? 'bg-blue-500 text-white animate-pulse' : 'bg-gray-100 text-dark/50'">
              <svg v-if="step.done" class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
              </svg>
              <span v-else>{{ i + 1 }}</span>
            </div>
            <div>
              <p class="font-medium text-sm text-dark">{{ step.label }}</p>
              <p v-if="step.status" class="text-xs text-dark/40">{{ step.status }}</p>
            </div>
          </div>
          <div class="flex items-center gap-2">
            <button :disabled="step.locked || step.running"
              class="px-4 py-1.5 text-xs font-bold rounded-lg transition-all disabled:opacity-40"
              :class="step.done ? 'bg-gray-100 text-dark/50 hover:bg-gray-200' : 'bg-tertiary text-white hover:bg-blue-700'"
              @click="runStep(step.key)">
              <span v-if="step.running">
                <svg class="animate-spin h-3 w-3 inline mr-1" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Running...
              </span>
              <span v-else-if="step.done">Re-run</span>
              <span v-else>Run</span>
            </button>
          </div>
        </div>

        <div v-if="step.result || step.error" class="bg-gray-50 rounded-lg p-3 text-xs font-mono max-h-40 overflow-auto">
          <pre v-if="step.result" class="text-green-700">{{ step.result }}</pre>
          <pre v-if="step.error" class="text-red-600">{{ step.error }}</pre>
        </div>

        <div v-if="step.images && step.images.length" class="mt-3 grid grid-cols-6 gap-2">
          <img v-for="url in step.images" :key="url" :src="`${apiBase}${url}`"
            class="w-full aspect-square object-cover rounded border border-gray-200" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase
const { show: showError } = useToast()

const steps = ref([
  { key: "download", label: "Download Dataset (Kaggle)", locked: false, running: false, done: false, result: null, error: null, status: null, images: [] },
  { key: "convert", label: "Convert to YOLO-seg (Polygon Masks)", locked: true, running: false, done: false, result: null, error: null, status: null, images: [] },
  { key: "train", label: "Train YOLOv26m-seg Model", locked: true, running: false, done: false, result: null, error: null, status: null, images: [] },
  { key: "results", label: "Training Results & Curves", locked: true, running: false, done: false, result: null, error: null, status: null, images: [] },
  { key: "evaluate", label: "Validation & Test Evaluation", locked: true, running: false, done: false, result: null, error: null, status: null, images: [] },
  { key: "inference", label: "Inference (Upload Image)", locked: true, running: false, done: false, result: null, error: null, status: null, images: [] },
])

const stepsMap = computed(() => {
  const m: Record<string, any> = {}
  steps.value.forEach(s => m[s.key] = s)
  return m
})

onMounted(checkStatus)

async function checkStatus() {
  try {
    const res = await $fetch("/api/kaggle/download-status", { baseURL: apiBase }) as any
    if (res.dataset_exists) stepsMap.value.download.done = true
    if (res.labels_exist) stepsMap.value.convert.done = true
    unlockNext("download")
    if (stepsMap.value.download.done) unlockNext("convert")
    if (stepsMap.value.convert.done) unlockNext("train")
  } catch {}
}

function unlockNext(key: string) {
  const idx = steps.value.findIndex(s => s.key === key)
  if (idx >= 0 && idx + 1 < steps.value.length) {
    steps.value[idx + 1].locked = false
  }
}

async function runStep(key: string) {
  const step = stepsMap.value[key]
  if (!step || step.running) return
  step.running = true; step.error = null; step.result = null; step.images = []

  try {
    if (key === "download") {
      step.status = "Downloading dataset + generating masks..."
      const res = await $fetch("/api/kaggle/download", { baseURL: apiBase, method: "POST" })
      step.result = JSON.stringify(res, null, 2)
      step.done = true
      await checkStatus()
      unlockNext("convert")
    }
    else if (key === "convert") {
      step.status = "Generating polygon masks..."
      const res = await $fetch("/api/kaggle/convert", { baseURL: apiBase, method: "POST" })
      step.result = JSON.stringify(res, null, 2)
      step.done = true
      unlockNext("train")

      const viz = await $fetch("/api/kaggle/viz", { baseURL: apiBase }) as any
      if (viz.images) step.images = viz.images
      unlockNext("convert")
    }
    else if (key === "train") {
      step.status = "Training model (this takes time)..."
      const res = await $fetch("/api/kaggle/train?epochs=50&batch=16", { baseURL: apiBase, method: "POST" })
      step.result = JSON.stringify(res, null, 2)
      step.done = true
      unlockNext("results")
      unlockNext("evaluate")
      unlockNext("inference")
    }
    else if (key === "results") {
      const res = await $fetch("/api/kaggle/results", { baseURL: apiBase }) as any
      if (res.images?.length) {
        step.images = res.images.map((i: any) => i.url)
        step.result = `${res.images.length} result images found`
      } else {
        step.result = "No result images available"
      }
      step.done = true
    }
    else if (key === "evaluate") {
      step.status = "Evaluating on test set..."
      const res = await $fetch("/api/kaggle/evaluate", { baseURL: apiBase }) as any
      step.result = `mAP@50: ${(res.mAP50 * 100).toFixed(1)}% | mAP@50-95: ${(res.mAP50_95 * 100).toFixed(1)}% | Precision: ${(res.precision * 100).toFixed(1)}% | Recall: ${(res.recall * 100).toFixed(1)}%`
      step.done = true
    }
    else if (key === "inference") {
      step.status = "Upload image for inference"
      const input = document.createElement("input")
      input.type = "file"
      input.accept = "image/*"
      input.onchange = async () => {
        if (!input.files?.[0]) return
        step.status = "Running inference..."
        const form = new FormData()
        form.append("file", input.files[0])
        try {
          const res = await $fetch("/api/kaggle/inference", { baseURL: apiBase, method: "POST", body: form }) as any
          const blob = res as Blob
          const url = URL.createObjectURL(blob)
          step.images = [url]
          step.result = "Inference complete - image shown above"
          step.done = true
        } catch (e: any) {
          step.error = e?.data?.detail || e?.message || "Inference failed"
        }
      }
      input.click()
      return
    }
  } catch (e: any) {
    step.error = e?.data?.detail || e?.message || "Failed"
  } finally {
    step.running = false
  }
}
</script>
