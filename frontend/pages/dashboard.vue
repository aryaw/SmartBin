<template>
  <div class="space-y-6 max-w-4xl">
    <div class="bg-gradient-to-r from-tertiary to-secondary rounded-xl p-6 text-white shadow-sm">
      <h1 class="text-2xl font-bold">Dashboard Report</h1>
      <p class="text-white/80 text-sm mt-1">Upload file untuk deteksi sampah organik & non-organik</p>
    </div>

    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex items-center justify-between">
      <div>
        <h2 class="font-bold text-dark text-sm uppercase tracking-wide">Full Pipeline</h2>
        <p class="text-xs text-dark/50">Load dataset -> Convert masks -> Train YOLOv26m-seg -> Then infer on your images</p>
      </div>
      <button :disabled="pipelineRunning"
        class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm transition-all flex items-center gap-2 disabled:opacity-50"
        @click="runFullPipeline">
        <svg v-if="pipelineRunning" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
        </svg>
        {{ pipelineRunning ? 'Running pipeline...' : 'Run Full Pipeline' }}
      </button>
    </div>

    <div v-if="pipelineResult" class="bg-green-50 border border-green-200 rounded-xl p-4 text-sm text-green-700 flex items-center gap-3">
      <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <span>{{ pipelineResult }}</span>
    </div>

    <div class="bg-white rounded-xl shadow-sm border-2 border-dashed border-secondary p-10 text-center cursor-pointer hover:border-tertiary hover:bg-secondary/10 transition-all"
      @drop="onDrop" @dragover.prevent @click="inputRef?.click()">
      <div class="w-16 h-16 bg-secondary/20 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg class="w-8 h-8 text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
      </div>
      <p class="text-dark font-medium text-lg">Drag & drop file di sini</p>
      <p class="text-dark/50 text-sm mt-1">atau klik untuk browse</p>
      <p class="text-dark/40 text-xs mt-2">JPG, JPEG, PNG, MP4, AVI, MOV (maks 200MB)</p>
      <input ref="inputRef" type="file" class="hidden" accept=".jpg,.jpeg,.png,.mp4,.avi,.mov" @change="onFileChange" />
    </div>

    <div v-if="fileItem" class="bg-white rounded-xl shadow-sm p-4 border border-gray-100 flex items-center gap-3">
      <div class="w-12 h-12 rounded-lg overflow-hidden bg-gray-100 flex-shrink-0">
        <div class="w-full h-full flex items-center justify-center text-secondary">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M18.75 10.5a3 3 0 100-6 3 3 0 000 6z" />
          </svg>
        </div>
      </div>
      <div class="flex-1 min-w-0">
        <p class="text-sm font-medium text-dark truncate">{{ fileItem.file.name }}</p>
        <p class="text-xs text-dark/50">{{ (fileItem.file.size / 1024 / 1024).toFixed(1) }} MB</p>
      </div>
      <button class="w-8 h-8 bg-red-50 text-red-500 hover:bg-red-100 rounded-full flex items-center justify-center transition-colors" @click="fileItem = null; result = null">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>

    <button v-if="fileItem" :disabled="loading"
      class="w-full bg-tertiary hover:bg-blue-700 text-white font-bold py-3.5 px-6 rounded-xl transition-all disabled:opacity-50 flex items-center justify-center gap-2 shadow-sm"
      @click="detect">
      <svg v-if="loading" class="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
      <span v-else>
        <svg class="w-5 h-5 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </span>
      {{ loading ? 'Memproses...' : 'Deteksi Sekarang' }}
    </button>

    <div v-if="result" class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
      <div class="bg-gray-50 border-b border-gray-100 px-5 py-3 flex items-center justify-between">
        <span class="font-medium text-dark text-sm">Hasil Deteksi</span>
        <span class="bg-blue-100 text-blue-700 text-xs font-bold px-2.5 py-1 rounded-full">{{ result.file_type }}</span>
      </div>
      <div class="p-5">
        <img v-if="result.file_type === 'image'" :src="`${apiBase}${result.result_url}`" class="w-full rounded-lg border border-gray-100" />
        <video v-else :src="`${apiBase}${result.result_url}`" controls class="w-full rounded-lg border border-gray-100" />
      </div>
      <div class="grid grid-cols-3 gap-4 px-5 pb-5">
        <div class="bg-green-50 border border-green-200 rounded-xl p-4 text-center">
          <p class="text-3xl font-bold text-green-700">{{ result.summary.organik }}</p>
          <p class="text-xs text-green-600 font-medium mt-1">Organik</p>
        </div>
        <div class="bg-blue-50 border border-blue-200 rounded-xl p-4 text-center">
          <p class="text-3xl font-bold text-blue-700">{{ result.summary.non_organik }}</p>
          <p class="text-xs text-blue-600 font-medium mt-1">Non-Organik</p>
        </div>
        <div class="bg-gray-50 border border-gray-200 rounded-xl p-4 text-center">
          <p class="text-3xl font-bold">{{ result.summary.total }}</p>
          <p class="text-xs text-dark/60 font-medium mt-1">Total</p>
        </div>
      </div>
      <div class="px-5 pb-5">
        <p class="text-sm font-medium text-dark/60 uppercase tracking-wide">Rekomendasi</p>
        <p class="text-tertiary font-bold text-lg mt-1">{{ result.recommendation }}</p>
      </div>
    </div>

    <div v-if="result" class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
      <div class="bg-gray-50 border-b border-gray-100 px-5 py-3">
        <h3 class="font-bold text-dark/60 uppercase tracking-wide text-sm">Bounding Box Report</h3>
      </div>
      <BoundingBoxReport :objects="result.detected_objects" :file-type="result.file_type"
        :frames-processed="result.frames_processed" :result-url="`${apiBase}${result.result_url}`" />
    </div>

    <div v-if="error" class="bg-red-50 border border-red-200 rounded-xl p-4">
      <p class="text-red-600 text-sm flex items-center gap-2">
        <svg class="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
        </svg>
        {{ error }}
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase
const { show: showError } = useToast()

const inputRef = ref<HTMLInputElement | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
const result = ref<any>(null)
const fileItem = ref<{ file: File } | null>(null)
const pipelineRunning = ref(false)
const pipelineResult = ref<string | null>(null)

function onFileChange(e: Event) {
  const target = e.target as HTMLInputElement
  if (target.files?.length) addFile(target.files[0])
  target.value = ""
}

function onDrop(e: DragEvent) {
  if (e.dataTransfer?.files?.length) addFile(e.dataTransfer.files[0])
}

function addFile(f: File) {
  const ext = f.name.split(".").pop()?.toLowerCase()
  if (!["jpg", "jpeg", "png", "mp4", "avi", "mov"].includes(ext || "")) {
    error.value = `${f.name}: Format tidak didukung`; return
  }
  if (f.size > 200 * 1024 * 1024) {
    error.value = `${f.name}: File terlalu besar (maks 200MB)`; return
  }
  error.value = null; result.value = null
  fileItem.value = { file: f }
}

async function detect() {
  if (!fileItem.value) return
  loading.value = true; error.value = null; result.value = null
  try {
    const form = new FormData()
    form.append("file", fileItem.value.file)
    result.value = await $fetch("/api/detect", { baseURL: apiBase, method: "POST", body: form })
  } catch (e: any) {
    error.value = e.data?.detail || e.message || "Deteksi gagal"
  } finally { loading.value = false }
}

async function runFullPipeline() {
  pipelineRunning.value = true; pipelineResult.value = null
  try {
    const res = await $fetch("/api/kaggle/pipeline/run-full?epochs=50&batch=16", { baseURL: apiBase, method: "POST" }) as any
    pipelineResult.value = res.message || "Pipeline complete. Model is ready. Upload an image below to detect."
  } catch (e: any) { showError(e?.data?.detail || e?.message || 'Pipeline failed') } finally { pipelineRunning.value = false }
}
</script>
