<template>
  <div class="space-y-6 max-w-3xl">

    <div
      class="bg-white rounded-xl shadow-sm border-2 border-dashed border-secondary p-10 text-center cursor-pointer hover:border-tertiary hover:bg-secondary/10 transition-all"
      @drop="onDrop" @dragover.prevent @click="inputRef?.click()"
    >
      <div class="w-16 h-16 bg-secondary/20 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg class="w-8 h-8 text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
      </div>
      <p class="text-dark font-medium text-lg">Drag & drop file di sini</p>
      <p class="text-dark/50 text-sm mt-1">atau klik untuk browse</p>
      <p class="text-dark/40 text-xs mt-2">JPG, JPEG, PNG, MP4, AVI, MOV (maks 200MB)</p>
      <input ref="inputRef" type="file" class="hidden" :accept="accepts" multiple @change="onFileChange" />
    </div>

    <div v-if="files.length" class="bg-white rounded-xl shadow-sm p-4 space-y-2">
      <div class="flex justify-between items-center mb-2">
        <h2 class="font-bold text-tertiary text-sm uppercase tracking-wide">File Terpilih</h2>
        <span class="text-xs text-dark/50">{{ files.length }} file</span>
      </div>
      <div v-for="(f, i) in files" :key="i"
        class="flex items-center gap-3 bg-gray-50 rounded-lg p-3 border border-gray-100"
      >
        <div class="w-12 h-12 rounded-lg overflow-hidden bg-gray-100 flex-shrink-0">
          <img v-if="f.type.startsWith('image/')" :src="f.preview" class="w-full h-full object-cover" />
          <div v-else class="w-full h-full flex items-center justify-center text-secondary">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
        </div>
        <div class="flex-1 min-w-0">
          <p class="text-sm font-medium text-dark truncate">{{ f.file.name }}</p>
          <p class="text-xs text-dark/50">{{ (f.file.size / 1024 / 1024).toFixed(1) }} MB</p>
        </div>
        <button class="w-8 h-8 bg-red-50 text-red-500 hover:bg-red-100 rounded-full flex items-center justify-center transition-colors" @click="removeFile(i)">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>

    <div v-if="errors.length" class="bg-red-50 border border-red-200 rounded-xl p-4 space-y-1">
      <p v-for="(e, i) in errors" :key="i" class="text-red-600 text-sm flex items-center gap-2">
        <svg class="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
        </svg>
        {{ e }}
      </p>
    </div>

    <button
      v-if="files.length"
      class="w-full bg-tertiary hover:bg-blue-700 text-white font-bold py-3.5 px-6 rounded-xl transition-all disabled:opacity-50 flex items-center justify-center gap-2 shadow-sm hover:shadow"
      :disabled="loading"
      @click="detectAll"
    >
      <svg v-if="loading" class="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
      </svg>
      <span v-else>
        <svg class="w-5 h-5 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </span>
      {{ loading ? `Memproses ${done}/${files.length}...` : `Deteksi ${files.length} File` }}
    </button>

    <div v-if="results.length" class="space-y-4">
      <h2 class="font-bold text-xl text-tertiary flex items-center gap-2">
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
        </svg>
        Hasil Deteksi
      </h2>
      <div v-for="(r, i) in results" :key="i" class="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
        <div class="flex justify-between items-center mb-3">
          <div class="flex items-center gap-2 min-w-0">
            <div class="w-2 h-2 rounded-full" :class="r.file_type === 'image' ? 'bg-blue-500' : 'bg-purple-500'"></div>
            <span class="font-medium text-sm text-dark truncate">{{ r.filename }}</span>
          </div>
          <span :class="r.file_type === 'image' ? 'bg-blue-100 text-blue-700' : 'bg-purple-100 text-purple-700'"
            class="text-xs font-bold px-2.5 py-1 rounded-full flex-shrink-0">{{ r.file_type }}</span>
        </div>
        <img v-if="r.file_type === 'image'" :src="`${apiBase}${r.result_url}`" class="max-h-48 mx-auto rounded-lg mb-3 border border-gray-100" />
        <div class="grid grid-cols-3 gap-2 mb-3">
          <div class="bg-green-50 rounded-lg p-2 text-center">
            <p class="text-lg font-bold text-green-700">{{ r.summary.organik }}</p>
            <p class="text-xs text-green-600">Organik</p>
          </div>
          <div class="bg-blue-50 rounded-lg p-2 text-center">
            <p class="text-lg font-bold text-blue-700">{{ r.summary.non_organik }}</p>
            <p class="text-xs text-blue-600">Non-Organik</p>
          </div>
          <div class="bg-gray-50 rounded-lg p-2 text-center">
            <p class="text-lg font-bold">{{ r.summary.total }}</p>
            <p class="text-xs text-dark/60">Total</p>
          </div>
        </div>
        <button class="text-sm font-medium text-secondary hover:text-tertiary transition-colors flex items-center gap-1"
          @click="viewDetail(r)">
          Lihat Detail <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" /></svg>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase

const accepts = ".jpg,.jpeg,.png,.mp4,.avi,.mov"
const inputRef = ref<HTMLInputElement | null>(null)
const loading = ref(false)
const done = ref(0)
const errors = ref<string[]>([])
const results = ref<any[]>([])

interface FileItem { file: File; preview: string; type: string }
const files = ref<FileItem[]>([])

const STORAGE_KEY = "smartbin_history"

function saveToHistory(result: any) {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    const history = raw ? JSON.parse(raw) : []
    history.unshift({ ...result, time: new Date().toLocaleString("id-ID"), savedAt: Date.now() })
    if (history.length > 50) history.length = 50
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
  } catch {}
}

function onFileChange(e: Event) {
  const target = e.target as HTMLInputElement
  if (target.files?.length) { for (const f of target.files) addFile(f) }
  target.value = ""
}

function onDrop(e: DragEvent) {
  if (e.dataTransfer?.files?.length) { for (const f of e.dataTransfer.files) addFile(f) }
}

function addFile(f: File) {
  const ext = f.name.split(".").pop()?.toLowerCase()
  if (!["jpg", "jpeg", "png", "mp4", "avi", "mov"].includes(ext || "")) {
    errors.value.push(`${f.name}: Format tidak didukung`); return
  }
  if (f.size > 200 * 1024 * 1024) {
    errors.value.push(`${f.name}: File terlalu besar (maks 200MB)`); return
  }
  errors.value = []
  files.value.push({ file: f, preview: URL.createObjectURL(f), type: f.type })
}

function removeFile(i: number) {
  URL.revokeObjectURL(files.value[i].preview)
  files.value.splice(i, 1)
}

async function detectAll() {
  loading.value = true; done.value = 0; results.value = []; errors.value = []
  for (const f of files.value) {
    try {
      const form = new FormData(); form.append("file", f.file)
      const res: any = await $fetch("/api/detect", { baseURL: apiBase, method: "POST", body: form })
      results.value.push(res)
      saveToHistory(res)
    } catch (e: any) {
      errors.value.push(`${f.file.name}: ${e.data?.detail || "Gagal"}`)
    } finally { done.value++ }
  }
  loading.value = false
}

function viewDetail(r: any) {
  navigateTo(`/result?data=${encodeURIComponent(JSON.stringify(r))}`)
}
</script>
