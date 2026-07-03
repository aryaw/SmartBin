<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex items-center justify-between">
      <div>
        <h1 class="font-bold text-lg text-dark">All Raw Data</h1>
        <p class="text-sm text-dark/50">{{ items.length }} images</p>
      </div>
      <div class="flex items-center gap-2">
        <button :disabled="downloading"
          class="bg-secondary hover:bg-blue-600 text-white text-sm font-bold px-4 py-2.5 rounded-xl shadow-sm transition-all flex items-center gap-2 disabled:opacity-50"
          @click="runDownload">
          <svg v-if="downloading" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          {{ downloading ? 'Downloading...' : 'Download Raw Data' }}
        </button>
        <button :disabled="splitting"
          class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-4 py-2.5 rounded-xl shadow-sm transition-all flex items-center gap-2 disabled:opacity-50"
          @click="runSplit">
        <svg v-if="splitting" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
        {{ splitting ? 'Splitting...' : 'Split Train/Test/Inference' }}
      </button>
      <button :disabled="resetting"
        class="bg-red-500 hover:bg-red-600 text-white text-sm font-bold px-4 py-2.5 rounded-xl shadow-sm transition-all flex items-center gap-2 disabled:opacity-50"
        @click="runReset">
        <svg v-if="resetting" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5">
          <path stroke-linecap="round" stroke-linejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
        </svg>
        {{ resetting ? 'Resetting...' : 'Reset Dataset' }}
      </button>
      </div>
    </div>

    <div v-if="downloadResult" class="bg-green-50 border border-green-200 rounded-xl p-4 text-sm text-green-700 flex items-center gap-3">
      <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <span>Downloaded {{ downloadResult.success }} images ({{ downloadResult.failed }} failed)</span>
    </div>

    <div v-if="splitResult" class="bg-green-50 border border-green-200 rounded-xl p-4 text-sm text-green-700 flex items-center gap-3">
      <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <span>Split done: {{ splitResult.train }} train, {{ splitResult.val }} inference, {{ splitResult.test }} test</span>
    </div>

    <div v-if="loading" class="flex justify-center py-12">
      <svg class="animate-spin h-8 w-8 text-tertiary" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
    </div>

    <div v-else-if="!paginatedItems.length" class="bg-white rounded-xl p-10 text-center border border-gray-100">
      <p class="text-dark/50 text-sm">No raw images in dataset.</p>
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
            <p class="text-xs text-dark/40">{{ img.size_kb }} KB</p>
          </div>
        </div>
      </div>
      <Pagination :page="page" :total="items.length" :per="perPage" @update:page="page = $event" />
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
const page = ref(1)
const perPage = 20
const zoomUrl = ref<string | null>(null)
const downloading = ref(false)
const downloadResult = ref<any>(null)
const splitting = ref(false)
const splitResult = ref<any>(null)
const resetting = ref(false)

const items = computed(() => data.value?.raw || [])
const paginatedItems = computed(() => {
  const start = (page.value - 1) * perPage
  return items.value.slice(start, start + perPage)
})

async function runDownload() {
  downloading.value = true; downloadResult.value = null
  try {
    downloadResult.value = await $fetch("/api/dataset/download", { baseURL: apiBase, method: "POST" })
    await load()
  } catch (e: any) { showError(e?.data?.detail || e?.message || 'Download failed') } finally { downloading.value = false }
}

async function runSplit() {
  splitting.value = true; splitResult.value = null
  try {
    splitResult.value = await $fetch("/api/dataset/split", { baseURL: apiBase, method: "POST" })
    await load()
  } catch (e: any) { showError(e?.data?.detail || e?.message || 'Split failed') } finally { splitting.value = false }
}

async function runReset() {
  resetting.value = true
  try {
    await $fetch("/api/dataset/reset", { baseURL: apiBase, method: "POST" })
    downloadResult.value = null
    splitResult.value = null
    await load()
  } catch (e: any) { showError(e?.data?.detail || e?.message || 'Reset failed') } finally { resetting.value = false }
}

async function load() {
  loading.value = true
  try { data.value = await $fetch("/api/dataset/grid", { baseURL: apiBase })
  } catch (e: any) { showError(e?.data?.detail || e?.message || 'Failed to load data') } finally { loading.value = false }
}

onMounted(load)
</script>
