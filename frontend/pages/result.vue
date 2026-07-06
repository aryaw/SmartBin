<template>
  <div class="min-h-screen">
    <header class="bg-gradient-to-br from-tertiary to-secondary text-white text-center p-8 shadow-lg">
      <div class="flex items-center justify-center gap-3 mb-2">
        <div class="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center text-2xl font-bold">
          SB
        </div>
        <h1 class="text-4xl font-bold tracking-tight">Hasil Deteksi</h1>
      </div>
      <p v-if="data" class="text-white/80 text-sm truncate">{{ data.filename }}</p>
    </header>

    <main class="max-w-3xl mx-auto p-6 -mt-6" v-if="data">
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
        <div class="bg-gray-50 border-b border-gray-100 px-5 py-3 flex items-center justify-between">
          <div class="flex items-center gap-2">
            <div class="w-2.5 h-2.5 rounded-full" :class="data.file_type === 'image' ? 'bg-blue-500' : 'bg-purple-500'"></div>
            <span class="font-medium text-dark text-sm">{{ data.file_type === 'image' ? 'Gambar' : 'Video' }}</span>
          </div>
          <span :class="data.file_type === 'image' ? 'bg-blue-100 text-blue-700' : 'bg-purple-100 text-purple-700'"
            class="text-xs font-bold px-2.5 py-1 rounded-full">{{ data.file_type }}</span>
        </div>
        <div class="p-5 flex justify-center">
          <img v-if="data.file_type === 'image'" :src="`${apiBase}${data.result_url}`"
            class="max-w-[500px] w-full rounded-lg shadow-sm border border-gray-100 cursor-pointer hover:opacity-90 transition-opacity"
            @click="previewUrl = `${apiBase}${data.result_url}`; previewCaption = data.filename" />
          <video v-else :src="`${apiBase}${data.result_url}`" controls
            class="max-w-[500px] w-full rounded-lg shadow-sm border border-gray-100" />
        </div>
      </div>

      <div class="mt-6 grid grid-cols-3 gap-4">
        <div class="bg-green-50 border border-green-200 rounded-xl p-4 text-center">
          <p class="text-3xl font-bold text-green-700">{{ data.summary.organik }}</p>
          <p class="text-xs text-green-600 font-medium mt-1">Organik</p>
        </div>
        <div class="bg-blue-50 border border-blue-200 rounded-xl p-4 text-center">
          <p class="text-3xl font-bold text-blue-700">{{ data.summary.non_organik }}</p>
          <p class="text-xs text-blue-600 font-medium mt-1">Non-Organik</p>
        </div>
        <div class="bg-gray-50 border border-gray-200 rounded-xl p-4 text-center">
          <p class="text-3xl font-bold">{{ data.summary.total }}</p>
          <p class="text-xs text-dark/60 font-medium mt-1">Total</p>
        </div>
      </div>

      <div class="mt-6 bg-white rounded-xl shadow-sm border border-gray-100 p-5">
        <p class="text-sm font-medium text-dark/60 uppercase tracking-wide">Rekomendasi</p>
        <p class="text-tertiary font-bold text-lg mt-1">{{ data.recommendation }}</p>
      </div>

      <div class="mt-6">
        <BoundingBoxReport :objects="data.detected_objects" :file-type="data.file_type"
          :frames-processed="data.frames_processed" :result-url="`${apiBase}${data.result_url}`" />
      </div>

      <div class="mt-6 flex gap-3">
        <button class="flex-1 bg-gradient-to-r from-secondary to-tertiary hover:from-tertiary hover:to-tertiary text-white font-bold py-3.5 px-6 rounded-xl transition-all shadow-md hover:shadow-lg flex items-center justify-center gap-2"
          @click="navigateTo('/')">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Deteksi Lagi
        </button>
        <button class="bg-gray-700 hover:bg-gray-800 text-white font-bold py-3.5 px-6 rounded-xl transition-all shadow-md flex items-center justify-center gap-2"
          @click="downloadReport">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Download Report
        </button>
      </div>
    </main>

    <p v-else class="text-center text-dark/60 mt-12">Tidak ada data deteksi.</p>

    <footer class="bg-gradient-to-r from-gray-700 to-gray-800 text-gray-300 text-center p-5 text-sm mt-12">
      SmartBin &copy; 2026 &mdash; Sistem Deteksi Sampah Organik & Non-Organik
    </footer>

    <ZoomModal :url="previewUrl" :caption="previewCaption" @close="previewUrl = null" />
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase

const route = useRoute()
const raw = route.query.data as string
const data = raw ? JSON.parse(raw) : null

const previewUrl = ref<string | null>(null)
const previewCaption = ref("")

function downloadReport() {
  if (!data.value) return
  const header = "Label,Category,Confidence,BBOX_X1,BBOX_Y1,BBOX_X2,BBOX_Y2\n"
  const rows = data.value.detected_objects.map((o: any) => {
    const bbox = o.bbox ? o.bbox.join(",") : "N/A"
    return `${o.label},${o.category},${(o.confidence * 100).toFixed(2)}%,${bbox}`
  }).join("\n")
  const csv = header + rows
  const blob = new Blob([csv], { type: "text/csv" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `smartbin_report_${data.value.filename}.csv`
  a.click()
  URL.revokeObjectURL(url)
}
</script>
