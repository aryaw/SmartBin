<template>
  <div class="space-y-6">

    <!-- Empty State -->
    <div v-if="!history.length" class="max-w-lg mx-auto text-center py-16">
      <div class="w-20 h-20 bg-gradient-to-br from-blue-400 to-blue-500 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg shadow-blue-200">
        <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
      </div>
      <h2 class="text-2xl font-bold text-gray-800 mb-2">Welcome to Smart Bin</h2>
      <p class="text-gray-500 mb-6">Upload gambar atau video untuk memulai deteksi sampah organik dan non-organik.</p>
      <NuxtLink to="/test" class="inline-flex items-center gap-2 bg-gradient-to-r from-blue-400 to-blue-500 text-white font-semibold px-6 py-3 rounded-xl shadow-md hover:shadow-lg transition-shadow">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15"/>
        </svg>
        Upload File untuk Deteksi
      </NuxtLink>
    </div>

    <!-- Dashboard -->
    <div v-if="history.length" class="space-y-6">
      <!-- Stats Row -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="bg-gradient-to-br from-blue-400 to-blue-500 rounded-xl p-4 text-white shadow-sm">
          <div class="flex items-center justify-between mb-2">
            <p class="text-sm font-medium text-blue-100">Total Deteksi</p>
            <div class="w-9 h-9 bg-white/20 rounded-lg flex items-center justify-center">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25zM6.75 12h.008v.008H6.75V12zm0 3h.008v.008H6.75V15zm0 3h.008v.008H6.75V18z"/>
              </svg>
            </div>
          </div>
          <p class="text-3xl font-bold">{{ history.length }}</p>
          <p class="text-xs text-blue-100 mt-1">{{ todayCount }} hari ini</p>
        </div>

        <div class="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <div class="flex items-center justify-between mb-2">
            <p class="text-sm font-medium text-gray-500">Organik</p>
            <div class="w-9 h-9 bg-green-50 rounded-lg flex items-center justify-center">
              <svg class="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z"/>
              </svg>
            </div>
          </div>
          <p class="text-3xl font-bold text-green-600">{{ totalOrganik }}</p>
          <p class="text-xs text-gray-400 mt-1">{{ organikPct }}% dari total</p>
        </div>

        <div class="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <div class="flex items-center justify-between mb-2">
            <p class="text-sm font-medium text-gray-500">Non-Organik</p>
            <div class="w-9 h-9 bg-blue-50 rounded-lg flex items-center justify-center">
              <svg class="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9.53 16.122a3 3 0 00-5.78 1.128 2.25 2.25 0 01-2.4 2.245 4.5 4.5 0 008.4-2.245c0-.399-.078-.78-.22-1.128zm0 0a15.998 15.998 0 003.388-1.62m-5.043-.025a15.994 15.994 0 011.622-3.395m3.42 3.42a15.995 15.995 0 004.764-4.648l3.876-5.814a1.151 1.151 0 00-1.597-1.597L14.146 6.32a15.996 15.996 0 00-4.649 4.763m3.42 3.42a6.776 6.776 0 00-3.42-3.42"/>
              </svg>
            </div>
          </div>
          <p class="text-3xl font-bold text-blue-600">{{ totalNonOrganik }}</p>
          <p class="text-xs text-gray-400 mt-1">{{ nonOrganikPct }}% dari total</p>
        </div>

        <div class="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <div class="flex items-center justify-between mb-2">
            <p class="text-sm font-medium text-gray-500">Rasio</p>
            <div class="w-9 h-9 bg-purple-50 rounded-lg flex items-center justify-center">
              <svg class="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6a7.5 7.5 0 107.5 7.5h-7.5V6z"/>
                <path stroke-linecap="round" stroke-linejoin="round" d="M13.5 10.5H21A7.5 7.5 0 0013.5 3v7.5z"/>
              </svg>
            </div>
          </div>
          <p class="text-3xl font-bold text-gray-800">{{ organikPct }}:{{ nonOrganikPct }}</p>
          <p class="text-xs text-gray-400 mt-1">Organik : Non-Organik</p>
        </div>
      </div>

      <!-- Ratio Bar -->
      <div class="bg-white rounded-xl p-5 border border-gray-100 shadow-sm">
        <div class="flex items-center justify-between mb-3">
          <h3 class="font-semibold text-gray-800 text-sm">Distribusi Kategori</h3>
          <span class="text-xs text-gray-400">{{ history.length }} total deteksi</span>
        </div>
        <div class="flex gap-1 h-4 rounded-full overflow-hidden">
          <div class="bg-green-400 transition-all" :style="{ width: organikPct + '%' }" :title="`Organik ${organikPct}%`"></div>
          <div class="bg-blue-400 transition-all" :style="{ width: nonOrganikPct + '%' }" :title="`Non-Organik ${nonOrganikPct}%`"></div>
        </div>
        <div class="flex justify-between text-xs text-gray-400 mt-2">
          <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-green-400"></span> Organik {{ organikPct }}%</span>
          <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-blue-400"></span> Non-Organik {{ nonOrganikPct }}%</span>
        </div>
      </div>

      <!-- History -->
      <div class="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
        <div class="flex items-center justify-between px-5 py-4 border-b border-gray-100">
          <h3 class="font-semibold text-gray-800 text-sm">Riwayat Deteksi</h3>
          <div class="flex items-center gap-3">
            <span class="text-xs text-gray-400">{{ history.length }} file</span>
            <button class="text-xs text-red-400 hover:text-red-600 font-medium" @click="clearHistory">Hapus</button>
          </div>
        </div>
        <div class="divide-y divide-gray-50 max-h-80 overflow-y-auto">
          <div v-for="(item, i) in recentHistory" :key="i"
            class="flex items-center gap-3 px-5 py-3 hover:bg-gray-50 cursor-pointer transition-colors"
            @click="navigateToResult(item)"
          >
            <div class="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
              :class="item.file_type === 'image' ? 'bg-blue-50' : 'bg-purple-50'">
              <svg class="w-5 h-5" :class="item.file_type === 'image' ? 'text-blue-500' : 'text-purple-500'" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5">
                <path v-if="item.file_type === 'image'" stroke-linecap="round" stroke-linejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M18.75 10.5a3 3 0 100-6 3 3 0 000 6z"/>
                <path v-else stroke-linecap="round" stroke-linejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"/>
              </svg>
            </div>
            <div class="flex-1 min-w-0">
              <p class="text-sm font-medium text-gray-800 truncate">{{ item.filename }}</p>
              <p class="text-xs text-gray-400">{{ item.time }}</p>
            </div>
            <div class="text-right flex-shrink-0">
              <p class="text-sm font-semibold">
                <span class="text-green-600">{{ item.summary.organik }}</span>
                <span class="text-gray-300 mx-1">/</span>
                <span class="text-blue-600">{{ item.summary.non_organik }}</span>
              </p>
              <p class="text-xs text-gray-400">{{ item.summary.total }} total</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const STORAGE_KEY = "smartbin_history"
const history = ref<any[]>([])

function loadHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    history.value = raw ? JSON.parse(raw) : []
  } catch { history.value = [] }
}

function clearHistory() {
  history.value = []
  localStorage.removeItem(STORAGE_KEY)
}

const totalOrganik = computed(() => history.value.reduce((s: number, h: any) => s + (h.summary?.organik || 0), 0))
const totalNonOrganik = computed(() => history.value.reduce((s: number, h: any) => s + (h.summary?.non_organik || 0), 0))
const grandTotal = computed(() => totalOrganik.value + totalNonOrganik.value)
const organikPct = computed(() => grandTotal.value ? Math.round(totalOrganik.value / grandTotal.value * 100) : 0)
const nonOrganikPct = computed(() => grandTotal.value ? Math.round(totalNonOrganik.value / grandTotal.value * 100) : 0)

const todayCount = computed(() => {
  const today = new Date().toDateString()
  return history.value.filter((h: any) => new Date(h.savedAt).toDateString() === today).length
})

const recentHistory = computed(() => history.value.slice(0, 20))

function navigateToResult(item: any) {
  navigateTo(`/result?data=${encodeURIComponent(JSON.stringify(item))}`)
}

onMounted(loadHistory)
</script>
