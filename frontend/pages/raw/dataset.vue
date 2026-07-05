<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
      <h1 class="font-bold text-lg text-dark">Dataset</h1>
      <p class="text-sm text-dark/50">Load and profile the waste classification dataset</p>
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
            <td class="px-4 py-3 font-medium">Load Dataset</td>
            <td class="px-4 py-3 text-dark/60">Scan Waste_Classification_Dataset and register metadata</td>
            <td class="px-4 py-3 text-center">
              <span v-if="loadStatus === 'done'" class="text-green-600 bg-green-50 px-2 py-0.5 rounded">Done</span>
              <span v-else-if="loadStatus === 'running'" class="text-blue-600 bg-blue-50 px-2 py-0.5 rounded">Running</span>
              <span v-else class="text-gray-400 bg-gray-50 px-2 py-0.5 rounded">Not Started</span>
            </td>
            <td class="px-4 py-3 text-dark/60">{{ loadSummary }}</td>
            <td class="px-4 py-3 text-center">
              <button :disabled="loadStatus === 'running'" class="px-3 py-1 text-xs font-bold rounded bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runLoad">Load Dataset</button>
            </td>
          </tr>
          <tr>
            <td class="px-4 py-3 font-medium">Dataset Profiling</td>
            <td class="px-4 py-3 text-dark/60">Analyze structure, classes, image counts, and balance</td>
            <td class="px-4 py-3 text-center">
              <span v-if="profileStatus === 'done'" class="text-green-600 bg-green-50 px-2 py-0.5 rounded">Done</span>
              <span v-else-if="profileStatus === 'running'" class="text-blue-600 bg-blue-50 px-2 py-0.5 rounded">Running</span>
              <span v-else class="text-gray-400 bg-gray-50 px-2 py-0.5 rounded">Not Started</span>
            </td>
            <td class="px-4 py-3 text-dark/60">{{ profileSummary }}</td>
            <td class="px-4 py-3 text-center">
              <div class="flex items-center justify-center gap-1">
                <button :disabled="profileStatus === 'running'" class="px-3 py-1 text-xs font-bold rounded bg-tertiary text-white hover:bg-blue-700 disabled:opacity-40" @click="runProfile">Run Profiling</button>
                <button v-if="profileReport" class="px-3 py-1 text-xs font-bold rounded bg-gray-100 text-dark/50 hover:bg-gray-200" @click="showReport = !showReport">View Report</button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="showReport && profileReport" class="bg-gray-50 rounded-xl p-4 text-xs font-mono max-h-96 overflow-auto">
      <pre class="text-green-700">{{ profileReport }}</pre>
    </div>
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase
const { show: showError } = useToast()
const loadStatus = ref('not_started'); const loadSummary = ref('')
const profileStatus = ref('not_started'); const profileSummary = ref(''); const profileReport = ref(''); const showReport = ref(false)

async function runLoad() {
  loadStatus.value = 'running'
  try {
    const res = await $fetch("/api/kaggle/download?source=local", { baseURL: apiBase, method: "POST" }) as any
    loadSummary.value = `${res.total_images} images, ${res.classes} classes`
    loadStatus.value = 'done'
  } catch (e: any) { showError(e?.data?.detail || e?.message); loadStatus.value = 'not_started' }
}
async function runProfile() {
  profileStatus.value = 'running'; profileReport.value = ''
  try {
    const res = await $fetch("/api/kaggle/explore", { baseURL: apiBase }) as any
    let out = `${res.total_images} images across ${res.num_classes} classes\n\n`
    const maxC = Math.max(...res.distribution.map((d: any) => d.count))
    for (const d of res.distribution) {
      const bar = '█'.repeat(Math.round(d.count / maxC * 25))
      out += `  [${d.class_id}] ${d.name.padEnd(25)} ${String(d.count).padStart(4)} ${bar}\n`
    }
    profileReport.value = out
    profileSummary.value = `${res.num_classes} classes, ${res.total_images} total images`
    profileStatus.value = 'done'
  } catch (e: any) { showError(e?.data?.detail || e?.message); profileStatus.value = 'not_started' }
}
</script>
