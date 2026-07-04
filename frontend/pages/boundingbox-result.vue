<template>
  <div class="space-y-6">
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex items-center justify-between">
      <div>
        <h1 class="font-bold text-lg text-dark">BoundingBox Result</h1>
        <p class="text-sm text-dark/50">{{ cocoItems.length }} COCO annotated images</p>
      </div>
      <button :disabled="training"
        class="bg-green-600 hover:bg-green-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm transition-all flex items-center gap-2 disabled:opacity-50"
        @click="runFullTrain">
        <svg v-if="training" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        {{ training ? 'Training...' : 'Train Model' }}
      </button>
    </div>

    <div v-if="training" class="bg-white rounded-xl shadow-sm border border-gray-100 p-5 space-y-4">
      <div class="flex items-center gap-2 text-sm font-medium text-dark">
        <svg class="animate-spin h-4 w-4 text-green-600" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        Training in progress...
      </div>

      <div class="space-y-2">
        <div v-for="s in steps" :key="s.name"
          class="flex items-center gap-3 text-sm"
          :class="stepStatus(s.name) === 'active' ? 'text-dark' : stepStatus(s.name) === 'done' ? 'text-green-600' : 'text-dark/40'">
          <svg v-if="stepStatus(s.name) === 'done'" class="w-4 h-4 shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
          </svg>
          <svg v-else-if="stepStatus(s.name) === 'active'" class="animate-spin h-4 w-4 shrink-0 text-green-500" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span v-else class="w-4 h-4 shrink-0 rounded-full border-2 border-dark/20" />
          <span>{{ s.label }}</span>
          <span v-if="stepStatus(s.name) === 'active' && s.name === 'train' && currentEpoch"
            class="text-xs text-dark/50 ml-auto tabular-nums">
            Epoch {{ currentEpoch.epoch }} / {{ currentEpoch.epochs }}
            <span v-if="currentEpoch.loss"> | loss: {{ currentEpoch.loss }}</span>
          </span>
        </div>
      </div>

      <div v-if="currentEpoch && currentEpoch.epochs" class="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
        <div class="h-full bg-green-500 rounded-full transition-all duration-300"
          :style="{ width: (currentEpoch.epoch / currentEpoch.epochs * 100) + '%' }" />
      </div>

      <p v-if="progressMessage" class="text-xs text-dark/50">{{ progressMessage }}</p>
    </div>

    <div v-if="trainResult && !training" class="bg-green-50 border border-green-200 rounded-xl p-4 text-sm text-green-700">
      <div class="flex items-center gap-2 font-medium mb-1">
        <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
        </svg>
        Training Complete
      </div>
      <div class="text-xs ml-2 space-y-1">
        <div v-for="(v, k) in trainResult" :key="k" class="flex gap-2">
          <span class="text-dark/50 capitalize">{{ k }}:</span>
          <span class="font-medium">{{ v }}</span>
        </div>
      </div>
    </div>

    <div v-if="loading" class="flex justify-center py-12">
      <svg class="animate-spin h-8 w-8 text-tertiary" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
    </div>

    <div v-else-if="!paginatedItems.length" class="bg-white rounded-xl p-10 text-center border border-gray-100">
      <p class="text-dark/50 text-sm">No COCO annotations. Click Generate COCO above.</p>
    </div>

    <template v-else>
      <div class="space-y-4">
        <div v-for="img in paginatedItems" :key="img.filename"
          class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
          <div class="flex flex-col lg:flex-row">
            <div class="lg:w-72 p-3">
              <img :src="`${apiBase}${img.viz_url || img.image_url}`"
                class="w-full aspect-square object-cover rounded-lg border-2 border-green-200 cursor-pointer hover:opacity-90 transition-opacity"
                @click="zoomUrl = `${apiBase}${img.viz_url || img.image_url}`" />
              <p class="text-xs font-medium text-dark mt-2 truncate">{{ img.filename }}</p>
              <p class="text-xs text-dark/40">{{ img.annotations.length }} object(s)</p>
            </div>
            <div class="flex-1 overflow-x-auto p-3">
              <table class="w-full text-xs">
                <thead>
                  <tr class="text-dark/50 uppercase tracking-wide border-b border-gray-100">
                    <th class="p-2 text-left font-medium">#</th>
                    <th class="p-2 text-left font-medium">Class</th>
                    <th class="p-2 text-left font-medium">Category</th>
                    <th class="p-2 text-right font-medium">X</th>
                    <th class="p-2 text-right font-medium">Y</th>
                    <th class="p-2 text-right font-medium">W</th>
                    <th class="p-2 text-right font-medium">H</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(ann, j) in img.annotations" :key="j" class="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                    <td class="p-2 text-dark/40 font-mono">{{ j + 1 }}</td>
                    <td class="p-2 font-medium text-dark">{{ ann.class_id }}</td>
                    <td class="p-2">
                      <span :class="ann.category === 'Organik' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'"
                        class="font-bold px-2 py-0.5 rounded text-xs">{{ ann.category }}</span>
                    </td>
                    <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[0].toFixed(0) }}</td>
                    <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[1].toFixed(0) }}</td>
                    <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[2].toFixed(0) }}</td>
                    <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[3].toFixed(0) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
      <Pagination :page="page" :total="cocoItems.length" :per="perPage" @update:page="page = $event" />
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
const perPage = 10
const zoomUrl = ref<string | null>(null)
const training = ref(false)
const trainResult = ref<Record<string, any> | null>(null)

const steps = [
  { name: "coco", label: "Export COCO annotations" },
  { name: "coco_viz", label: "Render COCO visualizations" },
  { name: "yolo_inference", label: "YOLO inference on train images" },
  { name: "yolo_viz", label: "Render YOLO visualizations" },
  { name: "train", label: "Train YOLO model" },
]
const stepStates = ref<Record<string, "idle" | "active" | "done">>({})
const progressMessage = ref("")
const currentEpoch = ref<{ epoch: number; epochs: number; loss?: number } | null>(null)

function stepStatus(name: string) {
  return stepStates.value[name] || "idle"
}

const cocoItems = computed(() => (data.value?.coco || []).filter((img: any) => img.annotation_count > 0))

const paginatedItems = computed(() => {
  const start = (page.value - 1) * perPage
  return cocoItems.value.slice(start, start + perPage)
})

function runFullTrain() {
  training.value = true
  trainResult.value = null
  stepStates.value = {}
  currentEpoch.value = null
  progressMessage.value = ""

  const source = new EventSource(`${apiBase}/api/dataset/pipeline/yolo/train/stream`)

  source.addEventListener("step", (e: MessageEvent) => {
    const d = JSON.parse(e.data)
    if (d.step && d.status) {
      if (d.status === "start") stepStates.value[d.step] = "active"
      else if (d.status === "done") stepStates.value[d.step] = "done"
      else if (d.status === "epoch") {
        stepStates.value[d.step] = "active"
        currentEpoch.value = { epoch: d.epoch, epochs: d.epochs, loss: d.loss }
      }
    }
    if (d.message) progressMessage.value = d.message
  })

  source.addEventListener("done", (e: MessageEvent) => {
    const d = JSON.parse(e.data)
    trainResult.value = { ...d }
    source.close()
    training.value = false
    load()
  })

  source.addEventListener("error", (e: MessageEvent) => {
    const d = e.data ? JSON.parse(e.data) : { message: "Training failed" }
    source.close()
    training.value = false
    showError(d.message || "Training failed")
  })

  source.onerror = () => {
    source.close()
    training.value = false
    showError("Connection lost during training")
  }
}

async function load() {
  loading.value = true
  try {
    data.value = await $fetch("/api/dataset/grid", { baseURL: apiBase })
  } catch (e: any) { showError(e?.data?.detail || e?.message || 'Failed to load data') } finally { loading.value = false }
}

onMounted(load)
</script>
