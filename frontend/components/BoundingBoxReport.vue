<template>
  <div class="space-y-4">
    <!-- Annotated Image -->
    <div v-if="resultUrl" class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
      <div class="bg-gray-50 border-b border-gray-100 px-4 py-3">
        <h3 class="font-bold text-tertiary text-sm uppercase tracking-wide">Annotated Image</h3>
      </div>
      <div class="p-4">
        <img :src="resultUrl" class="w-full rounded-lg border border-gray-100" />
      </div>
    </div>

    <!-- Bounding Box Data Grid -->
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
      <div class="bg-gray-50 border-b border-gray-100 px-4 py-3 flex items-center justify-between">
        <h3 class="font-bold text-tertiary text-sm uppercase tracking-wide">Bounding Box Report</h3>
        <div class="flex gap-2">
          <button class="text-xs bg-secondary/10 text-secondary hover:bg-secondary/20 font-medium px-2.5 py-1 rounded transition-colors" @click="copyYolo">Copy YOLO</button>
          <button class="text-xs bg-secondary/10 text-secondary hover:bg-secondary/20 font-medium px-2.5 py-1 rounded transition-colors" @click="copyCsv">Copy CSV</button>
        </div>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="bg-gray-50 text-dark/60 text-xs uppercase tracking-wide sticky top-0">
            <tr>
              <th class="p-3 text-left font-medium">#</th>
              <th class="p-3 text-left font-medium">Label</th>
              <th class="p-3 text-left font-medium">Category</th>
              <th class="p-3 text-right font-medium">Confidence</th>
              <th class="p-3 text-right font-medium">X1</th>
              <th class="p-3 text-right font-medium">Y1</th>
              <th class="p-3 text-right font-medium">X2</th>
              <th class="p-3 text-right font-medium">Y2</th>
              <th class="p-3 text-right font-medium">W</th>
              <th class="p-3 text-right font-medium">H</th>
              <th class="p-3 text-left font-medium">YOLO Format</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(obj, i) in objects" :key="i"
              class="border-b border-gray-100 hover:bg-gray-50/50 transition-colors"
            >
              <td class="p-3 text-dark/50 font-mono text-xs">{{ i + 1 }}</td>
              <td class="p-3 font-medium text-dark">{{ obj.label }}</td>
              <td class="p-3">
                <span :class="obj.category === 'Organik' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'"
                  class="text-xs font-bold px-2 py-0.5 rounded">{{ obj.category }}</span>
              </td>
              <td class="p-3 text-right font-mono text-dark font-medium">{{ (obj.confidence * 100).toFixed(1) }}%</td>
              <td v-if="obj.bbox" class="p-3 text-right font-mono text-dark/70 text-xs">{{ obj.bbox[0].toFixed(0) }}</td>
              <td v-if="obj.bbox" class="p-3 text-right font-mono text-dark/70 text-xs">{{ obj.bbox[1].toFixed(0) }}</td>
              <td v-if="obj.bbox" class="p-3 text-right font-mono text-dark/70 text-xs">{{ obj.bbox[2].toFixed(0) }}</td>
              <td v-if="obj.bbox" class="p-3 text-right font-mono text-dark/70 text-xs">{{ obj.bbox[3].toFixed(0) }}</td>
              <td v-if="obj.bbox" class="p-3 text-right font-mono text-dark/70 text-xs">{{ (obj.bbox[2] - obj.bbox[0]).toFixed(0) }}</td>
              <td v-if="obj.bbox" class="p-3 text-right font-mono text-dark/70 text-xs">{{ (obj.bbox[3] - obj.bbox[1]).toFixed(0) }}</td>
              <td v-if="obj.bbox" class="p-3 font-mono text-xs text-dark/50">
                <code>{{ yoloLine(obj, i) }}</code>
              </td>
              <td v-else colspan="6" class="p-3 text-dark/40 text-xs">-</td>
            </tr>
            <tr v-if="!objects.length">
              <td colspan="11" class="p-6 text-center text-dark/40 text-sm">Tidak ada objek terdeteksi</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="border-t border-gray-100 px-4 py-3 flex items-center justify-between text-xs">
        <span class="text-dark/50">{{ objects.length }} objek terdeteksi</span>
        <div class="flex gap-3">
          <span class="text-green-700 font-medium">{{ organikCount }} Organik</span>
          <span class="text-blue-700 font-medium">{{ nonOrganikCount }} Non-Organik</span>
        </div>
      </div>
    </div>

    <!-- Copy toast -->
    <div v-if="copied" class="fixed bottom-4 right-4 bg-gray-700 text-white text-sm px-4 py-2 rounded-lg shadow-lg z-50 animate-pulse">
      Copied!
    </div>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  objects: any[]
  fileType?: string
  framesProcessed?: number
  resultUrl?: string
}>()

const copied = ref(false)

const organikCount = computed(() => props.objects.filter((o: any) => o.category === "Organik").length)
const nonOrganikCount = computed(() => props.objects.filter((o: any) => o.category === "Non-Organik").length)

function clsId(category: string) {
  return category === "Organik" ? 0 : 1
}

function yoloLine(obj: any, i: number) {
  if (!obj.bbox) return ""
  const [x1, y1, x2, y2] = obj.bbox
  const w = x2 - x1
  const h = y2 - y1
  return `${clsId(obj.category)} ${x1.toFixed(1)} ${y1.toFixed(1)} ${w.toFixed(1)} ${h.toFixed(1)}`
}

function yoloLines() {
  return props.objects
    .filter((o: any) => o.bbox)
    .map((o: any) => yoloLine(o, 0))
    .join("\n")
}

function csvLines() {
  const h = "#,Label,Category,Confidence,X1,Y1,X2,Y2,Width,Height"
  const rows = props.objects.map((o: any, i: number) => {
    if (!o.bbox) return `${i + 1},${o.label},${o.category},${(o.confidence * 100).toFixed(1)}%,-,-,-,-,-,-`
    const [x1, y1, x2, y2] = o.bbox
    return `${i + 1},${o.label},${o.category},${(o.confidence * 100).toFixed(1)}%,${x1.toFixed(0)},${y1.toFixed(0)},${x2.toFixed(0)},${(x2 - x1).toFixed(0)},${(y2 - y1).toFixed(0)}`
  }).join("\n")
  return h + "\n" + rows
}

async function copy(text: string) {
  try {
    await navigator.clipboard.writeText(text)
    copied.value = true
    setTimeout(() => copied.value = false, 1500)
  } catch {}
}

function copyYolo() { copy(yoloLines()) }
function copyCsv() { copy(csvLines()) }
</script>
