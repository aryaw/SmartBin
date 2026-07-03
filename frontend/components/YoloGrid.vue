<template>
  <div class="space-y-4">
    <div v-for="img in items" :key="img.filename"
      class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden"
    >
      <div class="flex flex-col lg:flex-row">
        <div class="lg:w-72 p-3">
          <img :src="`${apiBase}${img.viz_url}`"
            class="w-full aspect-square object-cover rounded-lg border-2 border-blue-200 cursor-pointer hover:opacity-90 transition-opacity"
            @click="$emit('preview', `${apiBase}${img.viz_url}`)" />
          <p class="text-xs font-medium text-dark mt-2 truncate">{{ img.filename }}</p>
          <p class="text-xs text-dark/40">{{ img.predictions.length }} prediction(s)</p>
        </div>
        <div class="flex-1 overflow-x-auto p-3">
          <table class="w-full text-xs">
            <thead>
              <tr class="text-dark/50 uppercase tracking-wide border-b border-gray-100">
                <th class="p-2 text-left font-medium">#</th>
                <th class="p-2 text-left font-medium">Class</th>
                <th class="p-2 text-left font-medium">Category</th>
                <th class="p-2 text-right font-medium">Conf</th>
                <th class="p-2 text-right font-medium">X1</th>
                <th class="p-2 text-right font-medium">Y1</th>
                <th class="p-2 text-right font-medium">X2</th>
                <th class="p-2 text-right font-medium">Y2</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(pred, j) in img.predictions" :key="j"
                class="border-b border-gray-50 hover:bg-gray-50 transition-colors"
              >
                <td class="p-2 text-dark/40 font-mono">{{ j + 1 }}</td>
                <td class="p-2 font-medium text-dark">{{ pred.class_id }}</td>
                <td class="p-2">
                  <span :class="pred.category === 'Organik' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'"
                    class="font-bold px-2 py-0.5 rounded text-xs">{{ pred.category }}</span>
                </td>
                <td class="p-2 text-right font-mono text-dark/60">{{ (pred.confidence * 100).toFixed(0) }}%</td>
                <td class="p-2 text-right font-mono text-dark/60">{{ pred.bbox[0].toFixed(0) }}</td>
                <td class="p-2 text-right font-mono text-dark/60">{{ pred.bbox[1].toFixed(0) }}</td>
                <td class="p-2 text-right font-mono text-dark/60">{{ pred.bbox[2].toFixed(0) }}</td>
                <td class="p-2 text-right font-mono text-dark/60">{{ pred.bbox[3].toFixed(0) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{ items: any[]; apiBase: string }>()
defineEmits<{ (e: "preview", url: string): void }>()
</script>
