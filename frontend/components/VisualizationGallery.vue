<template>
  <div>
    <div v-if="!images.length" class="bg-white rounded-xl p-10 text-center border border-gray-100">
      <div class="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
        <svg class="w-6 h-6 text-dark/30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      </div>
      <p class="text-dark/50 text-sm">No visualization images available for this process.</p>
      <p class="text-dark/40 text-xs mt-1">Click "Regenerate" to generate visualization images.</p>
    </div>

    <div v-else class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
      <div v-for="(img, idx) in images" :key="idx"
        class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden hover:shadow-md transition-shadow group cursor-pointer"
        @click="$emit('preview', img.url)"
      >
        <div class="aspect-video overflow-hidden bg-gray-50">
          <img :src="img.url" :alt="img.title"
            class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            loading="lazy"
          />
        </div>
        <div class="p-2.5">
          <p class="text-xs font-medium text-dark truncate">{{ img.title }}</p>
          <p v-if="img.description" class="text-xs text-dark/40 mt-0.5 line-clamp-2">{{ img.description }}</p>
          <p class="text-xs text-dark/30 mt-1">Step {{ img.step }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
interface VizImage {
  step: number
  url: string
  title: string
  description?: string
}

defineProps<{
  processName?: string
  images: VizImage[]
}>()

defineEmits<{
  (e: "preview", url: string): void
}>()
</script>
