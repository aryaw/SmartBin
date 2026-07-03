<template>
  <div v-if="result" class="space-y-4">
    <!-- Annotated result -->
    <img v-if="result.file_type === 'image'" :src="imageUrl" class="w-full rounded-xl shadow" />
    <video v-else :src="imageUrl" controls class="w-full rounded-xl shadow" />

    <!-- Summary cards -->
    <div class="grid grid-cols-3 gap-3">
      <div class="bg-green-50 border border-green-200 rounded-xl p-3 text-center">
        <p class="text-2xl font-bold text-green-700">{{ result.summary.organik }}</p>
        <p class="text-xs text-green-600 font-medium">Organik</p>
      </div>
      <div class="bg-blue-50 border border-blue-200 rounded-xl p-3 text-center">
        <p class="text-2xl font-bold text-blue-700">{{ result.summary.non_organik }}</p>
        <p class="text-xs text-blue-600 font-medium">Non-Organik</p>
      </div>
      <div class="bg-gray-50 border border-gray-200 rounded-xl p-3 text-center">
        <p class="text-2xl font-bold text-gray-700">{{ result.summary.total }}</p>
        <p class="text-xs text-gray-600 font-medium">Total</p>
      </div>
    </div>

    <!-- Bounding box report -->
    <BoundingBoxReport :objects="result.detected_objects" :file-type="result.file_type"
      :frames-processed="result.frames_processed" />
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const props = defineProps<{ result: any }>()
const imageUrl = computed(() => `${config.public.apiBase}${props.result.result_url}`)
</script>
