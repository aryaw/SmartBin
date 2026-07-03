<template>
  <div v-if="visible" class="fixed inset-0 bg-dark/30 flex items-center justify-center z-50">
    <div class="bg-primary rounded-xl p-6 shadow-lg text-center">
      <div class="animate-spin h-8 w-8 border-4 border-tertiary border-t-transparent rounded-full mx-auto mb-4"></div>
      <p class="text-gray-300">Memproses...</p>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{ visible: boolean }>()
</script>
