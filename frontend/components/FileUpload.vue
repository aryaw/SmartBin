<template>
  <div
    class="border-2 border-dashed border-secondary rounded-xl p-12 text-center cursor-pointer hover:border-tertiary transition"
    @drop="onDrop"
    @dragover.prevent
    @click="$emit('select')"
  >
    <p class="text-dark">Drag & drop file di sini<br>atau klik untuk browse</p>
  </div>
</template>

<script setup lang="ts">
const emit = defineEmits<{ select: []; drop: [File] }>()

function onDrop(e: DragEvent) {
  if (e.dataTransfer?.files[0]) emit("drop", e.dataTransfer.files[0])
}
</script>
