<template>
  <div class="flex items-center justify-center h-64">
    <p class="text-dark/50">Redirecting to Dataset...</p>
  </div>
</template>
<script setup lang="ts">
await navigateTo("/raw/dataset", { replace: true })
</script>
