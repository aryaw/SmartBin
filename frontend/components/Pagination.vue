<template>
  <div v-if="totalPages > 1" class="flex items-center justify-center gap-1 pt-4">
    <button
      class="px-3 py-1.5 text-sm rounded-lg border border-gray-200 bg-white hover:bg-gray-100 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
      :disabled="page <= 1"
      @click="go(page - 1)"
    >Prev</button>

    <button
      v-for="p in visiblePages" :key="p"
      class="min-w-[32px] px-2 py-1.5 text-sm rounded-lg transition-colors font-medium"
      :class="p === page
        ? 'bg-tertiary text-white shadow-sm'
        : 'border border-tertiary/30 bg-white hover:bg-secondary/20 text-fb-text'"
      @click="go(p)"
    >{{ p }}</button>

    <button
      class="px-3 py-1.5 text-sm rounded-lg border border-gray-200 bg-white hover:bg-gray-100 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
      :disabled="page >= totalPages"
      @click="go(page + 1)"
    >Next</button>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  page: number
  total: number
  per: number
}>()

const emit = defineEmits<{ (e: "update:page", v: number): void }>()

const totalPages = computed(() => {
  const t = Math.ceil(props.total / props.per)
  return isNaN(t) ? 1 : Math.max(1, t)
})

const visiblePages = computed(() => {
  const p = props.page
  const t = totalPages.value
  const pages: number[] = []
  let start = Math.max(1, p - 2)
  let end = Math.min(t, p + 2)
  if (end - start < 4) {
    if (start === 1) end = Math.min(t, start + 4)
    else start = Math.max(1, end - 4)
  }
  for (let i = start; i <= end; i++) pages.push(i)
  return pages
})

function go(n: number) {
  if (n >= 1 && n <= totalPages.value) emit("update:page", n)
}
</script>
