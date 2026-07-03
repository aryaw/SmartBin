<template>
  <Teleport to="body">
    <div v-if="url" class="fixed inset-0 z-50 flex items-center justify-center p-6 bg-dark/60" @click="$emit('close')">
      <div class="relative max-w-5xl max-h-full select-none" @click.stop
        @wheel.prevent="onWheel"
        @mousedown.prevent="onDragStart"
        @mousemove.prevent="onDragMove"
        @mouseup.prevent="onDragEnd"
        @mouseleave.prevent="onDragEnd"
        @dblclick.prevent="resetZoom">
        <button class="absolute -top-3 -right-3 w-8 h-8 bg-white rounded-full shadow-md flex items-center justify-center hover:text-red-500 transition-colors z-20"
          @click.stop="$emit('close')">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
        <div class="absolute -top-3 left-0 flex gap-1 z-20">
          <button class="w-8 h-8 bg-white rounded-full shadow-md flex items-center justify-center hover:text-blue-600 transition-colors text-sm font-bold"
            @click.stop="zoomIn" title="Zoom in">+</button>
          <button class="w-8 h-8 bg-white rounded-full shadow-md flex items-center justify-center hover:text-blue-600 transition-colors text-sm font-bold"
            @click.stop="zoomOut" title="Zoom out">−</button>
          <button v-if="scale !== 1" class="w-8 h-8 bg-white rounded-full shadow-md flex items-center justify-center hover:text-blue-600 transition-colors text-xs font-bold"
            @click.stop="resetZoom" title="Reset zoom">↺</button>
        </div>
        <div class="overflow-hidden rounded-xl shadow-2xl border-4 border-white"
          :style="{ cursor: isDragging ? 'grabbing' : scale > 1 ? 'grab' : 'default' }">
          <img :src="url"
            :style="{
              transform: `translate(${panX}px, ${panY}px) scale(${scale})`,
              transition: isDragging ? 'none' : 'transform 0.15s ease-out',
              maxWidth: '80vw',
              maxHeight: '80vh',
              display: 'block',
            }"
            class="rounded-lg"
            draggable="false" />
        </div>
        <div class="absolute bottom-2 left-1/2 -translate-x-1/2 bg-black/50 text-white text-xs px-2 py-0.5 rounded-full z-20">
          {{ (scale * 100).toFixed(0) }}%
        </div>
        <p v-if="caption" class="text-white text-sm text-center mt-2 font-medium">{{ caption }}</p>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
defineProps<{ url: string | null; caption?: string }>()
defineEmits<{ (e: "close"): void }>()

import { ref } from 'vue'

const scale = ref(1)
const panX = ref(0)
const panY = ref(0)
const isDragging = ref(false)
const dragStart = ref({ x: 0, y: 0, panX: 0, panY: 0 })

const MIN_SCALE = 0.5
const MAX_SCALE = 10
const STEP = 0.25

function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v))
}

function zoomIn() {
  scale.value = clamp(scale.value + STEP, MIN_SCALE, MAX_SCALE)
}

function zoomOut() {
  scale.value = clamp(scale.value - STEP, MIN_SCALE, MAX_SCALE)
}

function resetZoom() {
  scale.value = 1
  panX.value = 0
  panY.value = 0
}

function onWheel(e: WheelEvent) {
  const delta = e.deltaY > 0 ? -STEP : STEP
  const newScale = clamp(scale.value + delta, MIN_SCALE, MAX_SCALE)
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
  const mx = e.clientX - rect.left
  const my = e.clientY - rect.top
  const ratio = newScale / scale.value
  panX.value = mx - ratio * (mx - panX.value)
  panY.value = my - ratio * (my - panY.value)
  scale.value = newScale
}

function onDragStart(e: MouseEvent) {
  if (scale.value <= 1) return
  isDragging.value = true
  dragStart.value = { x: e.clientX, y: e.clientY, panX: panX.value, panY: panY.value }
}

function onDragMove(e: MouseEvent) {
  if (!isDragging.value) return
  panX.value = dragStart.value.panX + (e.clientX - dragStart.value.x)
  panY.value = dragStart.value.panY + (e.clientY - dragStart.value.y)
}

function onDragEnd() {
  isDragging.value = false
}
</script>
