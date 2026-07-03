<template>
  <div class="min-h-screen bg-gray-50 flex">
    <aside class="w-64 bg-primary text-gray-300 flex flex-col flex-shrink-0 border-r border-gray-200">
      <div class="w-full h-20 bg-gray-900 border-b border-gray-700 flex px-5 items-center mb-4">
        <div class="flex items-center gap-3 pl-2">
          <div class="w-10 h-10 bg-secondary rounded-full flex items-center justify-center text-white">
            <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0"/>
            </svg>
          </div>
          <div>
            <h1 class="font-bold text-white text-lg">Smart Bin</h1>
            <p class="text-xs text-gray-400">Detection System</p>
          </div>
        </div>
      </div>

      <nav class="flex-1 px-4 space-y-1 overflow-y-auto">
        <p class="pl-4 text-xs font-semibold text-gray-400 mb-1 uppercase tracking-wide">Main</p>

        <NuxtLink v-for="item in navItems" :key="item.to"
          :to="item.to"
          class="w-full flex items-center h-10 pl-4 rounded-lg text-sm transition-colors text-gray-300 hover:text-white"
          :class="isActive(item.to) ? 'bg-gray-700/50 text-white' : 'hover:bg-gray-700/50'"
        >
          <span class="h-5 w-5 fill-current mr-3 flex-shrink-0 text-gray-300" v-html="item.icon"></span>
          <span>{{ item.label }}</span>
        </NuxtLink>

        <div class="mt-6 mb-2">
          <p class="pl-4 text-xs font-semibold text-gray-400 uppercase tracking-wide">Dataset</p>
        </div>

        <NuxtLink v-for="item in datasetItems" :key="item.to"
          :to="item.to"
          class="w-full flex items-center h-10 pl-4 rounded-lg text-sm transition-colors text-gray-300 hover:text-white"
          :class="isActive(item.to) ? 'bg-gray-700/50 text-white' : 'hover:bg-gray-700/50'"
        >
          <span class="h-5 w-5 fill-current mr-3 flex-shrink-0 text-gray-300" v-html="item.icon"></span>
          <span>{{ item.label }}</span>
        </NuxtLink>

        <div class="mt-6 mb-2">
          <p class="pl-4 text-xs font-semibold text-gray-400 uppercase tracking-wide">Annotations</p>
        </div>

        <NuxtLink v-for="item in annotationItems" :key="item.to"
          :to="item.to"
          class="w-full flex items-center h-10 pl-4 rounded-lg text-sm transition-colors text-gray-300 hover:text-white"
          :class="isActive(item.to) ? 'bg-gray-700/50 text-white' : 'hover:bg-gray-700/50'"
        >
          <span class="h-5 w-5 fill-current mr-3 flex-shrink-0 text-gray-300" v-html="item.icon"></span>
          <span>{{ item.label }}</span>
        </NuxtLink>
      </nav>

      <div class="p-4 border-t border-gray-700 text-xs text-gray-400 text-center">
        Smart Bin &copy; 2026
      </div>
    </aside>

    <div class="flex-1 min-h-screen">
      <header class="sticky top-0 z-40 w-full h-20 px-6 bg-white border-b border-gray-200 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <h2 class="font-bold text-lg text-gray-800">{{ pageTitle }}</h2>
        </div>
        <div class="flex items-center gap-2 text-xs text-gray-500">
          <span class="w-2 h-2 rounded-full bg-green-400"></span>
          <span>Online</span>
        </div>
      </header>

      <main class="p-6">
        <slot />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
const route = useRoute()

const navItems = [
  { to: "/dashboard", label: "Dashboard Report", icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" /></svg>` },
  { to: "/test", label: "Test Upload", icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>` },
]

const datasetItems = [
  { to: "/raw", label: "All Raw Data", icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>` },
  { to: "/train-data", label: "Train Data", icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" /></svg>` },
  { to: "/test-data", label: "Test Data", icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M18.75 10.5a3 3 0 100-6 3 3 0 000 6z" /></svg>` },
  { to: "/eval", label: "Evaluation Metrics", icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z" /></svg>` },
]

const annotationItems = [
  { to: "/coco-annotation", label: "COCO Annotation Result", icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>` },
  { to: "/yolo-annotation", label: "YOLO BoundingBox Result", icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" /></svg>` },
  { to: "/yolo-segmentation", label: "YOLO Segmentation Result", icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M4.26 10.147a60.438 60.438 0 0 0-.491 6.347A48.62 48.62 0 0 1 12 20.904a48.62 48.62 0 0 1 8.232-4.41 60.46 60.46 0 0 0-.491-6.347m-15.482 0a50.636 50.636 0 0 0-2.658-.813A59.906 59.906 0 0 1 12 3.493a59.903 59.903 0 0 1 10.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.717 50.717 0 0 1 12 13.489a50.702 50.702 0 0 1 7.74-3.342" /></svg>` },
  { to: "/val-result", label: "Validation Inference Result", icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>` },
  { to: "/test-result", label: "Test Inference Result", icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /></svg>` },
]

function isActive(path: string) {
  return route.path === path
}

const pageTitle = computed(() => {
  const map: Record<string, string> = {
    "/": "Home",
    "/dashboard": "Dashboard Report",
    "/test": "Test Upload",
    "/raw": "All Raw Data",
    "/test-data": "Test Data",
    "/train-data": "Train Data",
    "/eval": "Evaluation Metrics",
    "/coco-annotation": "COCO Annotation Result",
    "/yolo-annotation": "YOLO BoundingBox Result",
    "/yolo-segmentation": "YOLO Segmentation Result",
    "/val-result": "Validation Inference Result",
    "/test-result": "Test Inference Result",
  }
  return map[route.path] || "SmartBin"
})
</script>
