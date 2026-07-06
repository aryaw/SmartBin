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
          <span class="h-5 w-5 fill-current mr-3 flex-shrink-0" :class="isActive(item.to) ? 'text-white' : 'text-gray-300'" v-html="item.icon"></span>
          <span>{{ item.label }}</span>
        </NuxtLink>


        <div class="mt-6 mb-2">
          <p class="pl-4 text-xs font-semibold text-gray-400 uppercase tracking-wide">Report</p>
        </div>

        <NuxtLink v-for="item in reportItems" :key="item.to"
          :to="item.to"
          class="w-full flex items-center h-8 pl-4 rounded-lg text-xs transition-colors text-gray-300 hover:text-white"
          :class="isActive(item.to) ? 'bg-gray-700/50 text-white' : 'hover:bg-gray-700/50'"
        >
          <span class="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold mr-2 flex-shrink-0"
            :class="isActive(item.to) ? 'bg-tertiary text-white' : 'bg-gray-600 text-gray-300'">{{ item.num }}</span>
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
  <Toast />
</template>

<script setup lang="ts">
const route = useRoute()

const navItems = [
  { to: "/dashboard", label: "Dashboard", icon: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" /></svg>` },
]

const reportItems = [
  { to: "/train-eval", label: "Training Eval", num: "1" },
  { to: "/val-result", label: "Validation", num: "2" },
  { to: "/test-result", label: "Test Results", num: "3" },
  { to: "/inference-export", label: "Inference", num: "4" },
  { to: "/visualization", label: "Visualization", num: "5" },
]

function isActive(path: string) {
  return route.path === path
}

const pageTitle = computed(() => {
  const map: Record<string, string> = {
    "/": "Home",
    "/dashboard": "Dashboard",
  }
  return map[route.path] || "SmartBin"
})
</script>
