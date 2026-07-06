export const useVisualization = () => {
  const config = useRuntimeConfig()
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function triggerVisualization(process?: string): Promise<any> {
    loading.value = true
    error.value = null

    try {
      const params = process ? { process } : {}
      const res = await $fetch("/api/kaggle/visualization/run", {
        baseURL: config.public.apiBase,
        method: "POST",
        params,
      })
      return res
    } catch (e: any) {
      error.value = e.data?.detail || "Visualization generation failed"
      throw e
    } finally {
      loading.value = false
    }
  }

  async function getVisualizationStatus(): Promise<any> {
    try {
      return await $fetch("/api/kaggle/visualization/status", {
        baseURL: config.public.apiBase,
      })
    } catch (e: any) {
      error.value = e.data?.detail || "Failed to check visualization status"
      throw e
    }
  }

  return { triggerVisualization, getVisualizationStatus, loading, error }
}
