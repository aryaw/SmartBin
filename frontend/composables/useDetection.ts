export const useDetection = () => {
  const config = useRuntimeConfig()
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function detect(file: File): Promise<any> {
    loading.value = true
    error.value = null

    const form = new FormData()
    form.append("file", file)

    try {
      const res = await $fetch("/api/detect", {
        baseURL: config.public.apiBase,
        method: "POST",
        body: form,
      })
      return res
    } catch (e: any) {
      error.value = e.data?.detail || "Deteksi gagal"
      throw e
    } finally {
      loading.value = false
    }
  }

  return { detect, loading, error }
}
