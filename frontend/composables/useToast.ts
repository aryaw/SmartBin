const toasts = ref<{ id: number; message: string; type: 'error' | 'success' }[]>([])
let nextId = 0

export function useToast() {
  function show(message: string, type: 'error' | 'success' = 'error') {
    const id = nextId++
    toasts.value.push({ id, message, type })
    setTimeout(() => {
      toasts.value = toasts.value.filter(t => t.id !== id)
    }, 8000)
  }

  function dismiss(id: number) {
    toasts.value = toasts.value.filter(t => t.id !== id)
  }

  return { toasts, show, dismiss }
}
