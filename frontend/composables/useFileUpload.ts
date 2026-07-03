const ALLOWED_IMAGES = ["image/jpeg", "image/png"]
const ALLOWED_VIDEOS = ["video/mp4", "video/avi", "video/quicktime"]
const MAX_SIZE = 200 * 1024 * 1024

export const useFileUpload = () => {
  function validate(file: File): string | null {
    if (![...ALLOWED_IMAGES, ...ALLOWED_VIDEOS].includes(file.type))
      return "Format file tidak didukung"
    if (file.size > MAX_SIZE)
      return "File terlalu besar (maks 200MB)"
    return null
  }

  function toFormData(file: File): FormData {
    const form = new FormData()
    form.append("file", file)
    return form
  }

  return { validate, toFormData }
}
