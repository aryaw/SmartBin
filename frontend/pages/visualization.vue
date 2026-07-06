<template>
  <div class="space-y-6 max-w-6xl">
    <div class="bg-gradient-to-r from-tertiary to-secondary rounded-xl p-6 text-white shadow-sm">
      <h1 class="text-2xl font-bold">Pipeline Visualization</h1>
      <p class="text-white/80 text-sm mt-1">Step-by-step visualizations for all 6 pipeline processes</p>
    </div>

    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex items-center justify-between">
      <div>
        <h2 class="font-bold text-dark text-sm uppercase tracking-wide">Regenerate All</h2>
        <p class="text-xs text-dark/50">Generate all 54 visualization folders (pseudo_mask, stratified_split, augmentation, backbone, backend, frontend)</p>
      </div>
      <button :disabled="generatingAll"
        class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm transition-all flex items-center gap-2 disabled:opacity-50"
        @click="regenerateAll">
        <svg v-if="generatingAll" class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
        {{ generatingAll ? 'Generating...' : 'Regenerate All' }}
      </button>
    </div>

    <div v-if="progressMessage" class="bg-blue-50 border border-blue-200 rounded-xl p-4 text-sm text-blue-700 flex items-center gap-3">
      <svg class="w-5 h-5 flex-shrink-0 animate-spin" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
      <span>{{ progressMessage }}</span>
    </div>

    <div v-if="successMessage" class="bg-green-50 border border-green-200 rounded-xl p-4 text-sm text-green-700 flex items-center gap-3">
      <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <span>{{ successMessage }}</span>
    </div>

    <div v-for="group in processGroups" :key="group.id" class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
      <div class="bg-gray-50 border-b border-gray-100 px-5 py-3 flex items-center justify-between">
        <div>
          <h3 class="font-bold text-dark text-sm uppercase tracking-wide">{{ group.label }}</h3>
          <p class="text-xs text-dark/40 mt-0.5">{{ group.folderCount }} step folders &middot; {{ group.images.length }} images</p>
        </div>
        <button :disabled="generatingProcess === group.id"
          class="bg-tertiary hover:bg-blue-700 text-white text-xs font-bold px-4 py-2 rounded-lg shadow-sm transition-all flex items-center gap-1.5 disabled:opacity-50"
          @click="regenerateProcess(group.id)">
          <svg v-if="generatingProcess === group.id" class="animate-spin h-3.5 w-3.5" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <svg v-else class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          {{ generatingProcess === group.id ? 'Generating...' : 'Regenerate' }}
        </button>
      </div>
      <div class="p-5">
        <VisualizationGallery :images="group.images" :process-name="group.id" @preview="openPreview" />
      </div>
    </div>

    <LoadingOverlay :visible="generatingAll || !!generatingProcess" />
    <ZoomModal :url="previewUrl" :caption="previewCaption" @close="previewUrl = null" />
  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase
const { show: showToast } = useToast()
const { triggerVisualization, loading } = useVisualization()

const generatingAll = ref(false)
const generatingProcess = ref<string | null>(null)
const progressMessage = ref<string | null>(null)
const successMessage = ref<string | null>(null)
const previewUrl = ref<string | null>(null)
const previewCaption = ref("")

interface VizImage {
  step: number
  url: string
  title: string
  description?: string
}

interface ProcessGroup {
  id: string
  label: string
  folderCount: number
  images: VizImage[]
}

const processGroups = ref<ProcessGroup[]>([])

const processConfig: { id: string; label: string; steps: { folder: string; title: string; description: string; file: string }[] }[] = [
  {
    id: "pseudo_mask",
    label: "Pseudo-Polygon Mask Generation",
    steps: [
      { folder: "01_original_rgb", title: "Original RGB", description: "Original image loaded from disk", file: "*.jpg" },
      { folder: "02_grayscale", title: "Grayscale", description: "RGB to single channel luminance", file: "*.jpg" },
      { folder: "03_gaussian_blur_5x5", title: "Gaussian Blur 5x5", description: "Smooth noise with kernel size 5", file: "*.jpg" },
      { folder: "04_otsu_threshold", title: "Otsu Thresholding", description: "Auto-calculate threshold, binarize", file: "*.jpg" },
      { folder: "05_mean_check_diagram", title: "Mean Check", description: "Check if average pixel > 127", file: "*.jpg" },
      { folder: "06a_invert_mask", title: "Invert Mask", description: "Invert binary if mean > 127", file: "*.jpg" },
      { folder: "06b_morphological_close", title: "Morphological Close", description: "Close holes with 5x5 kernel", file: "*.jpg" },
      { folder: "07_find_contours", title: "Find Contours", description: "Largest contour with green outline", file: "*.jpg" },
      { folder: "08_area_check_diagram", title: "Area Check", description: "Check contour area >= 20%", file: "*.jpg" },
      { folder: "09a_edge_success_mask", title: "Edge Success", description: "Edge detection result with filled contour", file: "*.jpg" },
      { folder: "09b_fallback_ellipse", title: "Fallback Ellipse", description: "Elliptical polygon fallback", file: "*.jpg" },
      { folder: "09c_fallback_rounded_rect", title: "Fallback Rounded Rect", description: "Rounded rectangle fallback", file: "*.jpg" },
      { folder: "10_approximate_polygon", title: "Approx Polygon", description: "Polygon vertices and edges in red", file: "*.jpg" },
      { folder: "11_normalized_coordinates", title: "Normalized Coordinates", description: "Coordinates normalized to [0,1]", file: "*.jpg" },
      { folder: "12_yolo_seg_label", title: "YOLO-seg Label", description: "Final label format with class and coordinates", file: "*.jpg" },
      { folder: "summary_collage_4x4", title: "Summary Collage", description: "4x4 grid collage of all 16 steps", file: "collage.jpg" },
    ],
  },
  {
    id: "stratified_split",
    label: "Stratified Split 70/15/15",
    steps: [
      { folder: "01_split_bar_chart", title: "Split Bar Chart", description: "Horizontal bar chart with counts", file: "chart.jpg" },
      { folder: "02_split_pie_chart", title: "Split Pie Chart", description: "Pie chart with 3 segments", file: "chart.jpg" },
      { folder: "03_class_distribution_chart", title: "Class Distribution", description: "Grouped bar chart per split", file: "chart.jpg" },
      { folder: "04_split_table", title: "Split Table", description: "Data table with counts and percentages", file: "table.jpg" },
    ],
  },
  {
    id: "augmentation",
    label: "Online Augmentation",
    steps: [
      { folder: "01_original", title: "Original Image", description: "Base image before augmentation", file: "*.jpg" },
      { folder: "02_mosaic", title: "Mosaic 1.0", description: "4 images combined in 2x2 grid", file: "*.jpg" },
      { folder: "03_mixup", title: "Mixup 0.2", description: "2 images blended with alpha=0.2", file: "*.jpg" },
      { folder: "04_copy_paste", title: "Copy-Paste 0.15", description: "Object pasted onto another image", file: "*.jpg" },
      { folder: "05a_hsv_hue", title: "HSV Hue Shift", description: "Hue shift 0.05", file: "*.jpg" },
      { folder: "05b_hsv_saturation", title: "HSV Saturation Boost", description: "Saturation boost 0.8", file: "*.jpg" },
      { folder: "05c_hsv_value", title: "HSV Value Shift", description: "Value/brightness shift 0.5", file: "*.jpg" },
      { folder: "06a_rotate", title: "Rotate ±15°", description: "Rotation by 15 degrees", file: "*.jpg" },
      { folder: "06b_scale", title: "Scale ±50%", description: "Scale transformation", file: "*.jpg" },
      { folder: "06c_shear", title: "Shear 5°", description: "Shear distortion", file: "*.jpg" },
      { folder: "07_flip_horizontal", title: "Flip Horizontal", description: "Left-right mirror", file: "*.jpg" },
      { folder: "08_augmentation_pipeline_collage", title: "Pipeline Collage", description: "Split panel with original + augmentations", file: "*.jpg" },
      { folder: "09_summary_grid_3x3", title: "3x3 Summary Grid", description: "Original + 8 key augmentations grid", file: "grid.jpg" },
    ],
  },
  {
    id: "backbone",
    label: "Backbone: CSPDarknet Architecture",
    steps: [
      { folder: "01_overview_flowchart", title: "Backbone Overview", description: "Full backbone flowchart from input to SPP", file: "diagram.jpg" },
      { folder: "02_csp_stage_detail", title: "CSP Stage Detail", description: "CSP stage internals with split and merge", file: "diagram.jpg" },
      { folder: "03_resolution_progression", title: "Resolution Progression", description: "Resolution comparison with grid overlay", file: "diagram.jpg" },
      { folder: "04_feature_map_evolution", title: "Feature Map Evolution", description: "Conceptual feature maps per stage", file: "diagram.jpg" },
      { folder: "05_csp_vs_standard", title: "CSP vs Standard", description: "Comparison with FLOPs savings", file: "diagram.jpg" },
      { folder: "06_spp_multi_scale", title: "SPP Multi-Scale", description: "SPP layer with 3 pooling sizes", file: "diagram.jpg" },
      { folder: "07_full_flow_annotated", title: "Full Flow Annotated", description: "Complete backbone flow with annotations", file: "diagram.jpg" },
    ],
  },
  {
    id: "backend",
    label: "Backend API Flow",
    steps: [
      { folder: "01_api_detection_flow", title: "API Detection Flow", description: "Full request-response flow", file: "diagram.jpg" },
      { folder: "02_inference_pipeline", title: "Inference Pipeline", description: "YOLO inference internals call chain", file: "diagram.jpg" },
      { folder: "03_training_pipeline_api", title: "Training Pipeline API", description: "Full pipeline flow via API", file: "diagram.jpg" },
      { folder: "04_backend_routes_overview", title: "Backend Routes Overview", description: "All route groups overview", file: "diagram.jpg" },
      { folder: "05_detect_request_response", title: "Detect Request/Response", description: "HTTP request to JSON response schema", file: "diagram.jpg" },
      { folder: "06_model_lifecycle", title: "Model Lifecycle", description: "Model loading and inference lifecycle", file: "diagram.jpg" },
      { folder: "07_error_handling_flow", title: "Error Handling Flow", description: "Error paths and exception handling", file: "diagram.jpg" },
    ],
  },
  {
    id: "frontend",
    label: "Frontend UI Flow",
    steps: [
      { folder: "01_sidebar_navigation", title: "Sidebar Navigation", description: "Sidebar structure with active state", file: "wireframe.jpg" },
      { folder: "02_detection_ux_flow", title: "Detection UX Flow", description: "User journey from upload to result", file: "wireframe.jpg" },
      { folder: "03_component_tree", title: "Component Tree", description: "Component hierarchy and relationships", file: "tree.jpg" },
      { folder: "04_api_integration_flow", title: "API Integration Flow", description: "Frontend-backend interaction sequence", file: "sequence.jpg" },
      { folder: "05_batch_detect_flow", title: "Batch Detect Flow", description: "Multi-file upload and detection flow", file: "flowchart.jpg" },
      { folder: "06_responsive_layout", title: "Responsive Layout", description: "Page layout breakdown with slots", file: "wireframe.jpg" },
      { folder: "07_state_management", title: "State Management", description: "Reactive states per page diagram", file: "diagram.jpg" },
    ],
  },
]

const vizBase = computed(() => `${apiBase}/visualization`)

onMounted(() => {
  processGroups.value = processConfig.map((proc) => ({
    id: proc.id,
    label: proc.label,
    folderCount: proc.steps.length,
    images: proc.steps.map((step, i) => ({
      step: i + 1,
      url: `${vizBase.value}/${proc.id}/${step.folder}/${step.file}`,
      title: step.title,
      description: step.description,
    })),
  }))
})

async function regenerateAll() {
  generatingAll.value = true
  progressMessage.value = "Regenerating all visualization processes..."
  successMessage.value = null
  try {
    const res = await triggerVisualization()
    progressMessage.value = null
    successMessage.value = res.message || "All visualizations regenerated successfully."
  } catch (e: any) {
    progressMessage.value = null
    showToast(e.data?.detail || e.message || "Failed to regenerate visualizations")
  } finally {
    generatingAll.value = false
  }
}

async function regenerateProcess(processId: string) {
  generatingProcess.value = processId
  progressMessage.value = `Regenerating ${processId} visualizations...`
  successMessage.value = null
  try {
    const res = await triggerVisualization(processId)
    progressMessage.value = null
    successMessage.value = res.message || `${processId} visualizations regenerated successfully.`
  } catch (e: any) {
    progressMessage.value = null
    showToast(e.data?.detail || e.message || `Failed to regenerate ${processId}`)
  } finally {
    generatingProcess.value = null
  }
}

function openPreview(url: string) {
  previewUrl.value = url
  const parts = url.split("/")
  previewCaption.value = parts[parts.length - 2] || ""
}
</script>
