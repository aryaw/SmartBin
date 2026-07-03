<template>
  <div class="space-y-6">

    <!-- Stats -->
    <div v-if="tabKey === 'raw'" class="grid grid-cols-4 gap-4">
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
        <p class="text-3xl font-bold text-dark">{{ data?.stats?.raw || 0 }}</p>
        <p class="text-xs mt-1 text-dark/50 font-medium">Total Raw Data</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
        <p class="text-3xl font-bold text-blue-700">{{ data?.stats?.train || 0 }}</p>
        <p class="text-xs mt-1 text-blue-600 font-medium">Total Train Data</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
        <p class="text-3xl font-bold text-green-700">{{ data?.stats?.val || 0 }}</p>
        <p class="text-xs mt-1 text-green-600 font-medium">Total Inference Data</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
        <p class="text-3xl font-bold text-orange-600">{{ data?.stats?.test || 0 }}</p>
        <p class="text-xs mt-1 text-orange-600 font-medium">Total Test Data</p>
      </div>
    </div>
    <div v-if="tabKey === 'coco'" class="grid grid-cols-4 gap-4">
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
        <p class="text-3xl font-bold text-green-700">{{ data?.stats?.coco || 0 }}</p>
        <p class="text-xs mt-1 text-green-600 font-medium">Train Annotated</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
        <p class="text-3xl font-bold text-blue-700">{{ data?.stats?.coco_val || 0 }}</p>
        <p class="text-xs mt-1 text-blue-600 font-medium">Inference Annotated</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
        <p class="text-3xl font-bold text-purple-700">{{ totalCocoObjects }}</p>
        <p class="text-xs mt-1 text-purple-600 font-medium">Total Objects</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
        <p class="text-3xl font-bold text-dark">{{ data?.stats?.coco_total || 0 }}</p>
        <p class="text-xs mt-1 text-dark/50 font-medium">Train Images</p>
      </div>
    </div>
    <div v-if="tabKey === 'yolo'" class="grid grid-cols-4 gap-4">
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
        <p class="text-3xl font-bold text-dark">{{ data?.stats?.raw || 0 }}</p>
        <p class="text-xs mt-1 text-dark/50 font-medium">Total Raw</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
        <p class="text-3xl font-bold text-blue-700">{{ data?.stats?.train || 0 }}</p>
        <p class="text-xs mt-1 text-blue-600 font-medium">Train</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
        <p class="text-3xl font-bold text-green-700">{{ data?.stats?.val || 0 }}</p>
        <p class="text-xs mt-1 text-green-600 font-medium">Inference</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
        <p class="text-3xl font-bold text-orange-600">{{ data?.stats?.test || 0 }}</p>
        <p class="text-xs mt-1 text-orange-600 font-medium">Test</p>
      </div>
    </div>

    <!-- Train / Val Buttons -->
    <div v-if="tabKey === 'coco'" class="flex justify-end">
      <button
        class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm hover:shadow transition-all flex items-center gap-2 disabled:opacity-50"
        :disabled="training" @click="trainCoco()"
      >
        <svg v-if="training && trainingType === 'coco'" class="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
        </svg>
        {{ training && trainingType === 'coco' ? 'Training...' : 'Train COCO Pipeline' }}
      </button>
    </div>

    <div v-if="tabKey === 'yolo' && subKey === 'train_convert'" class="flex justify-end">
      <button
        class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm hover:shadow transition-all flex items-center gap-2 disabled:opacity-50"
        :disabled="training" @click="trainCoco()"
      >
        <svg v-if="training && trainingType === 'coco'" class="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
        </svg>
        {{ training && trainingType === 'coco' ? 'Training...' : 'Train YOLO' }}
      </button>
    </div>

    <div v-if="tabKey === 'yolo' && subKey === 'train_real'" class="flex justify-end">
      <button
        class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm hover:shadow transition-all flex items-center gap-2 disabled:opacity-50"
        :disabled="training" @click="trainYolo()"
      >
        <svg v-if="training && trainingType === 'yolo'" class="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
        </svg>
        {{ training && trainingType === 'yolo' ? 'Inferring...' : 'Run YOLO Inference' }}
      </button>
    </div>

    <div v-if="tabKey === 'yolo' && subKey === 'val_real'" class="flex justify-end">
      <button
        class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm hover:shadow transition-all flex items-center gap-2 disabled:opacity-50"
        :disabled="training" @click="trainYoloVal()"
      >
        <svg v-if="training && trainingType === 'yolo_val'" class="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
        </svg>
        {{ training && trainingType === 'yolo_val' ? 'Inferring...' : 'Run YOLO Inference' }}
      </button>
    </div>

    <!-- ==================== RAW ==================== -->
    <div v-if="tabKey === 'raw'">
      <div v-if="!paginatedRaw.length" class="bg-white rounded-xl p-10 text-center border border-gray-100">
        <div class="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
          <svg class="w-6 h-6 text-dark/30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>
        <p class="text-dark/50 text-sm">No raw images in dataset/raw/</p>
      </div>
      <template v-else>
        <div class="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 gap-3">
          <div v-for="img in paginatedRaw" :key="img.filename"
            class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden hover:shadow-md transition-shadow group cursor-pointer"
            @click="previewUrl = `${apiBase}${img.url}`"
          >
            <div class="aspect-square overflow-hidden bg-gray-50">
              <img :src="`${apiBase}${img.url}`" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
            </div>
            <div class="p-2">
              <p class="text-xs font-medium text-dark truncate">{{ img.filename }}</p>
              <p class="text-xs text-dark/40">{{ img.size_kb }} KB</p>
            </div>
          </div>
        </div>
        <Pagination :page="page" :total="data.raw.length" :per="perPage" @update:page="page = $event" />
      </template>
    </div>

    <!-- ==================== COCO ==================== -->
    <div v-if="tabKey === 'coco'">
      <div v-if="!paginatedCoco.length" class="bg-white rounded-xl p-10 text-center border border-gray-100">
        <div class="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
          <svg class="w-6 h-6 text-dark/30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
          </svg>
        </div>
        <p class="text-dark/50 text-sm">No COCO annotations. Click "Train COCO Pipeline" above.</p>
      </div>
      <template v-else>
        <div class="space-y-4">
          <div v-for="img in paginatedCoco" :key="img.filename"
            class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden"
          >
            <div class="flex flex-col lg:flex-row">
              <div class="lg:w-72 p-3">
                <img :src="`${apiBase}${img.viz_url || img.image_url}`"
                  class="w-full aspect-square object-cover rounded-lg border-2 border-green-200 cursor-pointer hover:opacity-90 transition-opacity"
                  @click="previewUrl = `${apiBase}${img.viz_url || img.image_url}`" />
                <p class="text-xs font-medium text-dark mt-2 truncate">{{ img.filename }}</p>
                <p class="text-xs text-dark/40">{{ img.annotations.length }} object(s)</p>
              </div>
              <div class="flex-1 overflow-x-auto p-3">
                <table class="w-full text-xs">
                  <thead>
                    <tr class="text-dark/50 uppercase tracking-wide border-b border-gray-100">
                      <th class="p-2 text-left font-medium">#</th>
                      <th class="p-2 text-left font-medium">Class</th>
                      <th class="p-2 text-left font-medium">Category</th>
                      <th class="p-2 text-right font-medium">X</th>
                      <th class="p-2 text-right font-medium">Y</th>
                      <th class="p-2 text-right font-medium">W</th>
                      <th class="p-2 text-right font-medium">H</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(ann, j) in img.annotations" :key="j"
                      class="border-b border-gray-50 hover:bg-gray-50 transition-colors"
                    >
                      <td class="p-2 text-dark/40 font-mono">{{ j + 1 }}</td>
                      <td class="p-2 font-medium text-dark">{{ ann.class_id }}</td>
                      <td class="p-2">
                        <span :class="ann.category === 'Organik' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'"
                          class="font-bold px-2 py-0.5 rounded text-xs">{{ ann.category }}</span>
                      </td>
                      <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[0].toFixed(0) }}</td>
                      <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[1].toFixed(0) }}</td>
                      <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[2].toFixed(0) }}</td>
                      <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[3].toFixed(0) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
        <Pagination :page="page" :total="filteredCoco.length" :per="perPage" @update:page="page = $event" />
      </template>
    </div>

    <!-- ==================== YOLO ==================== -->
    <div v-if="tabKey === 'yolo'">

      <!-- Train Image (plain) -->
      <div v-if="subKey === 'train_img'">
        <div class="flex items-center justify-between mb-3">
          <h3 class="font-bold text-tertiary uppercase tracking-wide text-sm">Train Images (plain)</h3>
          <div class="flex items-center gap-2">
            <button
              class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm hover:shadow transition-all flex items-center gap-2 disabled:opacity-50"
              :disabled="training" @click="trainCoco()"
            >
              <svg v-if="training && trainingType === 'coco'" class="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              </svg>
              {{ training && trainingType === 'coco' ? 'Converting...' : 'Annotation YOLO: From COCO' }}
            </button>
            <button
              class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm hover:shadow transition-all flex items-center gap-2 disabled:opacity-50"
              :disabled="training" @click="trainYolo()"
            >
              <svg v-if="training && trainingType === 'yolo'" class="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              </svg>
              {{ training && trainingType === 'yolo' ? 'Annotating...' : 'Annotation YOLO: For Real' }}
            </button>
          </div>
        </div>
        <div v-if="!paginatedTrainPlain.length" class="bg-white rounded-xl p-8 text-center border border-gray-100">
          <p class="text-dark/40 text-sm">No train images.</p>
        </div>
        <template v-else>
          <div class="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 gap-3">
            <div v-for="img in paginatedTrainPlain" :key="img.filename"
              class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden hover:shadow-md transition-shadow group cursor-pointer"
              @click="previewUrl = `${apiBase}${img.image_url}`"
            >
              <div class="aspect-square overflow-hidden bg-gray-50">
                <img :src="`${apiBase}${img.image_url}`" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
              </div>
              <div class="p-2">
                <p class="text-xs font-medium text-dark truncate">{{ img.filename }}</p>
              </div>
            </div>
          </div>
          <Pagination :page="pageTrain" :total="trainPlain.length" :per="perPage" @update:page="pageTrain = $event" />
        </template>
      </div>

      <!-- Train Convert YOLO (COCO→YOLO) -->
      <div v-if="subKey === 'train_convert'">
        <h3 class="font-bold text-tertiary uppercase tracking-wide text-sm mb-3">Train Convert YOLO (COCO → YOLO)</h3>
        <div v-if="!paginatedTrainConvert.length" class="bg-white rounded-xl p-8 text-center border border-gray-100">
          <p class="text-dark/40 text-sm">No converted annotations. Click "Convert COCO → YOLO".</p>
        </div>
        <template v-else>
          <YoloGrid :items="paginatedTrainConvert" :api-base="apiBase" @preview="previewUrl = $event" />
          <Pagination :page="pageTrain" :total="filteredTrainConvert.length" :per="perPage" @update:page="pageTrain = $event" />
        </template>
      </div>

      <!-- Train Real YOLO (model inference) -->
      <div v-if="subKey === 'train_real'">
        <h3 class="font-bold text-tertiary uppercase tracking-wide text-sm mb-3">Train Real YOLO (model inference)</h3>
        <div v-if="!paginatedTrainReal.length" class="bg-white rounded-xl p-8 text-center border border-gray-100">
          <p class="text-dark/40 text-sm">No predictions. Run YOLO inference.</p>
        </div>
        <template v-else>
          <YoloGrid :items="paginatedTrainReal" :api-base="apiBase" @preview="previewUrl = $event" />
          <Pagination :page="pageTrain" :total="filteredTrainReal.length" :per="perPage" @update:page="pageTrain = $event" />
        </template>
      </div>

      <!-- Inference Image YOLO -->
      <div v-if="subKey === 'val_img'">
        <div class="flex items-center justify-between mb-3">
          <h3 class="font-bold text-tertiary uppercase tracking-wide text-sm">Inference Image YOLO</h3>
          <button
            class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm hover:shadow transition-all flex items-center gap-2 disabled:opacity-50"
            :disabled="training" @click="trainYoloVal()"
          >
            <svg v-if="training && trainingType === 'yolo_val'" class="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
            </svg>
            {{ training && trainingType === 'yolo_val' ? 'Inferring...' : 'Run Inference' }}
          </button>
        </div>
        <div v-if="!paginatedValPlain.length" class="bg-white rounded-xl p-8 text-center border border-gray-100">
          <p class="text-dark/40 text-sm">No val images.</p>
        </div>
        <template v-else>
          <div class="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 gap-3">
            <div v-for="img in paginatedValPlain" :key="img.filename"
              class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden hover:shadow-md transition-shadow group cursor-pointer"
              @click="previewUrl = `${apiBase}${img.image_url}`"
            >
              <div class="aspect-square overflow-hidden bg-gray-50">
                <img :src="`${apiBase}${img.image_url}`" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
              </div>
              <div class="p-2">
                <p class="text-xs font-medium text-dark truncate">{{ img.filename }}</p>
              </div>
            </div>
          </div>
          <Pagination :page="pageVal" :total="valPlain.length" :per="perPage" @update:page="pageVal = $event" />
        </template>
      </div>

      <!-- Inference Result YOLO (detail + bounding box) -->
      <div v-if="subKey === 'val_real'">
        <h3 class="font-bold text-tertiary uppercase tracking-wide text-sm mb-3">Inference Result YOLO (detail + bounding box)</h3>
        <div v-if="!paginatedValResult.length" class="bg-white rounded-xl p-8 text-center border border-gray-100">
          <p class="text-dark/40 text-sm">No val results. Run Val on Val Image YOLO page.</p>
        </div>
        <template v-else>
          <div class="space-y-4 mb-4">
            <div v-for="img in paginatedValResult" :key="img.filename"
              class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden"
            >
              <div class="flex flex-col lg:flex-row">
                <div class="lg:w-72 p-3">
                  <img :src="`${apiBase}${img.viz_url}`"
                    class="w-full aspect-square object-cover rounded-lg border-2 border-blue-200 cursor-pointer hover:opacity-90 transition-opacity"
                    @click="previewUrl = `${apiBase}${img.viz_url}`" />
                  <p class="text-xs font-medium text-dark mt-2 truncate">{{ img.filename }}</p>
                  <p class="text-xs text-dark/40">{{ img.predictions.length }} object(s)</p>
                </div>
                <div class="flex-1 overflow-x-auto p-3">
                  <table class="w-full text-xs">
                    <thead>
                      <tr class="text-dark/50 uppercase tracking-wide border-b border-gray-100">
                        <th class="p-2 text-left font-medium">#</th>
                        <th class="p-2 text-left font-medium">Class</th>
                        <th class="p-2 text-left font-medium">Category</th>
                        <th class="p-2 text-right font-medium">Conf</th>
                        <th class="p-2 text-right font-medium">X1</th>
                        <th class="p-2 text-right font-medium">Y1</th>
                        <th class="p-2 text-right font-medium">X2</th>
                        <th class="p-2 text-right font-medium">Y2</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(ann, j) in img.predictions" :key="j"
                        class="border-b border-gray-50 hover:bg-gray-50 transition-colors"
                      >
                        <td class="p-2 text-dark/40 font-mono">{{ j + 1 }}</td>
                        <td class="p-2 font-medium text-dark">{{ ann.class_id }}</td>
                        <td class="p-2">
                          <span :class="ann.category === 'Organik' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'"
                            class="font-bold px-2 py-0.5 rounded text-xs">{{ ann.category }}</span>
                        </td>
                        <td class="p-2 text-right font-mono text-dark/60">{{ (ann.confidence * 100).toFixed(0) }}%</td>
                        <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[0].toFixed(0) }}</td>
                        <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[1].toFixed(0) }}</td>
                        <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[2].toFixed(0) }}</td>
                        <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[3].toFixed(0) }}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
          <Pagination :page="pageVal" :total="filteredValResult.length" :per="perPage" @update:page="pageVal = $event" />
        </template>
      </div>

      <!-- Annotation Result Convert (report) -->
      <div v-if="subKey === 'convert_result'">
        <h3 class="font-bold text-tertiary uppercase tracking-wide text-sm mb-3">Annotation Result Convert (COCO → YOLO)</h3>
        <div v-if="!paginatedConvertResult.length" class="bg-white rounded-xl p-8 text-center border border-gray-100">
          <p class="text-dark/40 text-sm">No annotation results. Run "Annotation YOLO: From COCO" on Train Image page.</p>
        </div>
        <template v-else>
          <div class="space-y-4 mb-4">
            <div v-for="img in paginatedConvertResult" :key="img.filename"
              class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden"
            >
              <div class="flex flex-col lg:flex-row">
                <div class="lg:w-72 p-3">
                  <img :src="`${apiBase}${img.viz_url}`"
                    class="w-full aspect-square object-cover rounded-lg border-2 border-green-200 cursor-pointer hover:opacity-90 transition-opacity"
                    @click="previewUrl = `${apiBase}${img.viz_url}`" />
                  <p class="text-xs font-medium text-dark mt-2 truncate">{{ img.filename }}</p>
                  <p class="text-xs text-dark/40">{{ img.predictions.length }} object(s)</p>
                </div>
                <div class="flex-1 overflow-x-auto p-3">
                  <table class="w-full text-xs">
                    <thead>
                      <tr class="text-dark/50 uppercase tracking-wide border-b border-gray-100">
                        <th class="p-2 text-left font-medium">#</th>
                        <th class="p-2 text-left font-medium">Class</th>
                        <th class="p-2 text-left font-medium">Category</th>
                        <th class="p-2 text-right font-medium">X1</th>
                        <th class="p-2 text-right font-medium">Y1</th>
                        <th class="p-2 text-right font-medium">X2</th>
                        <th class="p-2 text-right font-medium">Y2</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(ann, j) in img.predictions" :key="j"
                        class="border-b border-gray-50 hover:bg-gray-50 transition-colors"
                      >
                        <td class="p-2 text-dark/40 font-mono">{{ j + 1 }}</td>
                        <td class="p-2 font-medium text-dark">{{ ann.class_id }}</td>
                        <td class="p-2">
                          <span :class="ann.category === 'Organik' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'"
                            class="font-bold px-2 py-0.5 rounded text-xs">{{ ann.category }}</span>
                        </td>
                        <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[0].toFixed(0) }}</td>
                        <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[1].toFixed(0) }}</td>
                        <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[2].toFixed(0) }}</td>
                        <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[3].toFixed(0) }}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
          <Pagination :page="pageResult" :total="filteredConvertResult.length" :per="perPage" @update:page="pageResult = $event" />
        </template>
      </div>

      <!-- Annotation Result YOLO (report) -->
      <div v-if="subKey === 'real_result'">
        <h3 class="font-bold text-tertiary uppercase tracking-wide text-sm mb-3">Annotation Result YOLO (model inference)</h3>
        <div v-if="!paginatedRealResult.length" class="bg-white rounded-xl p-8 text-center border border-gray-100">
          <p class="text-dark/40 text-sm">No annotation results. Run "Annotation YOLO: For Real" on Train Image page.</p>
        </div>
        <template v-else>
          <div class="space-y-4 mb-4">
            <div v-for="img in paginatedRealResult" :key="img.filename"
              class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden"
            >
              <div class="flex flex-col lg:flex-row">
                <div class="lg:w-72 p-3">
                  <img :src="`${apiBase}${img.viz_url}`"
                    class="w-full aspect-square object-cover rounded-lg border-2 border-blue-200 cursor-pointer hover:opacity-90 transition-opacity"
                    @click="previewUrl = `${apiBase}${img.viz_url}`" />
                  <p class="text-xs font-medium text-dark mt-2 truncate">{{ img.filename }}</p>
                  <p class="text-xs text-dark/40">{{ img.predictions.length }} object(s)</p>
                </div>
                <div class="flex-1 overflow-x-auto p-3">
                  <table class="w-full text-xs">
                    <thead>
                      <tr class="text-dark/50 uppercase tracking-wide border-b border-gray-100">
                        <th class="p-2 text-left font-medium">#</th>
                        <th class="p-2 text-left font-medium">Class</th>
                        <th class="p-2 text-left font-medium">Category</th>
                        <th class="p-2 text-right font-medium">Conf</th>
                        <th class="p-2 text-right font-medium">X1</th>
                        <th class="p-2 text-right font-medium">Y1</th>
                        <th class="p-2 text-right font-medium">X2</th>
                        <th class="p-2 text-right font-medium">Y2</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(ann, j) in img.predictions" :key="j"
                        class="border-b border-gray-50 hover:bg-gray-50 transition-colors"
                      >
                        <td class="p-2 text-dark/40 font-mono">{{ j + 1 }}</td>
                        <td class="p-2 font-medium text-dark">{{ ann.class_id }}</td>
                        <td class="p-2">
                          <span :class="ann.category === 'Organik' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'"
                            class="font-bold px-2 py-0.5 rounded text-xs">{{ ann.category }}</span>
                        </td>
                        <td class="p-2 text-right font-mono text-dark/60">{{ (ann.confidence * 100).toFixed(0) }}%</td>
                        <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[0].toFixed(0) }}</td>
                        <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[1].toFixed(0) }}</td>
                        <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[2].toFixed(0) }}</td>
                        <td class="p-2 text-right font-mono text-orange-600">{{ ann.bbox[3].toFixed(0) }}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
          <Pagination :page="pageResult" :total="filteredRealResult.length" :per="perPage" @update:page="pageResult = $event" />
        </template>
      </div>

      <!-- Default: show stats -->
      <div v-if="!subKey" class="bg-white rounded-xl p-10 text-center border border-gray-100">
        <p class="text-dark/50 text-sm">Select a view from the sidebar.</p>
      </div>

    </div>

    <!-- ==================== EVALUATION ==================== -->
    <div v-if="tabKey === 'eval'">
      <div class="flex items-center justify-between mb-4">
        <h3 class="font-bold text-tertiary uppercase tracking-wide text-sm">Model Performance</h3>
        <button
          class="bg-tertiary hover:bg-blue-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl shadow-sm hover:shadow transition-all flex items-center gap-2 disabled:opacity-50"
          :disabled="evaluating" @click="runEval()"
        >
          <svg v-if="evaluating" class="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z" />
          </svg>
          {{ evaluating ? 'Evaluating...' : evalMetrics ? 'Refresh Evaluation' : 'Run Evaluation' }}
        </button>
      </div>

      <div v-if="evalError" class="bg-red-50 border border-red-200 rounded-xl p-4 mb-4">
        <p class="text-red-600 text-sm">{{ evalError }}</p>
      </div>

      <div v-if="evalMetrics" class="space-y-6">
        <div class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
          <table class="w-full text-sm">
            <thead class="bg-gray-50 text-dark/60 text-xs uppercase tracking-wide">
              <tr>
                <th class="p-3 text-left font-medium">Split</th>
                <th class="p-3 text-right font-medium">Images</th>
                <th class="p-3 text-right font-medium">mAP@0.5</th>
                <th class="p-3 text-right font-medium">mAP@0.5:0.95</th>
                <th class="p-3 text-right font-medium">Precision</th>
                <th class="p-3 text-right font-medium">Recall</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(m, split) in evalMetrics" :key="split"
                class="border-b border-gray-100 hover:bg-gray-50/50 transition-colors"
              >
                <td class="p-3 font-medium text-dark capitalize">{{ split }}</td>
                <td class="p-3 text-right text-dark">{{ splitCounts[split] || '-' }}</td>
                <td class="p-3 text-right font-mono" :class="scoreColor(m.mAP50)">{{ m.mAP50.toFixed(1) }}%</td>
                <td class="p-3 text-right font-mono" :class="scoreColor(m.mAP50_95)">{{ m.mAP50_95.toFixed(1) }}%</td>
                <td class="p-3 text-right font-mono" :class="scoreColor(m.precision)">{{ m.precision.toFixed(1) }}%</td>
                <td class="p-3 text-right font-mono" :class="scoreColor(m.recall)">{{ m.recall.toFixed(1) }}%</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div v-for="(m, split) in evalMetrics" :key="'pc-'+split" class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
          <div class="bg-gray-50 border-b border-gray-100 px-4 py-2">
            <h4 class="text-xs font-bold text-dark/60 uppercase tracking-wide">{{ split }} · Per-Class mAP@0.5</h4>
          </div>
          <table class="w-full text-sm" v-if="m.per_class && m.per_class.length">
            <thead class="text-dark/50 text-xs uppercase tracking-wide">
              <tr>
                <th class="p-2 pl-4 text-left font-medium">Class</th>
                <th class="p-2 text-right font-medium">mAP@0.5</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="pc in m.per_class" :key="pc.class_id"
                class="border-b border-gray-50 hover:bg-gray-50/50 transition-colors"
              >
                <td class="p-2 pl-4 font-medium text-dark">{{ pc.name }}</td>
                <td class="p-2 text-right font-mono" :class="scoreColor(pc.mAP50)">{{ pc.mAP50.toFixed(1) }}%</td>
              </tr>
            </tbody>
          </table>
          <p v-else class="p-4 text-center text-dark/40 text-xs">No per-class data</p>
        </div>
      </div>

      <div v-else-if="!evaluating" class="bg-white rounded-xl p-10 text-center border border-gray-100">
        <div class="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
          <svg class="w-6 h-6 text-dark/30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z" />
          </svg>
        </div>
        <p class="text-dark/50 text-sm">Click "Run Evaluation" to evaluate model performance on train, val, and test splits.</p>
      </div>
    </div>

    <!-- Preview Modal -->
    <Teleport to="body">
      <div v-if="previewUrl" class="fixed inset-0 bg-dark/60 z-50 flex items-center justify-center p-6" @click="previewUrl = null">
        <div class="relative max-w-3xl max-h-full" @click.stop>
          <button class="absolute -top-3 -right-3 w-8 h-8 bg-white rounded-full shadow-md flex items-center justify-center text-dark hover:text-red-500 transition-colors z-10" @click="previewUrl = null">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
          <img :src="previewUrl" class="max-w-full max-h-[90vh] rounded-xl shadow-2xl border-4 border-white" />
        </div>
      </div>
    </Teleport>

  </div>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase
const route = useRoute()
const router = useRouter()

const tabKey = computed(() => (route.query.tab as string) || "raw")
const subKey = computed(() => route.query.sub as string || "")

const data = ref<any>(null)
const error = ref<string | null>(null)
const loading = ref(true)
const training = ref(false)
const trainingType = ref('')
const previewUrl = ref<string | null>(null)

const evaluating = ref(false)
const evalMetrics = ref<any>(null)
const evalError = ref<string | null>(null)
const EVAL_STORAGE_KEY = 'sb_eval_metrics'

const page = ref(1)
const pageTrain = ref(1)
const pageVal = ref(1)
const pageResult = ref(1)
const perPage = 20

const splitCounts = ref<Record<string, number>>({ train: 0, val: 0, test: 0 })

watch(tabKey, () => { page.value = 1; pageTrain.value = 1; pageVal.value = 1; pageResult.value = 1 })

async function load() {
  loading.value = true; error.value = null
  try {
    data.value = await $fetch("/api/dataset/grid", { baseURL: apiBase })
  } catch (e: any) {
    error.value = e.data?.detail || e.message || "Failed to load"
  } finally { loading.value = false }
}

async function trainCoco() {
  if (training.value) return
  training.value = true
  trainingType.value = 'coco'
  try {
    await $fetch("/api/dataset/pipeline/coco", { baseURL: apiBase, method: "POST" })
    await load()
  } catch (e: any) {
    error.value = e.data?.detail || "COCO pipeline failed"
  } finally { training.value = false; trainingType.value = '' }
}

async function trainYolo() {
  if (training.value) return
  training.value = true
  trainingType.value = 'yolo'
  try {
    await $fetch("/api/dataset/pipeline/yolo", { baseURL: apiBase, method: "POST" })
    await load()
  } catch (e: any) {
    error.value = e.data?.detail || "YOLO pipeline failed"
  } finally { training.value = false; trainingType.value = '' }
}

async function trainYoloVal() {
  if (training.value) return
  training.value = true
  trainingType.value = 'yolo_val'
  try {
    await $fetch("/api/dataset/pipeline/yolo/val", { baseURL: apiBase, method: "POST" })
    await load()
  } catch (e: any) {
    error.value = e.data?.detail || "YOLO val pipeline failed"
  } finally { training.value = false; trainingType.value = '' }
}

const filteredCoco = computed(() => (data.value?.coco || []).filter((img: any) => img.annotation_count > 0))

const trainPlain = computed(() => (data.value?.train_convert || []).map((f: any) => ({
  filename: f.filename,
  image_url: f.image_url,
})))
const valPlain = computed(() => (data.value?.val_real || []).map((f: any) => ({
  filename: f.filename,
  image_url: f.image_url,
})))

const filteredTrainConvert = computed(() => (data.value?.train_convert || []).filter((img: any) => img.viz_url))
const filteredTrainReal = computed(() => (data.value?.train_real || []).filter((img: any) => img.viz_url))
const filteredValReal = computed(() => (data.value?.val_real || []).filter((img: any) => img.viz_url))
const filteredValResult = computed(() => (data.value?.val_real || []).filter((img: any) => img.viz_url && img.predictions.length > 0))

const filteredConvertResult = computed(() => (data.value?.train_convert || []).filter((img: any) => img.viz_url && img.predictions.length > 0))
const filteredRealResult = computed(() => (data.value?.train_real || []).filter((img: any) => img.viz_url && img.predictions.length > 0))

const totalCocoObjects = computed(() =>
  filteredCoco.value.reduce((s: number, img: any) => s + img.annotation_count, 0)
)

const paginatedRaw = computed(() => {
  const items = data.value?.raw || []
  const start = (page.value - 1) * perPage
  return items.slice(start, start + perPage)
})

const paginatedCoco = computed(() => {
  const start = (page.value - 1) * perPage
  return filteredCoco.value.slice(start, start + perPage)
})

const paginatedTrainPlain = computed(() => {
  const start = (pageTrain.value - 1) * perPage
  return trainPlain.value.slice(start, start + perPage)
})

const paginatedTrainConvert = computed(() => {
  const start = (pageTrain.value - 1) * perPage
  return filteredTrainConvert.value.slice(start, start + perPage)
})

const paginatedTrainReal = computed(() => {
  const start = (pageTrain.value - 1) * perPage
  return filteredTrainReal.value.slice(start, start + perPage)
})

const paginatedValPlain = computed(() => {
  const start = (pageVal.value - 1) * perPage
  return valPlain.value.slice(start, start + perPage)
})

const paginatedValReal = computed(() => {
  const start = (pageVal.value - 1) * perPage
  return filteredValReal.value.slice(start, start + perPage)
})

const paginatedValResult = computed(() => {
  const start = (pageVal.value - 1) * perPage
  return filteredValResult.value.slice(start, start + perPage)
})

const paginatedConvertResult = computed(() => {
  const start = (pageResult.value - 1) * perPage
  return filteredConvertResult.value.slice(start, start + perPage)
})

const paginatedRealResult = computed(() => {
  const start = (pageResult.value - 1) * perPage
  return filteredRealResult.value.slice(start, start + perPage)
})

function scoreColor(val: number): string {
  if (val >= 80) return 'text-green-600'
  if (val >= 60) return 'text-yellow-600'
  return 'text-red-600'
}

async function runEval() {
  evaluating.value = true; evalError.value = null; evalMetrics.value = null
  try {
    const res = await $fetch("/api/dataset/evaluate", { baseURL: apiBase, params: { split: "all" } })
    evalMetrics.value = res.metrics
    localStorage.setItem(EVAL_STORAGE_KEY, JSON.stringify(res.metrics))
  } catch (e: any) {
    evalError.value = e.data?.detail || e.message || "Evaluation failed"
  } finally { evaluating.value = false }
}

watch(data, (d) => {
  if (d?.stats) {
    splitCounts.train = d.stats.train
    splitCounts.val = d.stats.val
    splitCounts.test = d.stats.test
  }
})

onMounted(() => {
  load()
  try {
    const cached = localStorage.getItem(EVAL_STORAGE_KEY)
    if (cached) evalMetrics.value = JSON.parse(cached)
  } catch {}
})
</script>
