
<template>
  <div class="p-4 flex flex-col items-center justify-center min-h-screen font-sans">
    <div class="w-full max-w-sm text-center">
      <h1 class="text-3xl font-bold text-white mb-2">Liquid Slider</h1>
      <p class="text-gray-400 mb-6">Drag the liquid in the tank to change the value.</p>
    </div>

    <!-- <div class="flex items-center gap-8"> -->
      <!-- <!-- The Liquid Tank Component --> -->
      <!-- <div ref="containerRef" class="tank-container" @mousedown="startDrag"> -->
        <!-- <canvas ref="canvasRef"></canvas> -->
      <!-- </div> -->

      <!-- <!-- Value Display --> -->
      <!-- <div class="bg-gray-700 p-6 rounded-xl shadow-lg text-center"> -->
        <!-- <div class="text-gray-400 text-sm font-medium uppercase tracking-wider">Value</div> -->
        <!-- <!-- <div class="text-white text-5xl font-bold mt-1">{{ formattedValue }}</div> --> -->
      <!-- </div> -->
    <!-- </div> -->
    <!-- <div class="text-gray-500 text-xs mt-6"> -->
        <!-- Note: This is a single Vue component. The text and value display are for demonstration. -->
    <!-- </div> -->
  <!-- </div> -->
</template>

<!-- <script setup> -->
<script>
import { ref, onMounted, onUnmounted, computed, watch } from 'vue';

// --- Props and Emits ---
// Using defineModel for two-way binding (Vue 3.4+)
// This simplifies v-model usage.
const modelValue = defineModel({ type: Number, default: 0.5 });

// --- Template Refs ---
const containerRef = ref(null);
const canvasRef = ref(null);

// --- Component State ---
const isDragging = ref(false);
const animationFrameId = ref(null);

// Wave properties
const time = ref(0); // Phase shift for the sine wave
const amplitude = ref(0); // Current amplitude of the wave
const frequency = ref(10); // How many waves across the width
const speed = ref(0.1); // How fast the wave oscillates
const maxAmplitude = 0.05; // Max amplitude when actively dragging, relative to height
const settleDecay = 0.97; // Rate at which the wave settles (e.g., 0.98 = 2% decay per frame)

// --- Computed Properties ---
const formattedValue = computed(() => modelValue.value.toFixed(2));

// --- Event Handlers ---
const startDrag = (event) => {
  event.preventDefault();
  isDragging.value = true;
  amplitude.value = maxAmplitude;
  updateValueFromEvent(event);
  window.addEventListener('mousemove', handleDrag);
  window.addEventListener('mouseup', stopDrag);
};

const handleDrag = (event) => {
  if (!isDragging.value) return;
  updateValueFromEvent(event);
};

const stopDrag = () => {
  isDragging.value = false;
  window.removeEventListener('mousemove', handleDrag);
  window.removeEventListener('mouseup', stopDrag);
};

// --- Logic ---
const updateValueFromEvent = (event) => {
  if (!containerRef.value) return;
  const rect = containerRef.value.getBoundingClientRect();
  const mouseY = event.clientY - rect.top;
  const rawValue = 1 - (mouseY / rect.height);
  // Clamp value between 0 and 1
  modelValue.value = Math.max(0, Math.min(1, rawValue));
};

// --- Canvas Drawing ---
let ctx = null;

const draw = () => {
  if (!ctx || !canvasRef.value) return;

  const canvas = canvasRef.value;
  const { width, height } = canvas;

  // Clear canvas for next frame
  ctx.clearRect(0, 0, width, height);

  // Smoothly decay amplitude if not dragging
  if (!isDragging.value) {
    amplitude.value *= settleDecay;
  }

  // Increment time for wave motion
  time.value += speed.value;

  // Define the liquid properties
  const liquidHeight = height * modelValue.value;
  const surfaceY = height - liquidHeight;
  
  ctx.fillStyle = '#3b82f6'; // Blue color for the liquid
  ctx.beginPath();
  
  // Start path from bottom-left
  ctx.moveTo(0, height);
  
  // TODO: add amplitude basd on change

  // Draw the wavy surface if amplitude is above threshold
  if (modelValue.value > 0.01 && amplitude.value > 0.001) {
    for (let x = 0; x <= width; x++) {
      // Calculate y position of the wave at this x coordinate
      const waveY = surfaceY + Math.sin((x / width) * frequency.value + time.value) * height * amplitude.value * Math.sin(0.4*time.value); //* Math.sin(Math.PI * modelValue.value);
      ctx.lineTo(x, waveY);
    }
  } else {
    // If settled or empty, draw a flat surface
    ctx.lineTo(0, surfaceY);
    ctx.lineTo(width, surfaceY);
  }
  
  // Complete the shape to the bottom-right
  ctx.lineTo(width, height);
  ctx.closePath();
  ctx.fill();
};

const animationLoop = () => {
  draw();
  animationFrameId.value = requestAnimationFrame(animationLoop);
};

// --- Lifecycle Hooks ---
onMounted(() => {
  const canvas = canvasRef.value;
  ctx = canvas.getContext('2d');

  // Use a ResizeObserver to handle container resizing gracefully
  const observer = new ResizeObserver(entries => {
    const entry = entries[0];
    const { width, height } = entry.contentRect;
    // Set canvas resolution to match its display size
    canvas.width = width;
    canvas.height = height;
  });

  if (containerRef.value) {
    observer.observe(containerRef.value);
  }

  // Start the animation loop
  animationLoop();
});

onUnmounted(() => {
  // Clean up listeners and animation frames
  window.removeEventListener('mousemove', handleDrag);
  window.removeEventListener('mouseup', stopDrag);
  if (animationFrameId.value) {
    cancelAnimationFrame(animationFrameId.value);
  }
});

// Watch for external changes to the modelValue
watch(modelValue, (newValue, oldValue) => {
  // Give the liquid a 'jolt' if the value is changed programmatically
  if (!isDragging.value) {
    const diff = Math.abs(newValue - oldValue);
    // Add energy to the wave proportional to the change
    amplitude.value = Math.min(maxAmplitude, amplitude.value + diff * 100);
  }
});
</script>

<!-- <style scoped> -->
<!-- /* Scoped styles for the component */ -->
<!-- .tank-container { -->
  <!-- width: 150px; -->
  <!-- height: 300px; -->
  <!-- border: 4px solid black; -->
  <!-- background-color: #1f2937; /* bg-gray-800 */ -->
  <!-- border-radius: 0.75rem; /* rounded-xl */ -->
  <!-- cursor: ns-resize; /* North-South resize cursor indicates vertical dragging */ -->
  <!-- position: relative; -->
  <!-- overflow: hidden; /* Ensures canvas doesn't draw outside the border */ -->
  <!-- box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); -->
<!-- } -->

<!-- .tank-container canvas { -->
  <!-- display: block; -->
  <!-- width: 100%; -->
  <!-- height: 100%; -->
<!-- } -->
<!-- </style> -->
