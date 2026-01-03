
<template>
  <div class="p-4 flex flex-col items-center justify-center min-h-screen font-sans">
    <!-- <div class="w-full max-w-sm text-center"> -->
      <!-- <h1 class="text-3xl font-bold text-white mb-2">Liquid Slider</h1> -->
      <!-- <p class="text-gray-400 mb-6">Drag the liquid in the tank to change the value.</p> -->
    <!-- </div> -->

    <div class="flex items-center gap-8">
      Liquid Tank
      <!-- <div ref="containerRef" class="tank-container" @mousedown="startDrag"> -->
        <!-- <canvas ref="canvasRef"></canvas> -->
      <!-- </div> -->
      <div ref="containerRef" class="tank-container" @mousedown="startDrag">
        <canvas ref="canvasRef"></canvas>
      </div>

      <!-- Value Display -->
      <!-- <div class="bg-gray-700 p-6 rounded-xl shadow-lg text-center"> -->
        <!-- <div class="text-gray-400 text-sm font-medium uppercase tracking-wider">Value</div> -->
        <!-- <div class="text-white text-5xl font-bold mt-1">{{ formattedValue }}</div> -->
      <!-- </div> -->
    </div>
    <!-- <div class="text-gray-500 text-xs mt-6"> -->
        <!-- Note: This is a single Vue component. The text and value display are for demonstration. -->
    <!-- </div> -->
  </div>
</template>

<script>
export default {
    props: {
        value: {
            type: Number,
            default: 0.5,
        },
    },
    emits: ['update:value'],
    data() {
        return {
            // Local state variables for the component
            isDragging: false,
            animationFrameId: null,
            time: 0,
            amplitude: 0,
            frequency: 10,
            speed: 0.1,
            maxAmplitude: 0.05,
            settleDecay: 0.97,
            localValue: this.value, // Initialize local value from the prop
        };
    },
    mounted() {
        // Lifecycle hook for when the component is mounted
        const canvas = this.$refs.canvasRef;
        this.ctx = canvas.getContext('2d');

        const observer = new ResizeObserver(entries => {
            const entry = entries[0];
            const { width, height } = entry.contentRect;
            canvas.width = width;
            canvas.height = height;
        });

        if (this.$refs.containerRef) {
            observer.observe(this.$refs.containerRef);
        }

        this.animationLoop();
    },
    unmounted() {
        // Lifecycle hook for when the component is unmounted
        window.removeEventListener('mousemove', this.handleDrag);
        window.removeEventListener('mouseup', this.stopDrag);
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
    },
    watch: {
        // Watch for changes to the 'value' prop from the parent
        value(newValue) {
            this.localValue = newValue;
            if (!this.isDragging) {
                const diff = Math.abs(newValue - this.localValue);
                this.amplitude = Math.min(this.maxAmplitude, this.amplitude + diff * 100);
            }
        },
    },
    computed: {
        // A computed property to format the value
        formattedValue() {
            return this.localValue.toFixed(2);
        },
    },
    methods: {
        // --- Event Handlers ---
        startDrag(event) {
            event.preventDefault();
            this.isDragging = true;
            this.amplitude = this.maxAmplitude;
            this.updateValueFromEvent(event);
            window.addEventListener('mousemove', this.handleDrag);
            window.addEventListener('mouseup', this.stopDrag);
        },
        handleDrag(event) {
            if (!this.isDragging) return;
            this.updateValueFromEvent(event);
        },
        stopDrag() {
            this.isDragging = false;
            window.removeEventListener('mousemove', this.handleDrag);
            window.removeEventListener('mouseup', this.stopDrag);
        },
        // --- Logic ---
        updateValueFromEvent(event) {
            if (!this.$refs.containerRef) return;
            const rect = this.$refs.containerRef.getBoundingClientRect();
            const mouseY = event.clientY - rect.top;
            const rawValue = 1 - (mouseY / rect.height);
            const clampedValue = Math.max(0, Math.min(1, rawValue));
            
            // Update the local data property and emit the event
            this.localValue = clampedValue;
            this.$emit('update:value', clampedValue);
        },
        // --- Canvas Drawing ---
        draw() {
            const { canvasRef } = this.$refs;
            if (!this.ctx || !canvasRef) return;
            const { width, height } = canvasRef;

            this.ctx.clearRect(0, 0, width, height);
            if (!this.isDragging) {
                this.amplitude *= this.settleDecay;
            }
            this.time += this.speed;

            const liquidHeight = height * this.localValue;
            const surfaceY = height - liquidHeight;

            this.ctx.fillStyle = '#3b82f6';
            this.ctx.beginPath();
            this.ctx.moveTo(0, height);

            if (this.localValue > 0.01 && this.amplitude > 0.001) {
                for (let x = 0; x <= width; x++) {
                    const waveY = surfaceY + Math.sin((x / width) * this.frequency + this.time) * height * this.amplitude * Math.sin(0.4 * this.time);
                    this.ctx.lineTo(x, waveY);
                }
            } else {
                this.ctx.lineTo(0, surfaceY);
                this.ctx.lineTo(width, surfaceY);
            }
            this.ctx.lineTo(width, height);
            this.ctx.closePath();
            this.ctx.fill();
        },
        animationLoop() {
            this.draw();
            this.animationFrameId = requestAnimationFrame(this.animationLoop);
        },
        jiggle() {
          this.amplitude = this.maxAmplitude
        },
    },
};
</script>

<style scoped>
/* Scoped styles for the component */
.tank-container {
  width: 150px;
  height: 300px;
  border: 4px solid black;
  background-color: #1f2937; /* bg-gray-800 */
  border-radius: 0.75rem; /* rounded-xl */
  cursor: ns-resize; /* North-South resize cursor indicates vertical dragging */
  position: relative;
  overflow: hidden; /* Ensures canvas doesn't draw outside the border */
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

.tank-container canvas {
  display: block;
  width: 100%;
  height: 100%;
}
</style>
