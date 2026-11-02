<script>
// Mermaid shares parser/theme state; initialize and render each diagram together.
let renderQueue = Promise.resolve()
</script>

<script setup>
import { computed, onMounted, onUnmounted, ref, useId, watch } from 'vue'
import { useData } from 'vitepress'

const props = defineProps({ graph: { type: String, required: true } })
const source = computed(() => decodeURIComponent(props.graph))
const { isDark } = useData()
const id = `mermaid-${useId()}`
const svg = ref('')
const error = ref('')
let disposed = false
let stop

function render() {
  const code = source.value
  const theme = isDark.value ? 'dark' : 'default'
  renderQueue = renderQueue.then(async () => {
    if (disposed) return
    try {
      const { default: mermaid } = await import('mermaid')
      if (disposed) return
      mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme,
        flowchart: { htmlLabels: false } })
      const result = await mermaid.render(id, code)
      if (!disposed) {
        svg.value = result.svg
        error.value = ''
      }
    } catch (cause) {
      if (!disposed) {
        svg.value = ''
        error.value = String(cause)
      }
    }
  })
}

onMounted(() => { stop = watch([source, isDark], render, { immediate: true }) })
onUnmounted(() => { disposed = true; stop?.() })
</script>

<template>
  <div class="mermaid">
    <p v-if="error" role="alert">Diagram could not be rendered: {{ error }}</p>
    <div v-if="svg" v-html="svg" />
    <pre v-else>{{ source }}</pre>
  </div>
</template>

<style scoped>
.mermaid { margin: 24px 0; overflow-x: auto; }
.mermaid :deep(svg) { display: block; margin: auto; }
</style>
