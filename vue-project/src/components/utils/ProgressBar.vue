<template>
  <div class="progress-container">
    <div class="progress-bar">
      <div class="progress-fill" :style="progressStyle"></div>
    </div>
    <div class="progress-text">{{ processed }}/{{ total }}</div>
  </div>
</template>

<script setup>
defineProps({
  processed: {
    type: Number,
    required: true
  },
  total: {
    type: Number,
    required: true
  }
})

const progressStyle = computed(() => ({
  width: `${(props.processed / props.total) * 100}%`
}))
</script>

<style scoped>
.progress-container {
  width: 100%;
}

.progress-bar {
  height: 10px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 5px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #44ff44, #7cff7c);
  border-radius: 5px;
  transition: width 0.3s ease;
}

.progress-text {
  text-align: center;
  font-size: 0.8rem;
  color: #8a8a9a;
}
</style> 