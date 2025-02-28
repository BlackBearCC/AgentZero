<template>
  <div class="status-message" :class="type">
    <div class="message-content">
      <span class="status-icon">{{ getIcon }}</span>
      <span class="status-text">{{ message }}</span>
    </div>
    <div class="status-animation"></div>
  </div>
</template>

<script setup>
const props = defineProps({
  message: {
    type: String,
    required: true
  },
  type: {
    type: String,
    default: 'info',
    validator: (value) => ['info', 'success', 'warning', 'error'].includes(value)
  }
})

const getIcon = computed(() => {
  const icons = {
    info: 'ℹ',
    success: '✓',
    warning: '⚠',
    error: '✕'
  }
  return icons[props.type]
})
</script>

<style scoped>
.status-message {
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 4px;
  padding: 0.75rem 1rem;
  margin: 0.5rem 0;
  position: relative;
  overflow: hidden;
}

.message-content {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  z-index: 2;
  position: relative;
}

.status-icon {
  font-size: 1.2rem;
  line-height: 1;
}

.status-text {
  font-size: 0.9rem;
  color: #e0e0e0;
}

.status-animation {
  position: absolute;
  bottom: 0;
  left: 0;
  height: 2px;
  background: #44ff44;
  animation: status-scan 2s linear infinite;
}

/* 状态类型样式 */
.info {
  border-color: rgba(68, 255, 68, 0.3);
  .status-animation { background: #44ff44; }
}

.success {
  border-color: rgba(68, 255, 68, 0.5);
  .status-animation { background: #44ff44; }
}

.warning {
  border-color: rgba(255, 189, 82, 0.3);
  .status-animation { background: #ffbd52; }
}

.error {
  border-color: rgba(255, 82, 82, 0.3);
  .status-animation { background: #ff5252; }
}

@keyframes status-scan {
  0% { width: 0; left: 0; }
  50% { width: 100%; left: 0; }
  51% { width: 100%; right: 0; }
  100% { width: 0; right: 0; }
}
</style> 