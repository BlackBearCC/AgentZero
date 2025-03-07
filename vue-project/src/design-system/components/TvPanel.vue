<template>
  <div :class="['tv-panel', { 'with-border': border }]">
    <slot></slot>
  </div>
</template>

<script setup>
defineProps({
  border: {
    type: Boolean,
    default: true
  }
})
</script>

<style lang="scss" scoped>
.tv-panel {
  background: rgba(0, 20, 40, 0.7);
  border: 1px solid rgba(0, 195, 255, 0.3);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  position: relative;
  overflow: hidden;
  margin-bottom: var(--spacing-md);
  box-shadow: 
    0 0 15px rgba(0, 195, 255, 0.1),
    inset 0 0 10px rgba(0, 195, 255, 0.05);
  backdrop-filter: blur(5px);
  
  &.with-border {
    border: 1px solid var(--color-border);
  }
  
  &::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(
      45deg,
      transparent,
      rgba(0, 195, 255, 0.1),
      transparent
    );
    transform: rotate(45deg);
    animation: controlGlow 8s infinite;
    z-index: -1;
  }
}

@keyframes controlGlow {
  0% { transform: translate(-50%, -50%) rotate(45deg); }
  100% { transform: translate(150%, 150%) rotate(45deg); }
}
</style>