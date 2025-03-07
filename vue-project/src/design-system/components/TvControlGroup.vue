<template>
  <div class="tv-control-group">
    <div class="control-label" v-if="label">{{ label }}</div>
    <div class="control-content">
      <slot></slot>
    </div>
  </div>
</template>

<script setup>
defineProps({
  label: {
    type: String,
    default: ''
  }
})
</script>

<style lang="scss" scoped>
.tv-control-group {
  background: var(--color-panel);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-md);
  box-shadow: 0 0 15px var(--shadow-secondary), inset 0 0 10px rgba(0, 195, 255, 0.05);
  backdrop-filter: blur(5px);
  position: relative;
  overflow: hidden;
  
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
      var(--effect-scan),
      transparent
    );
    transform: rotate(45deg);
    animation: controlGlow 8s infinite;
    z-index: -1;
  }
  
  .control-label {
    font-size: var(--font-size-sm);
    color: var(--color-text);
    letter-spacing: var(--letter-spacing-tight);
    text-transform: uppercase;
    text-shadow: 0 0 8px var(--shadow-secondary);
  }
  
  .control-content {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
  }
}

@keyframes controlGlow {
  0% { transform: translate(-50%, -50%) rotate(45deg); }
  100% { transform: translate(150%, 150%) rotate(45deg); }
}
</style>