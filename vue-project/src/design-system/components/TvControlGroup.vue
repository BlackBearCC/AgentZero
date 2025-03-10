<template>
  <div class="tv-control-group">
    <div class="control-header" v-if="label">
      <div class="control-indicator"></div>
      <div class="control-label">{{ label }}</div>
      <div class="control-line"></div>
    </div>
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
  background: rgba(0, 20, 40, 0.7);
  border: 1px solid rgba(0, 195, 255, 0.3);
  border-radius: 4px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 16px;
  box-shadow: 0 0 15px rgba(0, 195, 255, 0.1), inset 0 0 10px rgba(0, 195, 255, 0.05);
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
      rgba(0, 195, 255, 0.05),
      transparent
    );
    transform: rotate(45deg);
    animation: controlGlow 8s infinite;
    z-index: -1;
  }
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(
      90deg,
      transparent,
      rgba(0, 195, 255, 0.5),
      transparent
    );
    opacity: 0.5;
  }
  
  .control-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }
  
  .control-indicator {
    width: 6px;
    height: 6px;
    background: #00c3ff;
    box-shadow: 0 0 8px rgba(0, 195, 255, 0.8);
    animation: pulse 2s infinite alternate;
  }
  
  .control-label {
    font-size: 12px;
    color: #00c3ff;
    letter-spacing: 1px;
    text-transform: uppercase;
    text-shadow: 0 0 8px rgba(0, 195, 255, 0.4);
    font-weight: 500;
  }
  
  .control-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(
      90deg,
      rgba(0, 195, 255, 0.5),
      transparent
    );
    margin-left: 4px;
  }
  
  .control-content {
    display: flex;
    flex-direction: column;
    gap: 10px;
    position: relative;
    z-index: 1;
    
    &::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: radial-gradient(
        circle at center,
        rgba(0, 195, 255, 0.03) 0%,
        transparent 70%
      );
      pointer-events: none;
      z-index: -1;
    }
  }
  
  &:hover {
    border-color: rgba(0, 195, 255, 0.5);
    box-shadow: 0 0 20px rgba(0, 195, 255, 0.15), inset 0 0 15px rgba(0, 195, 255, 0.07);
    
    .control-indicator {
      animation: pulse-fast 1s infinite alternate;
    }
  }
}

@keyframes controlGlow {
  0% { transform: translate(-50%, -50%) rotate(45deg); }
  100% { transform: translate(150%, 150%) rotate(45deg); }
}

@keyframes pulse {
  0% { opacity: 0.7; }
  100% { opacity: 1; }
}

@keyframes pulse-fast {
  0% { opacity: 0.7; }
  100% { opacity: 1; }
}
</style>