<template>
    <button 
      :class="[
        'tv-button',
        {
          'primary': primary,
          'disabled': disabled,
          'active': active,
          [size]: size
        }
      ]"
      :disabled="disabled"
      @click="$emit('click')"
    >
      <div class="button-face">
        <slot></slot>
        <div v-if="indicator" class="power-indicator" :class="{ 'active': active }"></div>
      </div>
    </button>
  </template>
  
  <script setup>
  defineProps({
    primary: Boolean,
    disabled: Boolean,
    active: Boolean,
    indicator: Boolean,
    size: {
      type: String,
      default: '',
      validator: (value) => ['sm', 'md', 'lg', ''].includes(value)
    }
  })
  
  defineEmits(['click'])
  </script>
  
  <style lang="scss" scoped>
  .tv-button {
    background: rgba(0, 20, 40, 0.8);
    border: 1px solid rgba(0, 195, 255, 0.5);
    border-radius: 5px;
    padding: 10px;
    color: #00c3ff;
    font-family: 'Eurostile', 'Share Tech Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 2px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    cursor: pointer;
    text-shadow: 0 0 5px rgba(0, 195, 255, 0.4);
    
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
      animation: buttonGlow 3s infinite;
      z-index: -1;
    }
    
    &:hover:not(:disabled) {
      background: rgba(0, 40, 80, 0.8);
      border-color: #00c3ff;
      box-shadow: 
        0 0 15px rgba(0, 195, 255, 0.5),
        inset 0 0 15px rgba(0, 195, 255, 0.3);
      transform: translateY(-1px);
    }
    
    &.primary {
      background: rgba(0, 40, 80, 0.9);
      border-color: #00c3ff;
      padding: 12px 30px;
      font-size: 1.2rem;
      text-shadow: 0 0 8px rgba(0, 195, 255, 0.6);
      
      &:hover:not(:disabled) {
        background: rgba(0, 60, 100, 0.9);
        box-shadow: 
          0 0 20px rgba(0, 195, 255, 0.6),
          inset 0 0 20px rgba(0, 195, 255, 0.4);
        transform: translateY(-2px);
      }
    }
    
    &.active {
      background: rgba(0, 40, 80, 0.8);
      border-color: #00c3ff;
      box-shadow: 
        0 0 15px rgba(0, 195, 255, 0.5),
        inset 0 0 15px rgba(0, 195, 255, 0.3);
    }
    
    &.sm {
      padding: 5px 10px;
      font-size: 0.8rem;
    }
    
    &.lg {
      padding: 15px 30px;
      font-size: 1.2rem;
    }
    
    &:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      box-shadow: none;
    }
    
    .button-face {
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: relative;
      z-index: 100;
    }
    
    .power-indicator {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background-color: #444;
      transition: all 0.3s ease;
      position: relative;
      margin-left: 10px;
      z-index: 100;
      
      &.active {
        background-color: #00c3ff;
        box-shadow: 0 0 10px rgba(0, 195, 255, 0.5);
        
        &::after {
          content: '';
          position: absolute;
          top: -4px;
          left: -4px;
          right: -4px;
          bottom: -4px;
          border-radius: 50%;
          border: 1px solid rgba(0, 195, 255, 0.5);
          animation: pulse 2s infinite;
        }
      }
    }
  }
  
  @keyframes buttonGlow {
    0% { transform: translate(-50%, -50%) rotate(45deg); }
    100% { transform: translate(150%, 150%) rotate(45deg); }
  }
  
  @keyframes pulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.2); opacity: 0.5; }
    100% { transform: scale(1); opacity: 1; }
  }
  </style>