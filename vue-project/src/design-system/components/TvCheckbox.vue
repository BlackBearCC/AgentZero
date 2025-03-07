<template>
    <div class="tv-checkbox">
      <input 
        type="checkbox" 
        :id="id" 
        :checked="modelValue" 
        @change="$emit('update:modelValue', $event.target.checked)"
      />
      <label :for="id">
        <slot></slot>
      </label>
    </div>
  </template>
  
  <script setup>
  import { computed } from 'vue'
  
  const props = defineProps({
    modelValue: Boolean,
    id: {
      type: String,
      default: () => `checkbox-${Math.random().toString(36).substring(2, 9)}`
    }
  })
  
  defineEmits(['update:modelValue'])
  </script>
  
  <style lang="scss" scoped>
  .tv-checkbox {
    display: flex;
    align-items: center;
    gap: 10px;
    
    input[type="checkbox"] {
      appearance: none;
      width: 20px;
      height: 20px;
      border: 2px solid rgba(0, 195, 255, 0.5);
      border-radius: 4px;
      background: rgba(0, 20, 40, 0.8);
      cursor: pointer;
      position: relative;
      transition: all 0.3s ease;
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
          rgba(0, 195, 255, 0.1),
          transparent
        );
        transform: rotate(45deg);
        animation: checkboxGlow 4s infinite;
        z-index: -1;
        opacity: 0.5;
      }
      
      &:checked {
        background: #00c3ff;
        box-shadow: 0 0 10px rgba(0, 195, 255, 0.5);
        
        &::before {
          content: '✓';
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          color: #000;
          font-size: 14px;
          z-index: 2;
        }
        
        &::after {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: radial-gradient(
            circle at center,
            rgba(0, 195, 255, 0.8) 0%,
            rgba(0, 195, 255, 0.4) 60%,
            transparent 100%
          );
          animation: checkboxPulse 2s infinite;
          z-index: 1;
        }
      }
      
      &:hover:not(:checked) {
        border-color: #00c3ff;
        box-shadow: 0 0 8px rgba(0, 195, 255, 0.3);
      }
    }
    
    label {
      font-size: 1rem;
      color: #00c3ff;
      cursor: pointer;
      text-shadow: 0 0 5px rgba(0, 195, 255, 0.4);
      font-family: 'Eurostile', 'Share Tech Mono', monospace;
      letter-spacing: 1px;
      text-transform: uppercase;
    }
  }
  
  @keyframes checkboxGlow {
    0% { transform: translate(-50%, -50%) rotate(45deg); }
    100% { transform: translate(150%, 150%) rotate(45deg); }
  }
  
  @keyframes checkboxPulse {
    0% { opacity: 0.7; }
    50% { opacity: 0.3; }
    100% { opacity: 0.7; }
  }
  </style>