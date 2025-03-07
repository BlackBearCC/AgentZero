<template>
  <div class="tv-slider">
    <div class="slider-label" v-if="label">{{ label }}</div>
    <div class="slider-control">
      <input 
        type="range" 
        :min="min" 
        :max="max" 
        :step="step"
        :value="modelValue" 
        @input="$emit('update:modelValue', $event.target.value)"
        class="slider-input"
        :disabled="disabled"
      />
      <div class="slider-value">{{ modelValue }}</div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  modelValue: {
    type: [Number, String],
    required: true
  },
  min: {
    type: [Number, String],
    default: 0
  },
  max: {
    type: [Number, String],
    default: 100
  },
  step: {
    type: [Number, String],
    default: 1
  },
  label: {
    type: String,
    default: ''
  },
  disabled: {
    type: Boolean,
    default: false
  }
})

defineEmits(['update:modelValue'])
</script>

<style lang="scss" scoped>
.tv-slider {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  
  .slider-label {
    font-size: var(--font-size-sm);
    color: var(--color-text);
    letter-spacing: var(--letter-spacing-tight);
    text-transform: uppercase;
    text-shadow: 0 0 5px var(--shadow-secondary);
  }
  
  .slider-control {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
  }
  
  .slider-input {
    flex: 1;
    height: 5px;
    -webkit-appearance: none;
    background: var(--color-panel);
    border-radius: var(--border-radius-sm);
    outline: none;
    border: 1px solid var(--color-border);
    
    &::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 15px;
      height: 15px;
      border-radius: 50%;
      background: var(--color-primary);
      cursor: pointer;
      box-shadow: 0 0 5px var(--shadow-primary);
    }
    
    &:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
  }
  
  .slider-value {
    min-width: 30px;
    text-align: center;
    color: var(--color-text);
    font-family: var(--font-family-secondary);
  }
}
</style>