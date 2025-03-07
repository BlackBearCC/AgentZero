<template>
  <div class="tv-input-wrapper">
    <label v-if="label" :for="id" class="input-label">{{ label }}</label>
    <input
      :id="id"
      class="tv-input"
      :type="type"
      :value="modelValue"
      @input="$emit('update:modelValue', $event.target.value)"
      v-bind="$attrs"
    />
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  modelValue: [String, Number],
  label: String,
  type: {
    type: String,
    default: 'text'
  },
  id: {
    type: String,
    default: () => `input-${Math.random().toString(36).substring(2, 9)}`
  }
})

defineEmits(['update:modelValue'])
</script>

<style lang="scss" scoped>
.tv-input-wrapper {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  
  .input-label {
    font-size: var(--font-size-sm);
    color: var(--color-text);
    margin-bottom: var(--spacing-xs);
    text-transform: uppercase;
    letter-spacing: var(--letter-spacing-tight);
    text-shadow: 0 0 5px var(--shadow-secondary);
    font-family: var(--font-family-secondary);
  }
  
  .tv-input {
    background: rgba(0, 20, 40, 0.8);
    border: 1px solid rgba(0, 195, 255, 0.3);
    border-radius: var(--border-radius-sm);
    padding: var(--spacing-sm) var(--spacing-md);
    color: var(--color-text-light);
    font-family: var(--font-family-secondary);
    outline: none;
    transition: all var(--animation-speed-fast);
    
    &:focus {
      border-color: var(--color-primary);
      box-shadow: 0 0 10px var(--shadow-primary);
    }
    
    &::placeholder {
      color: rgba(0, 195, 255, 0.5);
    }
  }
}
</style>