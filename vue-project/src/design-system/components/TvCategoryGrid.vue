<template>
  <div class="tv-category-grid">
    <div 
      v-for="(category, index) in categories" 
      :key="index"
      class="category-option"
    >
      <TvCheckbox 
        :id="`category-${index}`"
        v-model="selectedCategories"
        :value="category.key"
      >
        {{ category.title }}
      </TvCheckbox>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import TvCheckbox from './TvCheckbox.vue'

const props = defineProps({
  categories: {
    type: Array,
    required: true
  },
  modelValue: {
    type: Array,
    default: () => []
  }
})

const selectedCategories = ref(props.modelValue)

watch(selectedCategories, (newVal) => {
  emit('update:modelValue', newVal)
})

const emit = defineEmits(['update:modelValue'])
</script>

<style lang="scss" scoped>
.tv-category-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: var(--spacing-md);
  margin-top: var(--spacing-sm);
  
  .category-option {
    display: flex;
    align-items: center;
    padding: var(--spacing-sm);
    background: rgba(0, 20, 40, 0.7);
    border-radius: var(--border-radius-sm);
    transition: all var(--animation-speed-fast);
    
    &:hover {
      background: rgba(0, 40, 80, 0.7);
      box-shadow: 0 0 10px var(--shadow-secondary);
    }
  }
}

@media (max-width: 768px) {
  .tv-category-grid {
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  }
}
</style>