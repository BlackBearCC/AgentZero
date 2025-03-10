<template>
  <div class="tv-checkbox-group">
    <div v-if="showSelectAll" class="tv-checkbox select-all">
      <input 
        type="checkbox" 
        :id="`${id}-all`" 
        :checked="allSelected"
        :indeterminate="indeterminate"
        @change="toggleAll"
      />
      <label :for="`${id}-all`">
        {{ selectAllLabel }}
      </label>
    </div>
    
    <div class="checkbox-items">
      <div 
        v-for="(option, index) in options" 
        :key="index"
        class="tv-checkbox"
      >
        <input 
          type="checkbox" 
          :id="`${id}-${index}`" 
          :checked="isSelected(option.value)"
          @change="toggleOption(option.value)"
        />
        <label :for="`${id}-${index}`">
          {{ option.label }}
        </label>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'

const props = defineProps({
  modelValue: {
    type: Array,
    default: () => []
  },
  options: {
    type: Array,
    default: () => []
  },
  id: {
    type: String,
    default: () => `checkbox-group-${Math.random().toString(36).substring(2, 9)}`
  },
  showSelectAll: {
    type: Boolean,
    default: true
  },
  selectAllLabel: {
    type: String,
    default: '全选'
  }
})

const emit = defineEmits(['update:modelValue'])

// 计算是否全部选中
const allSelected = computed(() => {
  return props.options.length > 0 && props.modelValue.length === props.options.length
})

// 计算是否部分选中
const indeterminate = computed(() => {
  return props.modelValue.length > 0 && props.modelValue.length < props.options.length
})

// 检查选项是否被选中
const isSelected = (value) => {
  return props.modelValue.includes(value)
}

// 切换单个选项
const toggleOption = (value) => {
  const newValue = [...props.modelValue]
  const index = newValue.indexOf(value)
  
  if (index === -1) {
    newValue.push(value)
  } else {
    newValue.splice(index, 1)
  }
  
  emit('update:modelValue', newValue)
}

// 全选/取消全选
const toggleAll = (event) => {
  if (event.target.checked) {
    // 全选
    emit('update:modelValue', props.options.map(option => option.value))
  } else {
    // 取消全选
    emit('update:modelValue', [])
  }
}

// 监听 DOM 更新后设置 indeterminate 属性
watch(() => indeterminate.value, (val) => {
  setTimeout(() => {
    const checkbox = document.getElementById(`${props.id}-all`)
    if (checkbox) {
      checkbox.indeterminate = val
    }
  }, 0)
}, { immediate: true })
</script>

<style lang="scss" scoped>
.tv-checkbox-group {
  display: flex;
  flex-direction: column;
  gap: 12px;
  
  .select-all {
    margin-bottom: 5px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(0, 195, 255, 0.2);
    
    label {
      font-weight: bold;
      letter-spacing: 1.5px;
    }
  }
  
  .checkbox-items {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-left: 10px;
  }
}

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
      
      &::after {
        content: '';
        position: absolute;
        top: 2px;
        left: 6px;
        width: 5px;
        height: 10px;
        border-right: 2px solid #000;
        border-bottom: 2px solid #000;
        transform: rotate(45deg);
        z-index: 2;
      }
      
      & + label {
        color: #00c3ff;
        text-shadow: 0 0 8px rgba(0, 195, 255, 0.6);
      }
    }
    
    &:hover:not(:checked) {
      border-color: #00c3ff;
      box-shadow: 0 0 8px rgba(0, 195, 255, 0.3);
    }
  }
  
  label {
    font-size: 0.9rem;
    color: rgba(0, 195, 255, 0.8);
    cursor: pointer;
    text-shadow: 0 0 5px rgba(0, 195, 255, 0.4);
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    
    &:hover {
      color: #00c3ff;
    }
  }
}

@keyframes checkboxGlow {
  0% { transform: translate(-50%, -50%) rotate(45deg); }
  100% { transform: translate(150%, 150%) rotate(45deg); }
}
</style>