<template>
    <div class="tv-checkbox-group" :class="layout">
      <div v-if="showSelectAll" class="select-all-container">
        <div 
          class="select-all-button" 
          :class="{ 'all-selected': allSelected, 'partial-selected': indeterminate }"
          @click="toggleAll"
          @mouseenter="showTooltip = true"
          @mouseleave="showTooltip = false"
        >
          <div class="select-all-icon">
            <div class="icon-inner"></div>
          </div>
          <div class="select-all-label">{{ selectAllLabel }}</div>
          <div class="select-all-status">
            <span v-if="allSelected">全部已选</span>
            <span v-else-if="indeterminate">部分已选</span>
            <span v-else>未选择</span>
          </div>
          <div class="select-all-effect"></div>
        </div>
        
        <div class="select-all-tooltip" v-if="showTooltip">
          <div class="tooltip-content">
            <div class="tooltip-title">全选控制</div>
            <div class="tooltip-description">点击可{{ allSelected ? '取消选择' : '选择' }}所有选项</div>
            <div class="tooltip-status">
              当前已选: {{ props.modelValue.length }}/{{ props.options.length }}
            </div>
          </div>
          <div class="tooltip-arrow"></div>
        </div>
      </div>
      
      <div class="checkbox-items">
        <div 
          v-for="(option, index) in options" 
          :key="index"
          class="tv-checkbox"
          @mouseenter="activeTooltip = option.value"
          @mouseleave="activeTooltip = null"
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
          
          <!-- 选项悬停提示 -->
          <div class="option-tooltip" v-if="activeTooltip === option.value && descriptions && descriptions[option.value]">
            <div class="tooltip-content">
              <div class="tooltip-description">{{ descriptions[option.value] }}</div>
            </div>
            <div class="tooltip-arrow"></div>
          </div>
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
    },
    layout: {
      type: String,
      default: '' // '', 'grid-layout', 'compact-layout'
    },
    descriptions: {
      type: Object,
      default: () => ({})
    }
  })
  
  const emit = defineEmits(['update:modelValue'])
  const showTooltip = ref(false)
  const activeTooltip = ref(null)
  
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
  const toggleAll = () => {
    if (allSelected.value) {
      // 取消全选
      emit('update:modelValue', [])
    } else {
      // 全选
      emit('update:modelValue', props.options.map(option => option.value))
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
    gap: 16px;
    
    .select-all-container {
      position: relative;
      margin-bottom: 10px;
    }
    
    .select-all-button {
      display: flex;
      align-items: center;
      padding: 10px 15px;
      background: rgba(0, 20, 40, 0.8);
      border: 1px solid rgba(0, 195, 255, 0.3);
      border-radius: 4px;
      cursor: pointer;
      position: relative;
      overflow: hidden;
      transition: all 0.3s ease;
      box-shadow: 0 0 15px rgba(0, 195, 255, 0.1), inset 0 0 10px rgba(0, 195, 255, 0.05);
      
      &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 195, 255, 0.5), transparent);
        opacity: 0.7;
      }
      
      &::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 195, 255, 0.3), transparent);
        opacity: 0.5;
      }
      
      .select-all-icon {
        width: 24px;
        height: 24px;
        border: 2px solid rgba(0, 195, 255, 0.6);
        border-radius: 4px;
        position: relative;
        margin-right: 12px;
        background: rgba(0, 20, 40, 0.9);
        transition: all 0.3s ease;
        
        .icon-inner {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%) scale(0);
          width: 12px;
          height: 12px;
          background: #00c3ff;
          border-radius: 2px;
          box-shadow: 0 0 8px rgba(0, 195, 255, 0.8);
          transition: all 0.3s ease;
        }
      }
      
      .select-all-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1rem;
        color: #00c3ff;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: bold;
        text-shadow: 0 0 8px rgba(0, 195, 255, 0.4);
        margin-right: auto;
      }
      
      .select-all-status {
        font-size: 0.8rem;
        color: rgba(0, 195, 255, 0.7);
        margin-left: 10px;
        opacity: 0.8;
        transition: all 0.3s ease;
      }
      
      .select-all-effect {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at center, rgba(0, 195, 255, 0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.5s ease;
        z-index: -1;
      }
      
      &:hover {
        background: rgba(0, 30, 60, 0.8);
        border-color: rgba(0, 195, 255, 0.5);
        box-shadow: 0 0 20px rgba(0, 195, 255, 0.2), inset 0 0 15px rgba(0, 195, 255, 0.1);
        
        .select-all-icon {
          border-color: rgba(0, 195, 255, 0.8);
          box-shadow: 0 0 10px rgba(0, 195, 255, 0.3);
        }
        
        .select-all-status {
          color: rgba(0, 195, 255, 0.9);
          opacity: 1;
        }
        
        .select-all-effect {
          opacity: 1;
          animation: pulse-radial 2s infinite;
        }
      }
      
      &.all-selected {
        background: rgba(0, 40, 70, 0.8);
        border-color: rgba(0, 195, 255, 0.6);
        box-shadow: 0 0 20px rgba(0, 195, 255, 0.2), inset 0 0 15px rgba(0, 195, 255, 0.15);
        
        .select-all-icon {
          border-color: rgba(0, 195, 255, 0.8);
          
          .icon-inner {
            transform: translate(-50%, -50%) scale(1);
          }
        }
        
        .select-all-label {
          color: #00c3ff;
          text-shadow: 0 0 10px rgba(0, 195, 255, 0.6);
        }
      }
      
      &.partial-selected {
        .select-all-icon {
          .icon-inner {
            transform: translate(-50%, -50%) scale(0.6);
            opacity: 0.7;
          }
        }
      }
    }
    
    .select-all-tooltip, .option-tooltip {
      position: absolute;
      background: rgba(0, 20, 40, 0.9);
      border: 1px solid rgba(0, 195, 255, 0.4);
      border-radius: 4px;
      padding: 12px;
      z-index: 10;
      box-shadow: 0 0 20px rgba(0, 0, 0, 0.5), 0 0 15px rgba(0, 195, 255, 0.2);
      animation: tooltip-appear 0.3s ease;
      
      .tooltip-content {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }
      
      .tooltip-title {
        font-family: 'Share Tech Mono', monospace;
        color: #00c3ff;
        font-size: 0.9rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
        text-shadow: 0 0 5px rgba(0, 195, 255, 0.5);
      }
      
      .tooltip-description {
        color: rgba(0, 195, 255, 0.8);
        font-size: 0.8rem;
        line-height: 1.4;
      }
      
      .tooltip-status {
        margin-top: 4px;
        padding-top: 4px;
        border-top: 1px solid rgba(0, 195, 255, 0.2);
        color: rgba(0, 195, 255, 0.7);
        font-size: 0.8rem;
      }
      
      .tooltip-arrow {
        position: absolute;
        width: 12px;
        height: 12px;
        background: rgba(0, 20, 40, 0.9);
        border-left: 1px solid rgba(0, 195, 255, 0.4);
        border-top: 1px solid rgba(0, 195, 255, 0.4);
        transform: rotate(45deg);
      }
    }
    
    .select-all-tooltip {
      top: calc(100% + 10px);
      left: 50%;
      transform: translateX(-50%);
      width: 220px;
      
      .tooltip-arrow {
        top: -6px;
        left: 50%;
        transform: translateX(-50%) rotate(45deg);
      }
    }
    
    .checkbox-items {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding-left: 10px;
      border-left: 1px solid rgba(0, 195, 255, 0.2);
      margin-left: 5px;
    }
    
    .tv-checkbox {
      display: flex;
      align-items: center;
      gap: 10px;
      position: relative;
      
      .option-tooltip {
        top: -10px;
        left: 100%;
        margin-left: 15px;
        width: 200px;
        
        .tooltip-arrow {
          top: 50%;
          left: -6px;
          transform: translateY(-50%) rotate(-45deg);
        }
      }
      
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
        
        &:focus {
          outline: none;
          border-color: #00c3ff;
          box-shadow: 0 0 12px rgba(0, 195, 255, 0.4);
        }
        
        &:disabled {
          opacity: 0.5;
          cursor: not-allowed;
          
          & + label {
            opacity: 0.5;
            cursor: not-allowed;
          }
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
        position: relative;
        padding: 2px 5px;
        
        &::before {
          content: '';
          position: absolute;
          bottom: 0;
          left: 0;
          width: 0;
          height: 1px;
          background: rgba(0, 195, 255, 0.6);
          transition: width 0.3s ease;
        }
        
        &:hover {
          color: #00c3ff;
          text-shadow: 0 0 8px rgba(0, 195, 255, 0.6);
          
          &::before {
            width: 100%;
            box-shadow: 0 0 5px rgba(0, 195, 255, 0.4);
          }
        }
      }
    }
  }


// 网格布局样式优化
.tv-checkbox-group.grid-layout {
  .checkbox-items {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px;
    padding-left: 0;
    border-left: none;
    margin-left: 0;
    margin-top: 15px;
    
    .tv-checkbox {
      background: rgba(0, 20, 40, 0.6);
      border: none;
      border-radius: 4px;
      padding: 15px;
      transition: all 0.3s ease;
      height: 100%;
      position: relative;
      overflow: hidden;
      box-shadow: 0 0 15px rgba(0, 195, 255, 0.1), inset 0 0 20px rgba(0, 195, 255, 0.05);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      
      &::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at center, rgba(0, 195, 255, 0.1) 0%, transparent 70%);
        animation: pulse-glow 3s infinite alternate;
        pointer-events: none;
      }
      
      &::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border: 1px solid rgba(0, 195, 255, 0.2);
        border-radius: 4px;
        box-shadow: 0 0 10px rgba(0, 195, 255, 0.1);
        animation: pulse-border 2s infinite alternate;
        pointer-events: none;
      }
      
      .option-tooltip {
        top: auto;
        bottom: calc(100% + 10px);
        left: 50%;
        transform: translateX(-50%);
        width: 180px;
        
        .tooltip-arrow {
          top: auto;
          bottom: -6px;
          left: 50%;
          transform: translateX(-50%) rotate(-135deg);
        }
      }
      
      input[type="checkbox"] {
        position: relative;
        margin-bottom: 10px;
        width: 28px;
        height: 28px;
        border-radius: 4px;
        border: 2px solid rgba(0, 195, 255, 0.5);
        background: rgba(0, 20, 40, 0.7);
        box-shadow: 0 0 15px rgba(0, 195, 255, 0.2);
        animation: pulse-icon 3s infinite alternate;
        
        &::before {
          content: '';
          position: absolute;
          top: -5px;
          left: -5px;
          right: -5px;
          bottom: -5px;
          border-radius: 4px;
          background: radial-gradient(circle at center, rgba(0, 195, 255, 0.2) 0%, transparent 70%);
          animation: pulse-halo 2s infinite;
          z-index: -1;
        }
        
        &:checked {
          background: rgba(0, 30, 60, 0.8);
          
          &::after {
            content: '';
            position: absolute;
            top: 4px;
            left: 9px;
            width: 6px;
            height: 12px;
            border-right: 2px solid #00c3ff;
            border-bottom: 2px solid #00c3ff;
            transform: rotate(45deg);
            box-shadow: 0 0 8px rgba(0, 195, 255, 0.8);
          }
        }
      }
      
      label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.95rem;
        color: #00c3ff;
        text-align: center;
        text-shadow: 0 0 8px rgba(0, 195, 255, 0.4);
        padding: 0;
        
        &::before {
          display: none;
        }
      }
      
      &:hover {
        background: rgba(0, 25, 50, 0.7);
        box-shadow: 0 0 20px rgba(0, 195, 255, 0.2), inset 0 0 25px rgba(0, 195, 255, 0.1);
        transform: translateY(-2px);
        
        input[type="checkbox"] {
          animation: pulse-icon-fast 1s infinite alternate;
          border-color: rgba(0, 195, 255, 0.8);
          
          &::before {
            animation: pulse-halo-fast 1s infinite;
          }
        }
      }
      
      &.selected {
        background: rgba(0, 30, 60, 0.6);
        box-shadow: 0 0 20px rgba(0, 195, 255, 0.2), inset 0 0 25px rgba(0, 195, 255, 0.1);
      }
    }
  }
}
// 添加与 TvFileInput 相同的动画
@keyframes pulse-glow {
  0% { opacity: 0.3; }
  100% { opacity: 0.7; }
}

@keyframes pulse-glow-fast {
  0% { opacity: 0.5; }
  100% { opacity: 0.9; }
}

@keyframes pulse-border {
  0% { opacity: 0.3; box-shadow: 0 0 5px rgba(0, 195, 255, 0.1); }
  100% { opacity: 0.7; box-shadow: 0 0 15px rgba(0, 195, 255, 0.2); }
}

@keyframes pulse-border-fast {
  0% { opacity: 0.5; box-shadow: 0 0 10px rgba(0, 195, 255, 0.2); }
  100% { opacity: 1; box-shadow: 0 0 20px rgba(0, 195, 255, 0.3); }
}

@keyframes pulse-icon {
  0% { transform: scale(0.95); box-shadow: 0 0 10px rgba(0, 195, 255, 0.1); }
  100% { transform: scale(1); box-shadow: 0 0 20px rgba(0, 195, 255, 0.3); }
}

@keyframes pulse-icon-fast {
  0% { transform: scale(0.95); box-shadow: 0 0 15px rgba(0, 195, 255, 0.2); }
  100% { transform: scale(1.05); box-shadow: 0 0 25px rgba(0, 195, 255, 0.4); }
}

@keyframes pulse-halo {
  0%, 100% { opacity: 0.3; transform: scale(0.9); }
  50% { opacity: 0.7; transform: scale(1.1); }
}

@keyframes pulse-halo-fast {
  0%, 100% { opacity: 0.5; transform: scale(0.9); }
  50% { opacity: 1; transform: scale(1.2); }
}

@keyframes checkboxGlow {
  0% { transform: translate(-50%, -50%) rotate(45deg); }
  100% { transform: translate(150%, 150%) rotate(45deg); }
}

@keyframes pulse-radial {
  0% { opacity: 0.3; }
  50% { opacity: 0.7; }
  100% { opacity: 0.3; }
}

@keyframes tooltip-appear {
  0% { opacity: 0; transform: translateX(-50%) translateY(-10px); }
  100% { opacity: 1; transform: translateX(-50%) translateY(0); }
}

// 添加紧凑布局选项
.tv-checkbox-group.compact-layout {
  .checkbox-items {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    padding-left: 0;
    border-left: none;
    
    .tv-checkbox {
      background: rgba(0, 20, 40, 0.5);
      border: 1px solid rgba(0, 195, 255, 0.2);
      border-radius: 4px;
      padding: 5px 10px;
      margin-right: 5px;
      margin-bottom: 5px;
      
      label {
        font-size: 0.8rem;
      }
      
      .option-tooltip {
        bottom: calc(100% + 10px);
        left: 50%;
        transform: translateX(-50%);
        top: auto;
        
        .tooltip-arrow {
          bottom: -6px;
          left: 50%;
          top: auto;
          transform: translateX(-50%) rotate(-135deg);
        }
      }
    }
  }
}

// 添加动画效果
.tv-checkbox-enter-active,
.tv-checkbox-leave-active {
  transition: all 0.3s ease;
}

.tv-checkbox-enter-from,
.tv-checkbox-leave-to {
  opacity: 0;
  transform: translateY(10px);
}
</style>