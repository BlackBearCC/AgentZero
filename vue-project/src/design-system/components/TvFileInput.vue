<template>
  <div 
    class="tv-file-input" 
    :class="{ 'drag-over': isDragging, 'has-file': fileName }"
    @dragover.prevent="handleDragOver"
    @dragleave.prevent="handleDragLeave"
    @drop.prevent="handleDrop"
  >
    <input 
      type="file" 
      :id="id" 
      @change="handleFileChange" 
      :accept="accept"
      class="file-input"
    />
    <label :for="id" class="file-drop-area">
      <div class="upload-icon" v-if="!fileName">
        <div class="upload-arrow"></div>
      </div>
      <div class="file-info" v-if="fileName">
        <div class="file-type-icon"></div>
        <div class="file-name-display">{{ fileName }}</div>
        <div class="clear-button" @click.stop="clearFile">×</div>
      </div>
      <div class="upload-text" v-else>
        <div class="primary-text">{{ primaryText }}</div>
        <div class="secondary-text">{{ secondaryText }}</div>
      </div>
      <div class="scan-line"></div>
    </label>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  id: {
    type: String,
    default: () => `file-${Math.random().toString(36).substring(2, 9)}`
  },
  accept: {
    type: String,
    default: '*'
  },
  primaryText: {
    type: String,
    default: '拖拽文件到此处'
  },
  secondaryText: {
    type: String,
    default: '或点击选择文件'
  }
})

const fileName = ref('')
const isDragging = ref(false)
const emit = defineEmits(['file-change'])

const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (file) {
    fileName.value = file.name
    emit('file-change', file)
  }
}

const handleDragOver = () => {
  isDragging.value = true
}

const handleDragLeave = () => {
  isDragging.value = false
}

const handleDrop = (event) => {
  isDragging.value = false
  const file = event.dataTransfer.files[0]
  if (file) {
    fileName.value = file.name
    emit('file-change', file)
    
    // 更新文件输入元素，保持一致性
    const fileInput = document.getElementById(props.id)
    if (fileInput) {
      // 创建一个新的 DataTransfer 对象
      const dataTransfer = new DataTransfer()
      dataTransfer.items.add(file)
      fileInput.files = dataTransfer.files
    }
  }
}

const clearFile = () => {
  fileName.value = ''
  // 重置文件输入
  const fileInput = document.getElementById(props.id)
  if (fileInput) fileInput.value = ''
  emit('file-change', null)
}
</script>

<style lang="scss" scoped>
.tv-file-input {
  width: 100%;
  position: relative;
  
  .file-input {
    display: none;
  }
  
  .file-drop-area {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 120px;
    padding: 20px;
    background: rgba(0, 20, 40, 0.6);
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 15px rgba(0, 195, 255, 0.1), inset 0 0 20px rgba(0, 195, 255, 0.05);
    
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
  }
  
  .upload-icon {
    width: 60px;
    height: 60px;
    background: rgba(0, 20, 40, 0.7);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 15px;
    position: relative;
    box-shadow: 0 0 15px rgba(0, 195, 255, 0.2);
    animation: pulse-icon 3s infinite alternate;
    
    &::before {
      content: '';
      position: absolute;
      top: -5px;
      left: -5px;
      right: -5px;
      bottom: -5px;
      border-radius: 50%;
      background: radial-gradient(circle at center, rgba(0, 195, 255, 0.2) 0%, transparent 70%);
      animation: pulse-halo 2s infinite;
      z-index: -1;
    }
    
    .upload-arrow {
      position: relative;
      width: 24px;
      height: 24px;
      
      &::before {
        content: '';
        position: absolute;
        width: 16px;
        height: 2px;
        background: #00c3ff;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        box-shadow: 0 0 5px rgba(0, 195, 255, 0.8);
      }
      
      &::after {
        content: '';
        position: absolute;
        width: 2px;
        height: 16px;
        background: #00c3ff;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        box-shadow: 0 0 5px rgba(0, 195, 255, 0.8);
      }
    }
  }
  
  .file-type-icon {
    width: 24px;
    height: 30px;
    position: relative;
    margin-right: 10px;
    background: rgba(0, 20, 40, 0.7);
    border: 1px solid rgba(0, 195, 255, 0.5);
    box-shadow: 0 0 8px rgba(0, 195, 255, 0.3);
    
    &::before {
      content: '';
      position: absolute;
      top: 6px;
      left: 6px;
      right: 6px;
      height: 2px;
      background: rgba(0, 195, 255, 0.7);
      box-shadow: 0 0 4px rgba(0, 195, 255, 0.5);
    }
    
    &::after {
      content: '';
      position: absolute;
      top: 12px;
      left: 6px;
      right: 10px;
      height: 2px;
      background: rgba(0, 195, 255, 0.7);
      box-shadow: 0 0 4px rgba(0, 195, 255, 0.5);
    }
  }
  
  &.drag-over {
    .file-drop-area {
      background: rgba(0, 30, 60, 0.7);
      box-shadow: 0 0 25px rgba(0, 195, 255, 0.3), inset 0 0 30px rgba(0, 195, 255, 0.15);
      
      &::before {
        animation: pulse-glow-fast 1s infinite alternate;
      }
      
      &::after {
        border-color: rgba(0, 195, 255, 0.4);
        animation: pulse-border-fast 1s infinite alternate;
      }
    }
    
    .upload-icon {
      animation: pulse-icon-fast 1s infinite alternate;
      
      &::before {
        animation: pulse-halo-fast 1s infinite;
      }
    }
    
    .scan-line {
      opacity: 1;
      animation: scan-fast 1.5s ease-in-out infinite;
    }
  }
  
  &.has-file {
    .file-drop-area {
      background: rgba(0, 30, 60, 0.6);
      box-shadow: 0 0 20px rgba(0, 195, 255, 0.2), inset 0 0 25px rgba(0, 195, 255, 0.1);
    }
  }
  
  &:hover {
    .file-drop-area {
      background: rgba(0, 25, 50, 0.7);
      box-shadow: 0 0 20px rgba(0, 195, 255, 0.2), inset 0 0 25px rgba(0, 195, 255, 0.1);
    }
  }
}

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

@keyframes scan {
  0%, 100% { 
    transform: translateY(-100%);
    opacity: 0;
  }
  50% { 
    transform: translateY(1000%);
    opacity: 0.8;
  }
}

@keyframes scan-fast {
  0%, 100% { 
    transform: translateY(-100%);
    opacity: 0.5;
  }
  50% { 
    transform: translateY(1000%);
    opacity: 1;
  }
}
</style>