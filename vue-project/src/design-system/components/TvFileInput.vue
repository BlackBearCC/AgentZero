<template>
  <div class="tv-file-input">
    <input 
      type="file" 
      :id="id" 
      @change="handleFileChange" 
      :accept="accept"
      class="file-input"
    />
    <label :for="id">
      <TvButton>
        <slot>[ 选择文件 ]</slot>
      </TvButton>
    </label>
    <div v-if="showFileName" class="file-name">
      {{ fileName || '未选择文件' }}
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import TvButton from './TvButton.vue'

const props = defineProps({
  id: {
    type: String,
    default: () => `file-${Math.random().toString(36).substring(2, 9)}`
  },
  accept: {
    type: String,
    default: '*'
  },
  showFileName: {
    type: Boolean,
    default: true
  }
})

const fileName = ref('')
const emit = defineEmits(['file-change'])

const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (file) {
    fileName.value = file.name
    emit('file-change', file)
  }
}
</script>

<style lang="scss" scoped>
.tv-file-input {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  width: 100%;
  
  .file-input {
    display: none;
  }
  
  .file-name {
    margin-top: 5px;
    font-family: 'Share Tech Mono', monospace;
    color: #00c3ff;
    text-align: center;
    padding: 8px 10px;
    background: rgba(0, 20, 40, 0.5);
    border: 1px solid rgba(0, 195, 255, 0.3);
    border-radius: 4px;
    text-shadow: 0 0 5px rgba(0, 195, 255, 0.4);
    width: 100%;
    position: relative;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    transition: all 0.3s ease;
    
    &::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(0, 195, 255, 0.1) 50%,
        transparent 100%
      );
      transform: translateX(-100%);
      animation: shimmer 2s infinite;
      z-index: -1;
    }
    
    &:hover {
      background: rgba(0, 30, 60, 0.6);
      box-shadow: 0 0 10px rgba(0, 195, 255, 0.3);
    }
  }
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}
</style>