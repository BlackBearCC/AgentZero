<template>
  <div class="control-group">
    <div class="control-label">CHANNEL</div>
    <div class="channel-buttons">
      <button 
        v-for="channel in 3" 
        :key="channel"
        @click="changeChannel(channel)" 
        class="control-button channel-btn" 
        :class="{ 'active': activeChannel === channel }"
        :disabled="!isPoweredOn"
      >
        {{ channel }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { storeToRefs } from 'pinia'
import { useEvaluationStore } from '@/stores/evaluation'

const store = useEvaluationStore()
const { activeChannel, isPoweredOn } = storeToRefs(store)

const emit = defineEmits(['channel-change'])

const changeChannel = (channel) => {
  if (!isPoweredOn.value || channel === activeChannel.value) return
  
  // 触发换台效果
  emit('channel-change', {
    oldChannel: activeChannel.value,
    newChannel: channel,
    isChanging: true
  })
  
  // 延迟切换频道，模拟换台过程
  setTimeout(() => {
    store.setActiveChannel(channel)
    
    // 结束换台效果
    setTimeout(() => {
      emit('channel-change', {
        oldChannel: activeChannel.value,
        newChannel: channel,
        isChanging: false
      })
    }, 500)
  }, 1000)
}
</script>

<style scoped>
.control-group {
  margin-bottom: 1rem;
  padding: 0.5rem;
  border: 1px solid rgba(68, 255, 68, 0.3);
  background: rgba(0, 0, 0, 0.3);
  border-radius: 4px;
}

.control-label {
  font-size: 0.8rem;
  color: #44ff44;
  margin-bottom: 0.5rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  text-shadow: 0 0 5px rgba(68, 255, 68, 0.5);
}

.channel-buttons {
  display: flex;
  gap: 0.5rem;
  justify-content: space-between;
}

.channel-btn {
  flex: 1;
  background: none;
  border: 1px solid #44ff44;
  color: #44ff44;
  padding: 0.5rem;
  cursor: pointer;
  transition: all 0.3s ease;
  border-radius: 4px;
  min-width: 40px;
}

.channel-btn:hover:not(:disabled) {
  background: rgba(68, 255, 68, 0.1);
  box-shadow: 0 0 10px rgba(68, 255, 68, 0.2);
}

.channel-btn:active:not(:disabled) {
  transform: scale(0.98);
}

.channel-btn.active {
  background: rgba(68, 255, 68, 0.2);
  box-shadow: 0 0 15px rgba(68, 255, 68, 0.3);
}

.channel-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  border-color: rgba(68, 255, 68, 0.3);
}
</style> 