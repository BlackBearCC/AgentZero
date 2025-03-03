<template>
  <div class="batch-dialogue">
    <div v-if="!dialogueResults.length && !isProcessing" class="empty-state">
      <div class="tv-logo">BATCH DIALOGUE</div>
      <div class="channel-info">频道 2</div>
      <div class="instruction-text">请使用左侧控制面板上传对话文件并开始处理</div>
    </div>
    
    <div v-else-if="isProcessing" class="processing-state">
      <div class="tv-logo">BATCH DIALOGUE</div>
      <div class="processing-message">正在处理对话 {{ processedCount }}/{{ totalCount }}</div>
      <div class="processing-animation"></div>
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: `${(processedCount / Math.max(totalCount, 1)) * 100}%` }"></div>
      </div>
    </div>
    
    <div v-else-if="dialogueResults.length > 0" class="result-state">
      <div class="dialogue-header">
        <h2>批量对话结果</h2>
        <div class="dialogue-actions">
          <button @click="exportResults" class="crt-button">
            <span class="button-text">[ 导出对话结果 ]</span>
          </button>
        </div>
      </div>
      
      <div class="dialogue-list">
        <div v-for="(dialogue, index) in dialogueResults" :key="index" class="dialogue-item">
          <div class="dialogue-number">#{{ index + 1 }}</div>
          <div class="dialogue-content">
            <div class="user-message">{{ dialogue.user }}</div>
            <div class="ai-response">{{ dialogue.ai }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, defineEmits } from 'vue'

const emit = defineEmits(['scanning:start', 'scanning:stop'])

// 批量对话相关状态
const isProcessing = ref(false)
const processedCount = ref(0)
const totalCount = ref(0)
const dialogueResults = ref([])

// 导出对话结果
const exportResults = () => {
  if (dialogueResults.value.length === 0) return
  
  // 准备CSV数据
  const headers = ['序号', '用户输入', 'AI回复']
  const rows = dialogueResults.value.map((dialogue, index) => [
    index + 1,
    dialogue.user.replace(/"/g, '""'),
    dialogue.ai.replace(/"/g, '""')
  ])
  
  // 生成CSV内容
  const csvContent = [
    headers.join(','),
    ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
  ].join('\n')
  
  // 创建下载链接
  const blob = new Blob([new Uint8Array([0xEF, 0xBB, 0xBF]), csvContent], { 
    type: 'text/csv;charset=utf-8'
  })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = `对话结果_${new Date().toISOString().slice(0,10)}.csv`
  link.click()
  URL.revokeObjectURL(link.href)
}

// 暴露给父组件的方法
defineExpose({
  startProcessing(file, model) {
    isProcessing.value = true
    processedCount.value = 0
    totalCount.value = 10 // 模拟10条对话
    dialogueResults.value = []
    emit('scanning:start')
    
    // 模拟处理进度
    const processInterval = setInterval(() => {
      if (processedCount.value < totalCount.value) {
        processedCount.value++
        
        // 添加模拟结果
        dialogueResults.value.push({
          user: `这是用户的第${processedCount.value}条测试消息`,
          ai: `这是AI对第${processedCount.value}条消息的回复。在实际应用中，这里会显示真实的AI回复内容。`
        })
      } else {
        clearInterval(processInterval)
        setTimeout(() => {
          isProcessing.value = false
          emit('scanning:stop')
        }, 500)
      }
    }, 500)
  },
  
  reset() {
    isProcessing.value = false
    processedCount.value = 0
    totalCount.value = 0
    dialogueResults.value = []
  }
})
</script>

<style scoped>
.batch-dialogue {
  height: 100%;
  padding: 20px;
  overflow-y: auto;
  color: #e0e0e0;
}

.empty-state, .processing-state {
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
}

.tv-logo {
  font-size: 2rem;
  font-weight: bold;
  color: #44ff44;
  text-shadow: 0 0 15px rgba(68, 255, 68, 0.7);
  letter-spacing: 3px;
  margin-bottom: 2rem;
}

.instruction-text {
  font-size: 1.2rem;
  color: #a0a0a0;
  max-width: 80%;
  line-height: 1.6;
}

.channel-info {
  position: absolute;
  top: 10px;
  right: 10px;
  background: rgba(0, 0, 0, 0.7);
  color: #44ff44;
  padding: 0.3rem 0.6rem;
  border-radius: 3px;
  font-size: 0.8rem;
  border: 1px solid rgba(68, 255, 68, 0.3);
}

.processing-message {
  font-size: 1.2rem;
  color: #e0e0e0;
  margin-bottom: 2rem;
}

.processing-animation {
  width: 100px;
  height: 100px;
  border: 5px solid rgba(68, 255, 68, 0.3);
  border-top: 5px solid #44ff44;
  border-radius: 50%;
  animation: spin 1.5s linear infinite;
  margin-bottom: 2rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.progress-bar {
  width: 80%;
  height: 15px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 7px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(to right, #44ff44, #88ff88);
  border-radius: 7px;
  transition: width 0.3s ease;
}

.result-state {
  height: 100%;
  overflow-y: auto;
}

.dialogue-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.dialogue-header h2 {
  color: #44ff44;
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.5);
}

.crt-button {
  background: #2a2a3a;
  border: 1px solid #3a3a4a;
  color: #e0e0e0;
  padding: 0.5rem 1rem;
  border-radius: 3px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.crt-button:hover {
  background: #3a3a4a;
  border-color: #44ff44;
  color: #44ff44;
  text-shadow: 0 0 5px rgba(68, 255, 68, 0.5);
}

.dialogue-list {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.dialogue-item {
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 10px;
  padding: 1rem;
  display: flex;
  gap: 1rem;
}

.dialogue-number {
  font-size: 1.2rem;
  font-weight: bold;
  color: #44ff44;
  min-width: 40px;
}

.dialogue-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.user-message {
  background: rgba(255, 255, 255, 0.1);
  padding: 0.8rem;
  border-radius: 5px;
  border-left: 3px solid #8a8a9a;
}

.ai-response {
  background: rgba(68, 255, 68, 0.1);
  padding: 0.8rem;
  border-radius: 5px;
  border-left: 3px solid #44ff44;
}
</style> 