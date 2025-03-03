<template>
  <div class="batch-dialogue">
    <!-- 未上传文件时的空状态 -->
    <div v-if="!dialogueData && !isProcessing" class="empty-state">
      <div class="tv-logo">BATCH DIALOGUE</div>
      <div class="channel-info">频道 2</div>
      <div class="instruction-text">请上传对话数据文件开始批量处理</div>
      
      <!-- 控制区域整合到屏幕中 -->
      <div class="screen-controls">
        <div class="file-control">
          <input 
            type="file" 
            id="dialogue-file" 
            @change="handleFileChange" 
            accept=".csv,.xlsx"
            class="file-input"
          />
          <label for="dialogue-file" class="tv-button">
            <span class="button-text">[ 选择对话文件 ]</span>
          </label>
          <div class="file-name">{{ fileName || '未选择文件' }}</div>
        </div>
        
        <!-- 对话模型选择 -->
        <div class="model-select-group">
          <div class="model-label">对话模型</div>
          <select v-model="selectedModel" class="model-dropdown">
            <option value="gpt-3.5">GPT-3.5</option>
            <option value="gpt-4">GPT-4</option>
            <option value="claude">Claude</option>
          </select>
        </div>
        
        <!-- 系统提示词设置 -->
        <div class="system-prompt-group">
          <div class="prompt-toggle">
            <input type="checkbox" id="use-system-prompt" v-model="useSystemPrompt" />
            <label for="use-system-prompt">使用系统提示词</label>
          </div>
          <textarea 
            v-if="useSystemPrompt"
            v-model="systemPrompt" 
            class="system-prompt-input" 
            placeholder="输入系统提示词..."
            rows="3"
          ></textarea>
        </div>
      
        <!-- 操作按钮 -->
        <div class="action-buttons">
          <button @click="processDialogues" class="tv-button primary" :disabled="!canProcess">
            <span class="button-text">[ 开始处理 ]</span>
          </button>
        </div>
      </div>
    </div>
    
    <!-- 处理中状态 -->
    <div v-else-if="isProcessing" class="processing-state">
      <div class="tv-logo">BATCH DIALOGUE</div>
      <div class="channel-info">频道 2</div>
      <div class="processing-message">正在处理对话数据...</div>
      <div class="processing-animation"></div>
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: `${processingProgress}%` }"></div>
      </div>
      <div class="progress-text">{{ processingProgress }}%</div>
    </div>
    
    <!-- 处理结果 -->
    <div v-else class="result-state">
      <div class="dialogue-header">
        <h2>批量对话结果</h2>
        <div class="header-actions">
          <button @click="resetProcessor" class="crt-button">重新处理</button>
          <button @click="exportResults" class="crt-button">导出结果</button>
        </div>
      </div>
      
      <div class="dialogue-list">
        <div v-for="(item, index) in dialogueData" :key="index" class="dialogue-item">
          <div class="dialogue-number">#{{ index + 1 }}</div>
          <div class="dialogue-content">
            <div class="user-message">{{ item.user }}</div>
            <div class="ai-response">{{ item.ai }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

// 状态变量
const fileName = ref('')
const dialogueData = ref(null)
const isProcessing = ref(false)
const processingProgress = ref(0)
const selectedModel = ref('gpt-4')
const useSystemPrompt = ref(false)
const systemPrompt = ref('')

// 模拟文件上传处理
const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (file) {
    fileName.value = file.name
    // 在实际应用中，这里应该解析上传的文件
  }
}

// 计算属性：是否可以开始处理
const canProcess = computed(() => {
  return fileName.value && !isProcessing.value
})

// 开始处理对话数据
const processDialogues = () => {
  isProcessing.value = true
  processingProgress.value = 0
  
  // 模拟处理进度
  const interval = setInterval(() => {
    processingProgress.value += 5
    
    if (processingProgress.value >= 100) {
      clearInterval(interval)
      setTimeout(() => {
        // 模拟生成对话数据
        dialogueData.value = generateMockDialogues()
        isProcessing.value = false
      }, 500)
    }
  }, 300)
}

// 重置处理器
const resetProcessor = () => {
  dialogueData.value = null
  fileName.value = ''
  processingProgress.value = 0
}

// 导出结果
const exportResults = () => {
  // 在实际应用中，这里应该实现导出功能
  alert('对话结果已导出')
}

// 模拟生成对话数据
const generateMockDialogues = () => {
  return [
    {
      user: '你好，请介绍一下你自己。',
      ai: '你好！我是一个AI助手，由OpenAI开发。我可以帮助回答问题、提供信息和进行各种对话交流。有什么我可以帮助你的吗？'
    },
    {
      user: '解释一下什么是人工智能？',
      ai: '人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这包括视觉感知、语音识别、决策和语言翻译等。AI系统通过分析大量数据，识别模式，并使用这些见解来做出预测或决策。'
    },
    {
      user: '写一首关于春天的短诗。',
      ai: '春风轻抚绿意浓，\n花开满园香气融。\n雨后彩虹挂碧空，\n生机勃勃迎春中。'
    }
  ]
}
</script>

<style scoped>
.batch-dialogue {
  height: 100%;
  width: 100%;
  display: flex;
  flex-direction: column;
  color: #e0e0e0;
  overflow-y: auto;
}

.empty-state {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
}

.screen-controls {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  width: 60%;
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 10px;
  padding: 2rem;
  margin-top: 2rem;
}

.file-control {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.file-input {
  display: none;
}

.tv-button {
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(68, 255, 68, 0.5);
  color: #44ff44;
  padding: 0.8rem 1.5rem;
  border-radius: 5px;
  cursor: pointer;
  text-align: center;
  transition: all 0.3s ease;
}

.tv-button:hover {
  background: rgba(68, 255, 68, 0.2);
  text-shadow: 0 0 5px rgba(68, 255, 68, 0.7);
}

.tv-button.primary {
  background: rgba(68, 255, 68, 0.2);
}

.file-name {
  font-size: 0.9rem;
  color: #a0a0a0;
  margin-top: 0.5rem;
}

.model-select-group, .system-prompt-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.model-label {
  font-size: 0.9rem;
  color: #a0a0a0;
}

.model-dropdown {
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(68, 255, 68, 0.3);
  color: #e0e0e0;
  padding: 0.6rem;
  border-radius: 5px;
}

.prompt-toggle {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.system-prompt-input {
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(68, 255, 68, 0.3);
  color: #e0e0e0;
  padding: 0.6rem;
  border-radius: 5px;
  resize: vertical;
}

.action-buttons {
  display: flex;
  justify-content: center;
  margin-top: 1rem;
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