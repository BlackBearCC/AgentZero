<template>
  <div class="page-background">
    <div class="evaluation-container">
      <div class="control-panel">
        <div class="file-selector">
          <label class="upload-btn">
            <input type="file" @change="handleFileUpload" accept=".csv,.xlsx" />
            <span>选择文件</span>
          </label>
          <select v-model="selectedEvalType" class="eval-select">
            <option value="dialogue">对话质量评估</option>
            <option value="memory">记忆相关性评估</option>
          </select>
          <button @click="startEvaluation" :disabled="isEvaluating" class="start-btn">
            {{ isEvaluating ? '评估中...' : '开始评估' }}
          </button>
        </div>

        <div class="progress" v-if="isEvaluating">
          <div class="progress-text">进度: {{ processed }}/{{ total }}</div>
          <div class="progress-bar">
            <div :style="progressStyle"></div>
          </div>
        </div>
      </div>

      <!-- AI工作窗口 -->
      <div class="chat-container">
        <div class="chat-window" ref="chatWindow">
          <!-- 系统消息 -->
          <div class="message system-message" v-if="systemMessage">
            {{ systemMessage }}
          </div>
          
          <!-- 评估消息 -->
          <div class="message ai-message" v-if="evaluationText">
            <div class="message-header">
              <span class="ai-badge">AI</span>
              <span>评估结果</span>
            </div>
            <div class="message-content">{{ evaluationText }}</div>
          </div>

          <!-- 正在输入提示 -->
          <div class="typing-indicator" v-if="isEvaluating">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000' // 修改为你的后端地址

const selectedFile = ref(null)
const selectedEvalType = ref('dialogue')
const isEvaluating = ref(false)
const results = ref([])
const processed = ref(0)
const total = ref(0)
const chatWindow = ref(null)
const systemMessage = ref('我是评估助手，请上传文件开始评估。')
const evaluationText = ref('')

// 自动滚动到底部
const scrollToBottom = () => {
  if (chatWindow.value) {
    setTimeout(() => {
      chatWindow.value.scrollTop = chatWindow.value.scrollHeight
    }, 50)
  }
}

// 监听结果变化，自动滚动
watch(results, () => {
  scrollToBottom()
}, { deep: true })

const progressStyle = computed(() => ({
  width: `${(processed.value / total.value) * 100}%`
}))

const handleFileUpload = (event) => {
  selectedFile.value = event.target.files[0]
  systemMessage.value = `已选择文件: ${selectedFile.value.name}`
}

const startEvaluation = async () => {
  if (!selectedFile.value) return
  
  const formData = new FormData()
  formData.append('file', selectedFile.value)
  formData.append('eval_type', selectedEvalType.value)
  formData.append('user_id', 'default')
  
  try {
    isEvaluating.value = true
    evaluationText.value = ''
    systemMessage.value = '开始评估...'
    
    const response = await fetch(`${API_BASE_URL}/api/v1/evaluate/stream`, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const text = decoder.decode(value)
      const lines = text.split('\n')
      
      for (const line of lines) {
        if (!line.trim()) continue
        
        try {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(5))
            
            if (data.result) {
              evaluationText.value += data.result 
              scrollToBottom()
            }
          } else if (line.startsWith('event: error')) {
            const errorLine = lines[lines.indexOf(line) + 1]
            if (errorLine && errorLine.startsWith('data: ')) {
              const error = JSON.parse(errorLine.slice(5))
              systemMessage.value = `评估错误: ${error.error}`
            }
          }
        } catch (e) {
          console.error('解析SSE数据失败:', e)
        }
      }
      
      processed.value += 1
      total.value = Math.max(total.value, processed.value)
    }

  } catch (error) {
    console.error('评估失败:', error)
    systemMessage.value = `评估失败: ${error.message}`
  } finally {
    isEvaluating.value = false
    systemMessage.value = '评估完成！'
  }
}
</script>

<style scoped>
.page-background {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(to bottom, #1a1a2e, #16213e);
  overflow-y: auto;
  min-height: 100vh;
}

.evaluation-container {
  width: 100%;
  min-height: 100vh;
  padding: 2rem;
  display: flex;
  flex-direction: column;
  gap: 2rem;
  color: #e6e6e6;
}

.control-panel {
  width: 100%;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 16px;
  padding: 1.5rem;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.file-selector {
  display: flex;
  gap: 1rem;
  align-items: center;
  justify-content: center;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}

.upload-btn {
  position: relative;
  background: linear-gradient(45deg, #2196f3, #00bcd4);
  padding: 0.8rem 1.5rem;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  border: none;
  color: white;
  font-weight: 500;
}

.upload-btn input[type="file"] {
  position: absolute;
  opacity: 0;
  width: 100%;
  height: 100%;
  cursor: pointer;
}

.upload-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
}

.eval-select {
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  padding: 0.8rem 1.5rem;
  border-radius: 8px;
  color: white;
  outline: none;
}

.start-btn {
  background: linear-gradient(45deg, #7c4dff, #448aff);
  padding: 0.8rem 1.5rem;
  border-radius: 8px;
  border: none;
  color: white;
  cursor: pointer;
  transition: all 0.3s ease;
}

.start-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(124, 77, 255, 0.3);
}

.start-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.progress {
  margin-top: 1rem;
}

.progress-text {
  text-align: center;
  margin-bottom: 0.5rem;
  color: #a0a0a0;
}

.progress-bar {
  height: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  overflow: hidden;
}

.progress-bar div {
  height: 100%;
  background: linear-gradient(90deg, #00bcd4, #2196f3);
  transition: width 0.3s ease;
}

.chat-container {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 1rem;
  width: 100%;
}

.chat-window {
  width: 100%;
  height: calc(100vh - 250px);
  background: rgba(255, 255, 255, 0.05);
  border-radius: 16px;
  padding: 1.5rem;
  overflow-y: auto;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.message {
  padding: 1rem;
  border-radius: 12px;
  max-width: 100%;
}

.system-message {
  background: rgba(255, 255, 255, 0.1);
  text-align: center;
  font-size: 0.9rem;
  color: #a0a0a0;
}

.ai-message {
  background: linear-gradient(135deg, rgba(124, 77, 255, 0.2), rgba(33, 150, 243, 0.2));
  border: 1px solid rgba(124, 77, 255, 0.3);
}

.message-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.8rem;
  font-weight: 500;
}

.ai-badge {
  background: linear-gradient(45deg, #7c4dff, #448aff);
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
}

.message-content {
  line-height: 1.6;
  white-space: pre-wrap;
}

.typing-indicator {
  display: flex;
  gap: 0.4rem;
  justify-content: center;
  padding: 1rem;
}

.dot {
  width: 8px;
  height: 8px;
  background: #a0a0a0;
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out;
}

.dot:nth-child(1) { animation-delay: -0.32s; }
.dot:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}

::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.2);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.3);
}

@media (max-width: 768px) {
  .evaluation-container {
    padding: 1rem;
  }

  .file-selector {
    flex-direction: column;
    gap: 0.8rem;
  }

  .upload-btn, .eval-select, .start-btn {
    width: 100%;
  }

  .chat-window {
    height: calc(100vh - 300px);
  }
}

/* 确保滚动条样式应用到背景容器 */
.page-background::-webkit-scrollbar {
  width: 6px;
}

.page-background::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
}

.page-background::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.2);
  border-radius: 3px;
}

.page-background::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.3);
}
</style> 