<template>
  <div class="evaluation-container">
    <div class="file-selector">
      <input type="file" @change="handleFileUpload" accept=".csv,.xlsx" />
      <select v-model="selectedEvalType">
        <option value="dialogue">对话质量评估</option>
        <option value="memory">记忆相关性评估</option>
      </select>
      <button @click="startEvaluation" :disabled="isEvaluating">
        {{ isEvaluating ? '评估中...' : '开始评估' }}
      </button>
    </div>

    <div class="progress" v-if="isEvaluating">
      进度: {{ processed }}/{{ total }}
      <div class="progress-bar">
        <div :style="progressStyle"></div>
      </div>
    </div>

    <!-- AI工作窗口 -->
    <div class="chat-window" ref="chatWindow">
      <!-- 系统消息 -->
      <div class="message system-message" v-if="systemMessage">
        {{ systemMessage }}
      </div>
      
      <!-- 评估消息 -->
      <div class="message ai-message" v-if="evaluationText">
        <div class="message-header">AI评估结果</div>
        <div class="message-content typing-effect">{{ evaluationText }}</div>
      </div>

      <!-- 正在输入提示 -->
      <div class="typing-indicator" v-if="isEvaluating">
        AI正在评估...
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
              evaluationText.value += data.result + '\n\n'
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
.evaluation-container {
  max-width: 800px;
  margin: 20px auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.file-selector {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.progress-bar {
  height: 20px;
  background: #eee;
  border-radius: 10px;
  overflow: hidden;
}

.progress-bar div {
  height: 100%;
  background: #42b983;
  transition: width 0.3s ease;
}

.chat-window {
  height: 600px;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 20px;
  overflow-y: auto;
  background: #f9f9f9;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.message-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.message {
  max-width: 85%;
  padding: 12px;
  border-radius: 8px;
  margin: 4px 0;
}

.system-message {
  text-align: center;
  color: #666;
  background: #e8e8e8;
  padding: 8px;
  margin: 8px auto;
  border-radius: 16px;
  font-size: 0.9em;
}

.user-message {
  background: #e8e8e8;
  margin-right: auto;
}

.ai-message {
  background: #42b983;
  color: white;
  margin-left: auto;
}

.message-header {
  font-weight: bold;
  margin-bottom: 8px;
  font-size: 0.9em;
}

.message-content {
  line-height: 1.5;
}

.typing-effect {
  white-space: pre-wrap;
  font-family: monospace;
  border-right: 2px solid #42b983;
  animation: cursor-blink 0.7s infinite;
}

@keyframes cursor-blink {
  0%, 100% { border-color: transparent; }
  50% { border-color: #42b983; }
}

.typing-indicator {
  color: #666;
  font-style: italic;
  text-align: center;
  padding: 8px;
}
</style> 