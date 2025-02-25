<template>
  <div class="tv-container">
    <!-- 控制面板 -->
    <div class="control-panel">
      <div class="control-group">
        <div class="control-label">INPUT</div>
        <label class="control-button upload-btn">
          <input type="file" @change="handleFileUpload" accept=".csv,.xlsx" />
          <div class="button-face">
            <span class="button-icon">📁</span>
            <span class="button-text">SELECT FILE</span>
          </div>
        </label>
      </div>

      <div class="control-group">
        <div class="control-label">MODE</div>
        <select v-model="selectedEvalType" class="control-button mode-select">
          <option value="dialogue">DIALOGUE</option>
          <option value="memory">MEMORY</option>
        </select>
      </div>

      <div class="control-group">
        <div class="control-label">POWER</div>
        <button 
          @click="startEvaluation" 
          :disabled="isEvaluating || !fieldsConfirmed" 
          class="control-button power-btn"
        >
          <div class="button-face">
            <div class="power-indicator" :class="{ 'active': isEvaluating }"></div>
            <span class="button-text">{{ isEvaluating ? 'RUNNING' : 'START' }}</span>
          </div>
        </button>
      </div>

      <!-- 预留其他功能的控制组 -->
      <div class="control-group">
        <div class="control-label">CHANNEL</div>
        <div class="channel-buttons">
          <button class="control-button channel-btn">1</button>
          <button class="control-button channel-btn">2</button>
          <button class="control-button channel-btn">3</button>
        </div>
      </div>
    </div>

    <!-- 电视屏幕 -->
    <div class="tv-screen">
      <div class="screen-frame">
        <div class="screen-content">
          <div class="chat-window" ref="chatWindow">
            <div class="message system-message" v-if="systemMessage">
              {{ systemMessage }}
            </div>
            <div class="message ai-message" v-if="evaluationText">
              <div class="message-header">
                <span class="ai-badge">AI</span>
                <span>评估结果</span>
              </div>
              <div class="message-content typewriter">
                <pre class="typewriter-text">{{ evaluationText }}<span class="cursor" :class="{ 'blink': !isTyping }">|</span></pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 字段选择模态框 -->
    <div v-if="showFieldSelector" class="field-selector-modal">
      <div class="field-selector-content">
        <div class="modal-header">
          <h3>SELECT FIELDS</h3>
          <div class="field-count">{{ selectedFields.length }}/{{ availableFields.length }}</div>
        </div>
        <div class="field-list">
          <label v-for="field in availableFields" :key="field" class="field-item">
            <input type="checkbox" v-model="selectedFields" :value="field">
            <span class="field-name">{{ field }}</span>
          </label>
        </div>
        <div class="field-selector-actions">
          <button @click="selectAllFields" class="control-button">SELECT ALL</button>
          <button @click="confirmFields" class="control-button confirm-btn">CONFIRM</button>
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
const isTyping = ref(false)
let typingTimeout

const showFieldSelector = ref(false)
const availableFields = ref([])
const selectedFields = ref([])
const fieldsConfirmed = ref(false)

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

const handleFileUpload = async (event) => {
  const file = event.target.files[0]
  if (!file) return
  
  selectedFile.value = file
  systemMessage.value = `已选择文件: ${file.name}`
  
  // 获取文件字段
  const formData = new FormData()
  formData.append('file', file)
  
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/file/columns`, {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) throw new Error('获取字段失败')
    
    const data = await response.json()
    availableFields.value = data.columns
    selectedFields.value = [...data.columns] // 默认全选
    showFieldSelector.value = true
    fieldsConfirmed.value = false
  } catch (error) {
    systemMessage.value = `获取字段失败: ${error.message}`
  }
}

const confirmFields = () => {
  if (selectedFields.value.length === 0) {
    systemMessage.value = '请至少选择一个字段'
    return
  }
  showFieldSelector.value = false
  fieldsConfirmed.value = true
  systemMessage.value = `已选择 ${selectedFields.value.length} 个字段，可以开始评估`
}

const selectAllFields = () => {
  selectedFields.value = [...availableFields.value]
}

// 添加打字声音效果
const playTypeSound = () => {
  const audio = new Audio();
  audio.src = 'data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA/+M4wAAAAAAAAAAAAEluZm8AAAAPAAAAAwAAABQADw8PDw8PDw8PDw8PDw8PDw8PDw8PDw8VFRUVFRUVFRUVFRUVFRUVFRUVFRUVFR4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCT/wAARCAAIAAgDASIAAhEBAxEB/8QAWQABAQEAAAAAAAAAAAAAAAAAAAIEAQEBAQEAAAAAAAAAAAAAAAAAAgEF/8QAFwEBAQEBAAAAAAAAAAAAAAAAAAECA//EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AKpRqNQBX//Z';
  audio.volume = 0.05;
  audio.play().catch(() => {});
};

const startEvaluation = async () => {
  if (!selectedFile.value || !fieldsConfirmed.value) return
  
  const formData = new FormData()
  formData.append('file', selectedFile.value)
  formData.append('eval_type', selectedEvalType.value)
  formData.append('user_id', 'default')
  formData.append('selected_fields', JSON.stringify(selectedFields.value))
  
  try {
    isEvaluating.value = true
    evaluationText.value = ''
    systemMessage.value = '开始评估...'
    processed.value = 0
    total.value = 0
    
    const response = await fetch(`${API_BASE_URL}/api/v1/evaluate/stream`, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    let currentEvaluation = {
      index: null,
      content: '',
      originalData: ''
    }

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value)
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      
      for (const line of lines) {
        if (!line.trim() || !line.startsWith('data: ')) continue
        
        try {
          const data = JSON.parse(line.slice(5))
          
          if (data.total) {
            total.value = data.total
            continue
          }

          switch (data.type) {
            case 'start':
              if (currentEvaluation.index !== null && evaluationText.value) {
                evaluationText.value += '\n---\n'
              }
              currentEvaluation = {
                index: data.index,
                content: '',
                originalData: data.original_data
              }
              evaluationText.value += `评估项 ${data.index}:\n原始数据:\n${data.original_data}\n\n评估结果:\n`
              break

            case 'chunk':
              isTyping.value = true
              currentEvaluation.content += data.content
              evaluationText.value = evaluationText.value.split(`评估项 ${data.index}:`)[0] + 
                                   `评估项 ${data.index}:\n原始数据:\n${currentEvaluation.originalData}\n\n评估结果:\n${currentEvaluation.content}`
              playTypeSound()
              scrollToBottom()
              
              // 重置打字状态的计时器
              clearTimeout(typingTimeout)
              typingTimeout = setTimeout(() => {
                isTyping.value = false
              }, 100)
              break

            case 'end':
              processed.value = data.index
              break

            case 'error':
              systemMessage.value = `评估项 ${data.index} 错误: ${data.error}`
              break
          }
        } catch (e) {
          console.error('解析SSE数据失败:', e)
        }
      }
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
.tv-container {
  display: flex;
  gap: 2rem;
  padding: 2rem;
  min-height: 100vh;
  width: 100vw;
  background: #1a1a2e;
  box-sizing: border-box;
  position: fixed;
  top: 0;
  left: 0;
  overflow: hidden;
}

.control-panel {
  width: 280px;
  background: #2a2a3a;
  padding: 1.5rem;
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  gap: 2rem;
  box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
  border: 2px solid #3a3a4a;
  height: calc(100vh - 4rem);
  overflow-y: auto;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.control-label {
  color: #8a8a9a;
  font-size: 0.8rem;
  letter-spacing: 2px;
  text-transform: uppercase;
}

.control-button {
  background: #3a3a4a;
  border: none;
  border-radius: 4px;
  padding: 0.8rem;
  color: #fff;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.control-button:hover {
  background: #4a4a5a;
}

.control-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.button-face {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
}

.power-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #ff4444;
  transition: all 0.3s ease;
}

.power-indicator.active {
  background: #44ff44;
  box-shadow: 0 0 10px #44ff44;
}

.tv-screen {
  flex: 1;
  background: #000;
  border-radius: 20px;
  padding: 20px;
  position: relative;
  overflow: hidden;
  height: calc(100vh - 4rem);
  min-width: 0;
}

.screen-frame {
  background: linear-gradient(45deg, #1a1a2e, #2a2a3a);
  border-radius: 15px;
  padding: 15px;
  height: 100%;
  box-shadow: inset 0 0 50px rgba(0,0,0,0.5);
}

.screen-content {
  background: #000;
  border-radius: 10px;
  height: 100%;
  overflow: hidden;
  position: relative;
}

.screen-content::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    rgba(255,255,255,0.1) 50%,
    rgba(0,0,0,0.1) 50%
  );
  background-size: 100% 4px;
  pointer-events: none;
  animation: scanline 10s linear infinite;
}

@keyframes scanline {
  0% { transform: translateY(0); }
  100% { transform: translateY(100%); }
}

/* 保留之前的消息样式，但调整以适应新的电视效果 */
.chat-window {
  height: 100%;
  padding: 1rem;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: #4a4a5a #2a2a3a;
}

/* 自定义滚动条样式 */
.chat-window::-webkit-scrollbar {
  width: 8px;
}

.chat-window::-webkit-scrollbar-track {
  background: #2a2a3a;
  border-radius: 4px;
}

.chat-window::-webkit-scrollbar-thumb {
  background: #4a4a5a;
  border-radius: 4px;
}

/* 字段选择器样式调整 */
.field-selector-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  backdrop-filter: blur(10px);
}

.field-selector-content {
  background: #2a2a3a;
  padding: 2rem;
  border-radius: 16px;
  width: 90%;
  max-width: 500px;
  border: 2px solid #3a3a4a;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.field-count {
  color: #8a8a9a;
  font-size: 0.9rem;
}

.field-list {
  max-height: 300px;
  overflow-y: auto;
  margin: 1rem 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 0.5rem;
}

.field-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
  cursor: pointer;
}

.field-selector-actions {
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
  margin-top: 1rem;
}

.confirm-btn {
  background: linear-gradient(45deg, #7c4dff, #448aff);
  color: white;
}

/* 响应式设计优化 */
@media (max-width: 768px) {
  .tv-container {
    flex-direction: column;
    padding: 1rem;
    height: 100vh;
    overflow-y: auto;
  }

  .control-panel {
    width: 100%;
    height: auto;
    min-height: 200px;
  }

  .tv-screen {
    height: calc(100vh - 250px);
  }
}
</style> 