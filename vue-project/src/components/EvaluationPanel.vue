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
          <button @click="changeChannel(1)" class="control-button channel-btn" :class="{ 'active': activeChannel === 1 }">1</button>
          <button @click="changeChannel(2)" class="control-button channel-btn" :class="{ 'active': activeChannel === 2 }">2</button>
          <button @click="changeChannel(3)" class="control-button channel-btn" :class="{ 'active': activeChannel === 3 }">3</button>
        </div>
      </div>
    </div>

    <!-- 电视屏幕 -->
    <div class="tv-screen">
      <div class="screen-frame" :class="{ 'scanning': isScanning, 'changing-channel': isChangingChannel }">
        <div class="screen-content">
          <!-- 评估过程显示 - Channel 1 -->
          <div v-if="activeChannel === 1" class="chat-window" ref="chatWindow">
            <!-- 无数据时显示无信号 -->
            <div v-if="!evaluationText && !systemMessage" class="no-signal">
              <div class="static-effect"></div>
              <div class="no-signal-text">NO SIGNAL</div>
            </div>
            
            <!-- 有系统消息但无评估数据时显示待机画面 -->
            <div v-else-if="systemMessage && !evaluationText" class="standby-screen">
              <div class="tv-logo">AI EVALUATOR</div>
              <div class="standby-message">{{ systemMessage }}</div>
              <div class="standby-animation"></div>
            </div>
            
            <!-- 有评估数据时显示内容 -->
            <div v-else class="evaluation-content">
              <div class="message system-message" v-if="systemMessage">
                {{ systemMessage }}
              </div>
              <div class="message ai-message" v-if="evaluationText">
                <div class="message-header">
                  <span class="ai-badge">AI</span>
                  <span>评估结果</span>
                </div>
                <div class="message-content typewriter">
                  <pre class="typewriter-text">{{ evaluationText }}<span class="cursor" :class="{ 'blink': !isScanning }">|</span></pre>
                </div>
              </div>
            </div>
          </div>
          
          <!-- 评估报告显示 - Channel 2 -->
          <div v-if="activeChannel === 2" class="chat-window report-view">
            <!-- 无数据时显示无信号 -->
            <div v-if="!evaluationStats" class="no-signal">
              <div class="static-effect"></div>
              <div class="no-signal-text">NO SIGNAL</div>
            </div>
            
            <!-- 有数据时显示报告 -->
            <div v-else class="report-container">
              <h2 class="report-title">评估报告</h2>
              
              <!-- 总体评分 -->
              <div class="score-overview">
                <div class="score-card">
                  <div class="score-value">{{ evaluationStats.overall_scores.final_score }}</div>
                  <div class="score-label">总体评分</div>
                </div>
                <div class="score-card">
                  <div class="score-value">{{ evaluationStats.overall_scores.role_score }}</div>
                  <div class="score-label">角色评分</div>
                </div>
                <div class="score-card">
                  <div class="score-value">{{ evaluationStats.overall_scores.dialogue_score }}</div>
                  <div class="score-label">对话评分</div>
                </div>
              </div>
              
              <!-- 角色扮演评估 -->
              <div class="assessment-section">
                <h3>角色扮演评估</h3>
                <div class="score-bars">
                  <div class="score-bar-item" v-for="(item, key) in rolePlayItems" :key="key">
                    <div class="score-bar-label">{{ item.label }}</div>
                    <div class="score-bar-container">
                      <div class="score-bar" :style="{ width: `${getScoreValue(key, 'role_play')}%` }"></div>
                    </div>
                    <div class="score-bar-value">{{ getScoreValue(key, 'role_play') }}</div>
                  </div>
                </div>
              </div>
              
              <!-- 对话体验评估 -->
              <div class="assessment-section">
                <h3>对话体验评估</h3>
                <div class="score-bars">
                  <div class="score-bar-item" v-for="(item, key) in dialogueItems" :key="key">
                    <div class="score-bar-label">{{ item.label }}</div>
                    <div class="score-bar-container">
                      <div class="score-bar" :style="{ width: `${getScoreValue(key, 'dialogue_experience')}%` }"></div>
                    </div>
                    <div class="score-bar-value">{{ getScoreValue(key, 'dialogue_experience') }}</div>
                  </div>
                </div>
              </div>
              
              <!-- 优势和弱点 -->
              <div class="strengths-weaknesses">
                <div class="sw-column">
                  <h3>优势</h3>
                  <ul class="sw-list">
                    <li v-for="(count, strength) in evaluationStats.common_strengths" :key="strength">
                      {{ strength }}
                    </li>
                  </ul>
                </div>
                <div class="sw-column">
                  <h3>弱点</h3>
                  <ul class="sw-list">
                    <li v-for="(count, weakness) in evaluationStats.common_weaknesses" :key="weakness">
                      {{ weakness }}
                    </li>
                  </ul>
                </div>
              </div>
              
              <!-- 建议 -->
              <div class="suggestions-section">
                <h3>改进建议</h3>
                <ul class="suggestions-list">
                  <li v-for="(count, suggestion) in evaluationStats.common_suggestions" :key="suggestion">
                    {{ suggestion }}
                  </li>
                </ul>
              </div>
            </div>
          </div>
          
          <!-- 无信号显示 - Channel 3 -->
          <div v-if="activeChannel === 3" class="no-signal">
            <div class="static-effect"></div>
            <div class="no-signal-text">NO SIGNAL</div>
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
import { useRouter } from 'vue-router'

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
const isScanning = ref(false)
const typingTimeout = ref(null)

const showFieldSelector = ref(false)
const availableFields = ref([])
const selectedFields = ref([])
const fieldsConfirmed = ref(false)

// 新增：控制是否显示报告
const showReport = ref(false)
// 新增：报告数据
const evaluationStats = ref(null)
const activeChannel = ref(1) // 当前频道
const isChangingChannel = ref(false) // 是否正在换台

const router = useRouter()

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

    // 开始接收数据时启动扫描
    isScanning.value = true

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
      if (done) {
        // 数据接收完成，停止扫描
        isScanning.value = false
        break
      }

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
              if (!isScanning.value) {
                isScanning.value = true
              }
              currentEvaluation.content += data.content
              evaluationText.value = evaluationText.value.split(`评估项 ${data.index}:`)[0] + 
                                   `评估项 ${data.index}:\n原始数据:\n${currentEvaluation.originalData}\n\n评估结果:\n${currentEvaluation.content}`
              playTypeSound()
              scrollToBottom()
              
              // 重置打字状态的计时器
              clearTimeout(typingTimeout)
              typingTimeout = setTimeout(() => {
                isScanning.value = false
              }, 100)
              break

            case 'end':
              processed.value = data.index
              break

            case 'complete':
              // 处理评估统计数据
              showEvaluationReport(data.stats)
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
    isScanning.value = false
  } finally {
    isEvaluating.value = false
    systemMessage.value = '评估完成！'
    isScanning.value = false  // 确保扫描停止
  }
}

// 修改换台函数
const changeChannel = (channel) => {
  if (channel === activeChannel.value) return
  
  // 开始换台效果
  isChangingChannel.value = true
  isScanning.value = true
  
  // 延迟切换频道，模拟换台过程
  setTimeout(() => {
    activeChannel.value = channel
    
    // 如果切换到频道3，导出评估数据
    if (channel === 3 && evaluationStats.value) {
      exportEvaluationReport()
    }
    
    // 结束换台效果
    setTimeout(() => {
      isChangingChannel.value = false
      isScanning.value = false
    }, 500)
  }, 1000)
}

// 导出评估报告函数
const exportEvaluationReport = () => {
  if (evaluationStats.value) {
    const dataStr = JSON.stringify(evaluationStats.value, null, 2)
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr)
    
    const exportFileDefaultName = 'evaluation_report.json'
    
    const linkElement = document.createElement('a')
    linkElement.setAttribute('href', dataUri)
    linkElement.setAttribute('download', exportFileDefaultName)
    linkElement.click()
  } else {
    systemMessage.value = '没有评估数据可以导出'
  }
}

// 添加角色扮演评估项
const rolePlayItems = {
  consistency: { label: '角色一致性' },
  knowledge: { label: '角色知识' },
  language_style: { label: '语言风格' },
  emotional_expression: { label: '情感表达' },
  character_depth: { label: '角色深度' }
}

// 添加对话体验评估项
const dialogueItems = {
  response_quality: { label: '回应质量' },
  interaction_fluency: { label: '交互流畅度' },
  language_expression: { label: '语言表达' },
  context_adaptation: { label: '情境适应性' },
  personalization: { label: '个性化体验' }
}

// 获取评分值的辅助函数
const getScoreValue = (key, category) => {
  if (!evaluationStats.value || !evaluationStats.value[category] || !evaluationStats.value[category][key]) {
    return 0
  }
  return evaluationStats.value[category][key].avg || 0
}

// 修改显示统计报告的方法
const showEvaluationReport = (stats) => {
  if (!stats) {
    console.error('没有收到统计数据')
    systemMessage.value = '没有收到有效的统计数据，无法显示报告！'
    return
  }
  
  console.log('收到评估统计数据:', stats)
  
  // 直接使用统计数据
  evaluationStats.value = stats
  
  // 显示成功消息
  systemMessage.value = `评估完成！总体评分: ${stats.overall_scores?.final_score || 'N/A'}，角色评分: ${stats.overall_scores?.role_score || 'N/A'}，对话评分: ${stats.overall_scores?.dialogue_score || 'N/A'}`
  
  // 停止扫描效果
  isScanning.value = false
  
  // 自动切换到报告视图
  changeChannel(2)
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
  /* 外壳立体效果 */
  box-shadow: 
    -5px -5px 15px rgba(255,255,255,0.1),
    5px 5px 15px rgba(0,0,0,0.4);
  border: 2px solid #2a2a3a;
  background: 
    linear-gradient(45deg, #1a1a2e, #2a2a3a) padding-box,
    linear-gradient(45deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)) border-box;
}

.screen-frame {
  background: linear-gradient(45deg, #1a1a2e, #2a2a3a);
  border-radius: 15px;
  padding: 15px;
  height: 100%;
  position: relative;
  overflow: hidden;
  /* 玻璃屏幕表面效果 */
  box-shadow: 
    inset 0 0 50px rgba(0,0,0,0.5),
    inset 0 0 20px rgba(0,0,0,0.3);
}

/* 玻璃表面反光效果 */
.screen-frame::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: 
    repeating-linear-gradient(
      0deg,
      rgba(255,255,255,0) 0%,
      rgba(255,255,255,0.03) 0.5%,
      rgba(255,255,255,0) 1%
    ),
    radial-gradient(
      circle at center,
      rgba(255,255,255,0.1) 0%,
      rgba(0,0,0,0.2) 100%
    );
  pointer-events: none;
  animation: scanlines 8s linear infinite;
  border-radius: 15px;
  z-index: 2;
  opacity: 0.6;
}

/* 移动扫描效果只在scanning时显示 */
.screen-frame::after {
  content: '';
  position: absolute;
  top: -100%;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    0deg,
    transparent 0%,
    rgba(255,255,255,0.05) 50%,
    transparent 100%
  );
  pointer-events: none;
  z-index: 2;
  opacity: 0;
  animation: scanning 3s linear infinite;
  animation-play-state: paused;
}

.screen-frame.scanning::after {
  opacity: 1;
  animation-play-state: running;
}

.screen-content {
  background: #000;
  border-radius: 10px;
  height: 100%;
  overflow: hidden;
  position: relative;
  /* 内部显示效果 */
  box-shadow: inset 0 0 30px rgba(0,0,0,0.8);
}

/* 内部微光效果 */
.screen-content::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: radial-gradient(
    circle at 50% 50%,
    rgba(255,255,255,0.05) 0%,
    transparent 60%
  );
  pointer-events: none;
  opacity: 0.8;
  animation: glow 4s ease-in-out infinite;
  z-index: 1;
}

/* 调整扫描线动画速度和透明度 */
@keyframes scanlines {
  0% { background-position: 0 0; }
  100% { background-position: 0 100%; }
}

@keyframes scanning {
  0% { transform: translateY(0); }
  100% { transform: translateY(200%); }
}

@keyframes glow {
  0%, 100% { opacity: 0.8; }
  50% { opacity: 0.6; }
}

.chat-window {
  height: 100%;
  padding: 1rem;
  overflow-y: auto;
  position: relative;
  z-index: 1;
  color: rgba(255,255,255,0.9);
  text-shadow: 0 0 2px rgba(255,255,255,0.5);
}

/* 统一滚动条样式 */
.chat-window::-webkit-scrollbar,
.report-view::-webkit-scrollbar {
  width: 8px;
}

.chat-window::-webkit-scrollbar-track,
.report-view::-webkit-scrollbar-track {
  background: #2a2a3a;
  border-radius: 4px;
}

.chat-window::-webkit-scrollbar-thumb,
.report-view::-webkit-scrollbar-thumb {
  background: #4a4a5a;
  border-radius: 4px;
}

/* 报告视图样式调整 */
.report-view {
  height: 100%;
  overflow-y: auto;
  padding: 1.5rem;
  color: #e0e0e0;
}

.report-container {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.report-title {
  text-align: center;
  color: #44ff44;
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.5);
  margin-bottom: 1.5rem;
  font-size: 1.8rem;
}

.score-overview {
  display: flex;
  justify-content: space-around;
  margin-bottom: 2rem;
}

.score-card {
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 10px;
  padding: 1.5rem;
  text-align: center;
  width: 120px;
}

.score-value {
  font-size: 2.5rem;
  font-weight: bold;
  color: #44ff44;
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.5);
}

.score-label {
  margin-top: 0.5rem;
  font-size: 0.9rem;
  color: #a0a0a0;
}

.assessment-section {
  margin-bottom: 2rem;
}

.assessment-section h3 {
  margin-bottom: 1rem;
  color: #44ff44;
  text-shadow: 0 0 5px rgba(68, 255, 68, 0.3);
  font-size: 1.3rem;
}

.score-bars {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.score-bar-item {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.score-bar-label {
  width: 120px;
  text-align: right;
  font-size: 0.9rem;
}

.score-bar-container {
  flex: 1;
  height: 12px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  overflow: hidden;
}

.score-bar {
  height: 100%;
  background: linear-gradient(90deg, #44ff44, #7cff7c);
  border-radius: 6px;
  transition: width 1s ease-out;
}

.score-bar-value {
  width: 40px;
  text-align: right;
  font-size: 0.9rem;
}

.strengths-weaknesses {
  display: flex;
  gap: 2rem;
  margin-bottom: 2rem;
}

.sw-column {
  flex: 1;
}

.sw-column h3 {
  margin-bottom: 1rem;
  color: #44ff44;
  text-shadow: 0 0 5px rgba(68, 255, 68, 0.3);
  font-size: 1.3rem;
}

.sw-list, .suggestions-list {
  list-style-type: none;
  padding: 0;
  margin: 0;
}

.sw-list li, .suggestions-list li {
  background: rgba(0, 0, 0, 0.3);
  padding: 0.8rem;
  margin-bottom: 0.5rem;
  border-radius: 4px;
  border-left: 3px solid #44ff44;
}

.suggestions-section h3 {
  margin-bottom: 1rem;
  color: #44ff44;
  text-shadow: 0 0 5px rgba(68, 255, 68, 0.3);
  font-size: 1.3rem;
}

/* 换台效果 */
.changing-channel::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    to bottom,
    transparent,
    rgba(255, 255, 255, 0.2) 20%,
    rgba(255, 255, 255, 0.2) 80%,
    transparent
  );
  z-index: 10;
  animation: channel-change 1s ease-in-out;
  pointer-events: none;
}

@keyframes channel-change {
  0% { transform: translateY(-100%); opacity: 0.8; }
  50% { transform: translateY(0); opacity: 1; }
  100% { transform: translateY(100%); opacity: 0.8; }
}

/* 无信号效果 */
.no-signal {
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  position: relative;
  overflow: hidden;
}

.static-effect {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-image: url('data:image/svg+xml;utf8,<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><filter id="a"><feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch"/><feColorMatrix values="1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0"/></filter><rect width="100%" height="100%" filter="url(%23a)"/></svg>');
  opacity: 0.3;
  animation: static 0.5s steps(1) infinite;
}

@keyframes static {
  0% { transform: translate(0, 0); }
  10% { transform: translate(-5%, -5%); }
  20% { transform: translate(10%, 5%); }
  30% { transform: translate(-5%, 10%); }
  40% { transform: translate(5%, -10%); }
  50% { transform: translate(-10%, 5%); }
  60% { transform: translate(10%, 10%); }
  70% { transform: translate(-10%, -10%); }
  80% { transform: translate(5%, 5%); }
  90% { transform: translate(-5%, -5%); }
  100% { transform: translate(0, 0); }
}

.no-signal-text {
  font-size: 3rem;
  font-weight: bold;
  color: white;
  text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
  letter-spacing: 5px;
  animation: flicker 2s linear infinite;
  z-index: 2;
}

@keyframes flicker {
  0%, 19.999%, 22%, 62.999%, 64%, 64.999%, 70%, 100% {
    opacity: 0.99;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.5), 0 0 20px rgba(255, 255, 255, 0.4);
  }
  20%, 21.999%, 63%, 63.999%, 65%, 69.999% {
    opacity: 0.4;
    text-shadow: none;
  }
}

/* 激活的频道按钮样式 */
.channel-btn.active {
  background: rgba(68, 255, 68, 0.2);
  border: 1px solid #44ff44;
  color: #44ff44;
  text-shadow: 0 0 5px #44ff44;
  box-shadow: 0 0 10px rgba(68, 255, 68, 0.3);
}

/* 频道按钮组样式 */
.channel-buttons {
  display: flex;
  gap: 0.5rem;
}

.channel-btn {
  width: 40px;
  height: 40px;
  display: flex;
  justify-content: center;
  align-items: center;
  font-weight: bold;
  transition: all 0.3s ease;
}

.channel-btn:hover {
  transform: scale(1.05);
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
    transform: none;
  }
}

/* 待机画面样式 */
.standby-screen {
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  position: relative;
  background: #000;
  color: #44ff44;
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.7);
  overflow: hidden;
}

.standby-screen::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: 
    repeating-linear-gradient(
      0deg,
      rgba(0, 0, 0, 0.15) 0px,
      rgba(0, 0, 0, 0.15) 1px,
      transparent 1px,
      transparent 2px
    );
  pointer-events: none;
  z-index: 1;
}

.tv-logo {
  font-size: 3rem;
  font-weight: bold;
  letter-spacing: 5px;
  margin-bottom: 2rem;
  position: relative;
  z-index: 2;
  animation: pulse 2s infinite alternate;
}

.standby-message {
  font-size: 1.2rem;
  max-width: 80%;
  text-align: center;
  line-height: 1.6;
  background: rgba(0, 0, 0, 0.5);
  padding: 1rem;
  border-radius: 10px;
  border: 1px solid rgba(68, 255, 68, 0.3);
  position: relative;
  z-index: 2;
}

.standby-animation {
  position: absolute;
  bottom: 10%;
  width: 60%;
  height: 4px;
  background: rgba(68, 255, 68, 0.5);
  border-radius: 2px;
  overflow: hidden;
  z-index: 2;
}

.standby-animation::after {
  content: '';
  position: absolute;
  top: 0;
  left: -50%;
  width: 50%;
  height: 100%;
  background: #44ff44;
  animation: scanning-line 2s infinite linear;
}

@keyframes scanning-line {
  0% { transform: translateX(0); }
  100% { transform: translateX(200%); }
}

@keyframes pulse {
  0% { opacity: 0.7; }
  100% { opacity: 1; }
}

/* 评估内容容器 */
.evaluation-content {
  height: 100%;
  overflow-y: auto;
  padding: 1rem;
}
</style> 