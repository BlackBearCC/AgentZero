<template>
  <div class="character-generator">
    <!-- 未上传文件时的空状态 -->
    <div v-if="!characterData && !isGenerating" class="empty-state">
      <div class="tv-logo">CHARACTER GENERATOR</div>
      <div class="channel-info">频道 1</div>
      <div class="instruction-text">请上传角色资料文件开始生成</div>
      
      <!-- 控制区域整合到屏幕中 -->
      <div class="screen-controls">
        <div class="file-control">
          <input 
            type="file" 
            id="character-file" 
            @change="handleFileChange" 
            accept=".txt,.pdf,.docx"
            class="file-input"
          />
          <label for="character-file" class="tv-button">
            <span class="button-text">[ 选择角色文件 ]</span>
          </label>
          <div class="file-name">{{ fileName || '未选择文件' }}</div>
        </div>

        <!-- 生成选项 -->
        <div class="option-group">
          <div class="option-item">
            <input type="checkbox" id="extract-traits" v-model="extractTraits" />
            <label for="extract-traits">提取性格特质</label>
          </div>
          <div class="option-item">
            <input type="checkbox" id="extract-background" v-model="extractBackground" />
            <label for="extract-background">提取背景故事</label>
          </div>
          <div class="option-item">
            <input type="checkbox" id="extract-keywords" v-model="extractKeywords" />
            <label for="extract-keywords">提取关键词</label>
          </div>
        </div>
      
        <!-- 操作按钮 -->
        <div class="action-buttons">
          <button @click="generateCharacter" class="tv-button primary" :disabled="!canGenerate">
            <span class="button-text">[ 开始生成 ]</span>
          </button>
        </div>
      </div>
    </div>
    
    <!-- 生成中状态 -->
    <div v-else-if="isGenerating" class="processing-state">
      <div class="tv-logo">CHARACTER GENERATOR</div>
      <div class="generation-screen">
        <div class="scan-line"></div>
        <div class="screen-glare"></div>
        <StreamDisplay 
          :content="streamContent"
          :loading="isGenerating"
          placeholder="正在生成角色配置..."
          :typing-effect="true"
          :typing-speed="30"
        />
      </div>
    </div>
    
    <!-- 生成结果 -->
    <div v-else class="character-result">
      <div class="result-header">
        <h2>{{ characterData.name }}</h2>
        <div class="header-actions">
          <button @click="resetGenerator" class="tv-button">
            <span class="button-text">[ 重新生成 ]</span>
          </button>
          <button @click="exportCharacter" class="tv-button">
            <span class="button-text">[ 导出角色 ]</span>
          </button>
        </div>
      </div>
      
      <div class="character-content">
        <div class="character-section">
          <h3>背景故事</h3>
          <p>{{ characterData.background }}</p>
        </div>
        
        <div class="character-section">
          <h3>关键特点</h3>
          <div class="keywords-list">
            <span v-for="(keyword, index) in characterData.keywords" :key="index" class="keyword-tag">
              {{ keyword }}
            </span>
          </div>
        </div>
        
        <div class="character-section">
          <h3>性格特质</h3>
          <div class="traits-list">
            <div v-for="(trait, index) in characterData.traits" :key="index" class="trait-item">
              <div class="trait-name">{{ trait.name }}</div>
              <div class="trait-bar-container">
                <div class="trait-bar" :style="{ width: `${trait.value}%` }"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, nextTick } from 'vue'
import StreamDisplay from './StreamDisplay.vue'

// 文件状态
const fileName = ref('')
const selectedFile = ref(null)

// 生成选项
const extractTraits = ref(true)
const extractBackground = ref(true)
const extractKeywords = ref(true)

// 生成状态
const isGenerating = ref(false)
const characterData = ref(null)
const streamContent = ref('')

// 计算属性
const canGenerate = computed(() => {
  return selectedFile.value && !isGenerating.value
})

// 处理文件选择
const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (file) {
    selectedFile.value = file
    fileName.value = file.name
  } else {
    selectedFile.value = null
    fileName.value = ''
  }
}

// 生成角色
const generateCharacter = async () => {
  if (!canGenerate.value) return
  
  isGenerating.value = true
  characterData.value = null
  streamContent.value = ''  // 重置流内容
  
  try {
    const response = await fetch('/api/v1/generate_role_config/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify({
        reference: await selectedFile.value.text()
      })
    })
    
    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      
      buffer += decoder.decode(value, { stream: true })
      
      // 处理SSE事件
      let eventEndIndex
      while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
        const event = buffer.slice(0, eventEndIndex)
        buffer = buffer.slice(eventEndIndex + 2)
        
        if (event.startsWith('data: ')) {
          const data = JSON.parse(event.slice(6).trim())
          
          // 根据事件类型处理数据
          switch (data.type) {
            case 'chunk':
              streamContent.value += data.content
              break
            case 'complete':
              // 最终数据处理
              const finalData = JSON.parse(data.content)
              characterData.value = {
                name: finalData.基础信息?.[0]?.内容 || '',
                background: finalData.成长经历?.[0]?.内容 || '',
                keywords: finalData.关键词 || [],
                traits: finalData.性格特征?.map(t => ({
                  name: t.内容,
                  value: Math.floor(Math.random() * 40 + 60)
                })) || []
              }
              break
          }
        }
      }
    }
  } catch (e) {
    console.error('生成失败:', e)
  } finally {
    isGenerating.value = false
  }
}

// 导出角色
const exportCharacter = () => {
  if (!characterData.value) return
  
  const charData = JSON.stringify(characterData.value, null, 2)
  const blob = new Blob([charData], { type: 'application/json' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = `${characterData.value.name.replace(/\s+/g, '_')}.json`
  link.click()
  URL.revokeObjectURL(link.href)
}

// 重置生成器
const resetGenerator = () => {
  characterData.value = null
  selectedFile.value = null
  fileName.value = ''
  isGenerating.value = false
}
</script>

<style scoped>
.character-generator {
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
  margin-bottom: 2rem;
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

/* 集成控制区域样式 */
.screen-controls {
  width: 80%;
  max-width: 600px;
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 10px;
  padding: 20px;
  margin-top: 1rem;
}

.file-control {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 20px;
}

.file-input {
  position: absolute;
  width: 0.1px;
  height: 0.1px;
  opacity: 0;
  overflow: hidden;
  z-index: -1;
}

.tv-button {
  background: rgba(40, 40, 60, 0.8);
  border: 1px solid rgba(68, 255, 68, 0.5);
  border-radius: 5px;
  padding: 8px 20px;
  color: #44ff44;
  font-family: 'Courier New', monospace;
  cursor: pointer;
  transition: all 0.3s ease;
  display: inline-block;
  margin: 5px 0;
}

.tv-button:hover {
  background: rgba(60, 60, 80, 0.8);
  box-shadow: 0 0 10px rgba(68, 255, 68, 0.5);
}

.tv-button.primary {
  background: rgba(68, 255, 68, 0.3);
}

.tv-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.file-name {
  margin-top: 10px;
  color: #a0a0a0;
  font-style: italic;
}

.option-group {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 20px;
  align-items: flex-start;
}

.option-item {
  display: flex;
  align-items: center;
  gap: 10px;
}

.option-item input[type="checkbox"] {
  appearance: none;
  width: 18px;
  height: 18px;
  border: 1px solid #44ff44;
  border-radius: 3px;
  background: rgba(0, 0, 0, 0.5);
  position: relative;
  cursor: pointer;
}

.option-item input[type="checkbox"]:checked::after {
  content: '✓';
  position: absolute;
  color: #44ff44;
  font-size: 14px;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

.action-buttons {
  display: flex;
  justify-content: center;
  gap: 15px;
}

/* 处理中动画 */
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
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* 结果显示样式 */
.character-result {
  padding: 20px;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(68, 255, 68, 0.3);
}

.result-header h2 {
  color: #44ff44;
  margin: 0;
  font-size: 1.5rem;
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.5);
}

.header-actions {
  display: flex;
  gap: 10px;
}

.character-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.character-section {
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(68, 255, 68, 0.2);
  border-radius: 8px;
  padding: 15px;
}

.character-section h3 {
  color: #44ff44;
  margin-top: 0;
  margin-bottom: 10px;
  font-size: 1.2rem;
}

.keywords-list {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.keyword-tag {
  background: rgba(68, 255, 68, 0.2);
  border: 1px solid rgba(68, 255, 68, 0.5);
  border-radius: 15px;
  padding: 5px 12px;
  font-size: 0.9rem;
}

.traits-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.trait-item {
  display: flex;
  align-items: center;
}

.trait-name {
  width: 100px;
  font-size: 0.9rem;
}

.trait-bar-container {
  flex: 1;
  height: 15px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  overflow: hidden;
}

.trait-bar {
  height: 100%;
  background: linear-gradient(90deg, rgba(68, 255, 68, 0.5), #44ff44);
  border-radius: 8px;
}

.generation-screen {
  flex: 1;
  position: relative;
  background: rgba(0, 0, 0, 0.9);
  border: 2px solid rgba(68, 255, 68, 0.3);
  border-radius: 15px;
  overflow: hidden;
  padding: 20px;
  margin: 20px;
  margin-top: 60px; /* 为顶部logo留出空间 */
  width: calc(100% - 40px); /* 减去左右margin */
  height: calc(100% - 100px); /* 减去上下margin和logo高度 */
  box-shadow: 
    inset 0 0 50px rgba(68, 255, 68, 0.1),
    0 0 20px rgba(68, 255, 68, 0.2);
  display: flex; /* 使用flex布局 */
}

/* 扫描线效果 */
.scan-line {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: rgba(68, 255, 68, 0.2);
  animation: scanning 8s linear infinite;
  z-index: 2;
}

/* CRT屏幕反光效果 */
.screen-glare {
  position: absolute;
  top: -50%;
  left: -50%;
  right: -50%;
  bottom: -50%;
  background: radial-gradient(
    ellipse at center,
    rgba(255, 255, 255, 0.05) 0%,
    rgba(255, 255, 255, 0) 60%
  );
  pointer-events: none;
  z-index: 1;
}

@keyframes scanning {
  0% { transform: translateY(0); }
  100% { transform: translateY(100%); }
}

/* 修改 StreamDisplay 容器样式 */
:deep(.stream-content) {
  height: 100%;
  width: 100%;
  background: transparent;
  border: none;
  font-family: 'VT323', 'Courier New', monospace;
  color: #44ff44;
  text-shadow: 0 0 5px rgba(68, 255, 68, 0.5);
  padding: 20px;
  position: relative;
  z-index: 3;
  overflow-y: auto;
}

:deep(.stream-text) {
  font-size: 1.2rem;
  line-height: 1.6;
  width: 100%;
  /* 防止抖动的关键设置 */
  min-height: 100%;
  position: relative;
}

/* 修改光标样式以匹配主题 */
:deep(.stream-text.typing::after) {
  background-color: #44ff44;
  box-shadow: 0 0 5px rgba(68, 255, 68, 0.7);
  height: 1.2em;
}

:deep(.loading-indicator) {
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid rgba(68, 255, 68, 0.3);
  padding: 8px 15px;
}

/* 自定义滚动条 */
:deep(.stream-content::-webkit-scrollbar) {
  width: 10px;
}

:deep(.stream-content::-webkit-scrollbar-track) {
  background: rgba(0, 0, 0, 0.3);
}

:deep(.stream-content::-webkit-scrollbar-thumb) {
  background: rgba(68, 255, 68, 0.3);
  border-radius: 5px;
}

:deep(.stream-content::-webkit-scrollbar-thumb:hover) {
  background: rgba(68, 255, 68, 0.5);
}

/* 闪烁的光标效果 */
:deep(.stream-text.typing::after) {
  content: '█';
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

/* 适配电视机主题的其他样式调整 */
.tv-logo {
  position: absolute;
  top: 10px;
  left: 50%;
  transform: translateX(-50%);
  font-family: 'VT323', monospace;
  font-size: 1.5rem;
  color: #44ff44;
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.7);
  z-index: 4;
}
</style>