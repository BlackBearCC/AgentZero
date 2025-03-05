<template>
  <div class="character-generator">
    <!-- 未上传文件时的空状态 -->
    <div v-if="!hasGeneratedData && !isGenerating" class="empty-state">
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
            <input type="checkbox" id="batch-generate" v-model="batchGenerate" />
            <label for="batch-generate">批量生成属性</label>
          </div>
          <div class="option-item">
            <input type="checkbox" id="show-generation-process" v-model="showGenerationProcess" />
            <label for="show-generation-process">显示生成过程</label>
          </div>
        </div>
        
        <!-- 类别选择 -->
        <div class="category-selection" v-if="batchGenerate">
          <h4>选择要生成的属性类别</h4>
          <div class="category-grid">
            <div 
              v-for="(category, index) in categoryOptions" 
              :key="index"
              class="category-option"
            >
              <input 
                type="checkbox" 
                :id="`category-${index}`" 
                v-model="selectedCategories"
                :value="category.key"
              />
              <label :for="`category-${index}`">{{ category.title }}</label>
            </div>
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
    <div v-else-if="isGenerating && showGenerationProcess" class="processing-state">
      <div class="tv-logo">CHARACTER GENERATOR</div>
      <div class="generation-screen">
        <div class="scan-line"></div>
        <div class="screen-glare"></div>
        <StreamDisplay 
          :content="streamContent"
          :loading="isGenerating"
          placeholder="正在生成角色配置..."
          :typing-effect="true"
          :typing-speed="3"
        />
      </div>
    </div>
    
    <!-- 生成结果 - 使用新组件 -->
    <div v-else class="result-container">
      <CharacterReport 
        :character="characterData"
        :loadingCategories="loadingCategories"
        @reset="resetGenerator"
        @refresh="refreshCategory"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import StreamDisplay from './StreamDisplay.vue'
import CharacterReport from './CharacterReport.vue'

// 文件状态
const fileName = ref('')
const selectedFile = ref(null)

// 生成选项
const batchGenerate = ref(true)
const showGenerationProcess = ref(true)
const selectedCategories = ref([
  '基础信息', '性格特征', '能力特征', '兴趣爱好', '情感特质'
])

// 类别配置
const categoryOptions = [
  { key: '基础信息', title: '基础信息' },
  { key: '性格特征', title: '性格特征' },
  { key: '能力特征', title: '能力特征' },
  { key: '兴趣爱好', title: '兴趣爱好' },
  { key: '情感特质', title: '情感特质' },
  { key: '喜好厌恶', title: '喜好与厌恶' },
  { key: '成长经历', title: '成长经历' },
  { key: '价值观念', title: '价值观念' },
  { key: '社交关系', title: '社交关系' },
  { key: '禁忌话题', title: '禁忌话题' },
  { key: '行为模式', title: '行为模式' },
  { key: '隐藏设定', title: '隐藏设定' },
  { key: '目标动机', title: '目标动机' },
  { key: '弱点缺陷', title: '弱点缺陷' },
  { key: '特殊习惯', title: '特殊习惯' },
  { key: '语言风格', title: '语言风格' }
]

// 生成状态
const isGenerating = ref(false)
const characterData = reactive({})
const streamContent = ref('')
const loadingCategories = ref([])

// 计算属性
const canGenerate = computed(() => {
  return selectedFile.value && !isGenerating.value && 
         (batchGenerate.value ? selectedCategories.value.length > 0 : true)
})

const hasGeneratedData = computed(() => {
  return Object.keys(characterData).length > 0
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
  streamContent.value = ''  // 重置流内容
  
  // 如果是批量生成，设置正在加载的类别
  if (batchGenerate.value) {
    loadingCategories.value = [...selectedCategories.value]
  } else {
    // 如果不是批量生成，清空现有数据
    Object.keys(characterData).forEach(key => {
      delete characterData[key]
    })
  }
  
  try {
    const fileContent = await selectedFile.value.text()
    
    // 构建请求体
    const requestBody = {
      reference: fileContent
    }
    
    // 如果是批量生成，添加类别信息
    if (batchGenerate.value) {
      requestBody.categories = selectedCategories.value
    }
    
    const response = await fetch('/api/v1/generate_role_config/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify(requestBody)
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
            case 'start':
              if (data.category) {
                // 添加到正在加载的类别
                if (!loadingCategories.value.includes(data.category)) {
                  loadingCategories.value.push(data.category)
                }
              }
              break
            case 'chunk':
              streamContent.value += data.content
              break
            case 'complete':
              if (data.category && data.content) {
                // 更新特定类别的数据
                characterData[data.category] = data.content
                
                // 从加载列表中移除
                const index = loadingCategories.value.indexOf(data.category)
                if (index !== -1) {
                  loadingCategories.value.splice(index, 1)
                }
              } else if (!batchGenerate.value && data.content) {
                // 处理完整数据（非批量模式）
                Object.keys(data.content).forEach(key => {
                  characterData[key] = data.content[key]
                })
              }
              break
            case 'error':
              console.error('生成错误:', data.content)
              ElMessage.error(data.content)
              
              // 如果是特定类别的错误，从加载列表中移除
              if (data.category) {
                const index = loadingCategories.value.indexOf(data.category)
                if (index !== -1) {
                  loadingCategories.value.splice(index, 1)
                }
              }
              break
            case 'end':
              // 如果是特定类别的结束，不做特殊处理
              break
          }
        }
      }
    }
  } catch (e) {
    console.error('生成失败:', e)
    ElMessage.error('生成过程出现错误，请重试')
    loadingCategories.value = [] // 清空加载状态
  } finally {
    // 只有在没有正在加载的类别时才完全结束生成状态
    if (loadingCategories.value.length === 0) {
      isGenerating.value = false
    }
  }
}

// 导出角色
const exportCharacter = () => {
  if (Object.keys(characterData).length === 0) return
  
  const charData = JSON.stringify(characterData, null, 2)
  const blob = new Blob([charData], { type: 'application/json' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  
  // 尝试从基础信息中获取角色名称
  let fileName = '未命名角色'
  if (characterData.基础信息 && characterData.基础信息.length > 0) {
    const nameItem = characterData.基础信息.find(item => 
      item.关键词 && item.关键词.some(k => k.includes('姓名') || k.includes('名字'))
    )
    if (nameItem && nameItem.内容) {
      fileName = nameItem.内容.replace(/[（(].*[)）]/, '').trim()
    }
  }
  
  link.download = `${fileName.replace(/\s+/g, '_')}.json`
  link.click()
  URL.revokeObjectURL(link.href)
}

// 重置生成器
const resetGenerator = () => {
  // 清空所有数据
  Object.keys(characterData).forEach(key => {
    delete characterData[key]
  })
  selectedFile.value = null
  fileName.value = ''
  isGenerating.value = false
  loadingCategories.value = []
}

// 添加单个类别刷新函数
async function refreshCategory(categoryKey) {
  console.log('刷新类别被触发:', categoryKey); // 调试日志
  
  if (!selectedFile.value) {
    console.error('没有选择文件'); // 调试日志
    ElMessage.warning('没有可用的角色资料文件');
    return;
  }
  
  if (isGenerating.value) {
    console.warn('正在生成中，无法刷新'); // 调试日志
    ElMessage.warning('正在生成中，请稍后再试');
    return;
  }
  
  // 将类别添加到加载状态
  loadingCategories.value.push(categoryKey);
  console.log('加载状态更新:', loadingCategories.value); // 调试日志
  
  try {
    console.log('开始读取文件'); // 调试日志
    const fileContent = await selectedFile.value.text();
    
    // 构建请求体
    const requestBody = {
      reference: fileContent,
      categories: [categoryKey] // 只包含要刷新的类别
    };
    console.log('请求体:', requestBody); // 调试日志
    
    console.log('发送请求'); // 调试日志
    const response = await fetch('/api/v1/generate_role_config/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify(requestBody)
    });
    
    if (!response.ok) {
      console.error('请求失败:', response.status, response.statusText); // 调试日志
      throw new Error(`请求失败: ${response.status} ${response.statusText}`);
    }
    
    console.log('开始处理响应流'); // 调试日志
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log('响应流读取完成'); // 调试日志
        break;
      }
      
      const chunk = decoder.decode(value, { stream: true });
      console.log('收到数据块:', chunk); // 调试日志
      buffer += chunk;
      
      // 处理SSE事件
      let eventEndIndex;
      while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
        const event = buffer.slice(0, eventEndIndex);
        buffer = buffer.slice(eventEndIndex + 2);
        
        if (event.startsWith('data: ')) {
          try {
            const data = JSON.parse(event.slice(6).trim());
            console.log('解析的事件数据:', data); // 调试日志
            
            switch (data.type) {
              case 'complete':
                if (data.category && data.content) {
                  console.log(`更新类别 ${data.category} 的数据`); // 调试日志
                  characterData[data.category] = data.content;
                  ElMessage.success(`${data.category} 刷新成功`);
                }
                break;
              case 'error':
                console.error('生成错误:', data.content);
                ElMessage.error(data.content);
                break;
            }
          } catch (e) {
            console.error('解析事件数据失败:', e, event.slice(6).trim()); // 调试日志
          }
        }
      }
    }
  } catch (e) {
    console.error('刷新失败:', e);
    ElMessage.error(`刷新过程出现错误: ${e.message}`);
  } finally {
    // 从加载状态中移除该类别
    const index = loadingCategories.value.indexOf(categoryKey);
    if (index !== -1) {
      loadingCategories.value.splice(index, 1);
    }
    console.log('刷新完成，加载状态更新:', loadingCategories.value); // 调试日志
  }
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
  text-align: left; /* 确保文本左对齐 */
}

:deep(.stream-text) {
  font-size: 1.2rem;
  line-height: 1.6;
  width: 100%;
  min-height: 100%;
  position: relative;
  text-align: left; /* 确保文本左对齐 */
}

/* 修改光标样式以匹配主题 */
:deep(.stream-text.typing::after) {
  background-color: #44ff44;
  box-shadow: 0 0 5px rgba(68, 255, 68, 0.7);
  height: 1.2em;
  margin-left: 0; /* 移除左边距,使光标紧跟文本 */
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

/* 结果容器样式 */
.result-container {
  height: 100%;
  padding: 20px;
  background: rgba(0, 0, 0, 0.7);
  border-radius: 15px;
  overflow: hidden;
  position: relative;
  display: flex;
  flex-direction: column;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .screen-controls {
    width: 95%;
  }
  
  .generation-screen {
    margin: 10px;
    width: calc(100% - 20px);
  }
  
  .result-container {
    padding: 10px;
  }
}

/* 添加类别选择样式 */
.category-selection {
  margin: 15px 0;
  border-top: 1px solid rgba(68, 255, 68, 0.2);
  padding-top: 15px;
}

.category-selection h4 {
  color: #44ff44;
  margin: 0 0 10px 0;
  font-size: 1rem;
}

.category-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 10px;
}

.category-option {
  display: flex;
  align-items: center;
  gap: 8px;
}

.category-option label {
  cursor: pointer;
}
</style>