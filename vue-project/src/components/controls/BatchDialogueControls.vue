<template>
  <div class="control-section">
    <div class="section-title">批量对话控制</div>
    
    <!-- 文件上传 -->
    <div class="control-item">
      <div class="control-label">对话数据</div>
      <div class="file-upload">
        <input 
          type="file" 
          id="dialogue-file" 
          @change="handleFileChange" 
          accept=".csv,.xlsx"
          class="file-input"
        />
        <label for="dialogue-file" class="file-label">
          <span class="file-button">选择文件</span>
          <span class="file-name">{{ fileName || '未选择文件' }}</span>
        </label>
      </div>
    </div>
    
    <!-- 对话模型 -->
    <div class="control-item">
      <div class="control-label">对话模型</div>
      <select v-model="selectedModel" class="model-select">
        <option value="gpt-3.5">GPT-3.5</option>
        <option value="gpt-4">GPT-4</option>
        <option value="claude">Claude</option>
      </select>
    </div>
    
    <!-- 对话设置 -->
    <div class="control-item">
      <div class="control-label">对话设置</div>
      <div class="options-group">
        <div class="option-item">
          <input type="checkbox" id="use-system-prompt" v-model="useSystemPrompt" />
          <label for="use-system-prompt">使用系统提示词</label>
        </div>
        <div v-if="useSystemPrompt" class="system-prompt-container">
          <textarea 
            v-model="systemPrompt" 
            class="system-prompt" 
            placeholder="输入系统提示词..."
            rows="3"
          ></textarea>
        </div>
        <div class="option-item">
          <input type="checkbox" id="save-responses" v-model="saveResponses" />
          <label for="save-responses">自动保存回复</label>
        </div>
      </div>
    </div>
    
    <!-- 操作按钮 -->
    <div class="control-actions">
      <button @click="startProcessing" class="action-button primary" :disabled="!canProcess">
        开始处理
      </button>
      <button @click="resetProcessing" class="action-button secondary">
        重置
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, defineEmits } from 'vue'

const emit = defineEmits(['update:status'])

// 状态变量
const fileName = ref('')
const selectedFile = ref(null)
const selectedModel = ref('gpt-4')
const useSystemPrompt = ref(false)
const systemPrompt = ref('')
const saveResponses = ref(true)
const isProcessing = ref(false)

// 计算属性
const canProcess = computed(() => {
  return selectedFile.value && !isProcessing.value
})

// 处理文件选择
const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (file) {
    selectedFile.value = file
    fileName.value = file.name
    emit('update:status', `已选择文件: ${file.name}`)
  } else {
    selectedFile.value = null
    fileName.value = ''
  }
}

// 开始处理
const startProcessing = () => {
  if (!canProcess.value) return
  
  isProcessing.value = true
  emit('update:status', '正在处理批量对话...')
  
  // 这里应该调用父组件的方法来实际开始处理过程
  // 在实际应用中，你可能需要使用 refs 或事件来与父组件通信
  
  // 模拟处理完成
  setTimeout(() => {
    isProcessing.value = false
    emit('update:status', '批量对话处理完成')
  }, 5000)
}

// 重置处理
const resetProcessing = () => {
  selectedFile.value = null
  fileName.value = ''
  isProcessing.value = false
  emit('update:status', '系统就绪')
}
</script>

<style scoped>
.control-section {
  background-color: #222232;
  border: 1px solid #3a3a4a;
  border-radius: 5px;
  padding: 15px;
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.section-title {
  font-size: 1rem;
  color: #44ff44;
  margin-bottom: 10px;
  text-shadow: 0 0 5px rgba(68, 255, 68, 0.3);
}

.control-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.control-label {
  font-size: 0.8rem;
  color: #8a8a9a;
}

.file-upload {
  position: relative;
}

.file-input {
  position: absolute;
  width: 0.1px;
  height: 0.1px;
  opacity: 0;
  overflow: hidden;
  z-index: -1;
}

.file-label {
  display: flex;
  cursor: pointer;
}

.file-button {
  background-color: #1a1a2a;
  border: 1px solid #3a3a4a;
  border-radius: 5px 0 0 5px;
  padding: 8px 12px;
  color: #e0e0e0;
  transition: all 0.3s ease;
}

.file-name {
  flex: 1;
  background-color: #1a1a2a;
  border: 1px solid #3a3a4a;
  border-left: none;
  border-radius: 0 5px 5px 0;
  padding: 8px 12px;
  color: #8a8a9a;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.file-label:hover .file-button {
  background-color: #2a2a3a;
  border-color: #44ff44;
  color: #44ff44;
}

.options-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.option-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.option-item input[type="checkbox"] {
  appearance: none;
  width: 16px;
  height: 16px;
  border: 1px solid #3a3a4a;
  border-radius: 3px;
  background-color: #1a1a2a;
  cursor: pointer;
  position: relative;
}

.option-item input[type="checkbox"]:checked {
  background-color: #44ff44;
  border-color: #44ff44;
}

.option-item input[type="checkbox"]:checked::after {
  content: '✓';
  position: absolute;
  color: #000;
  font-size: 12px;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

.option-item label {
  cursor: pointer;
}

.system-prompt-container {
  margin-top: 5px;
  margin-bottom: 5px;
}

.system-prompt {
  width: 100%;
  background-color: #1a1a2a;
  border: 1px solid #3a3a4a;
  border-radius: 5px;
  padding: 8px;
  color: #e0e0e0;
  resize: vertical;
  font-family: inherit;
}

.system-prompt:focus {
  outline: none;
  border-color: #44ff44;
}

.model-select {
  background-color: #1a1a2a;
  border: 1px solid #3a3a4a;
  border-radius: 5px;
  padding: 8px 12px;
  color: #e0e0e0;
  width: 100%;
  cursor: pointer;
}

.model-select:focus {
  outline: none;
  border-color: #44ff44;
}

.control-actions {
  display: flex;
  gap: 10px;
  margin-top: 10px;
}

.action-button {
  flex: 1;
  padding: 10px;
  border-radius: 5px;
  cursor: pointer;
  font-weight: bold;
  transition: all 0.3s ease;
}

.action-button.primary {
  background-color: #44ff44;
  border: 1px solid #44ff44;
  color: #000;
}

.action-button.primary:hover:not(:disabled) {
  background-color: #66ff66;
  box-shadow: 0 0 10px rgba(68, 255, 68, 0.5);
}

.action-button.primary:disabled {
  background-color: #2a3a2a;
  border-color: #3a4a3a;
  color: #5a5a6a;
  cursor: not-allowed;
}

.action-button.secondary {
  background-color: #1a1a2a;
  border: 1px solid #3a3a4a;
  color: #e0e0e0;
}

.action-button.secondary:hover {
  background-color: #2a2a3a;
  border-color: #44ff44;
  color: #44ff44;
}
</style> 