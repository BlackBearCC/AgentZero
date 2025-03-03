<template>
  <div class="control-section">
    <div class="section-title">评估控制</div>
    
    <!-- 评估模式 -->
    <div class="control-item">
      <div class="control-label">评估模式</div>
      <div class="mode-selector">
        <button 
          @click="evaluationMode = 'single'" 
          class="mode-button" 
          :class="{ 'active': evaluationMode === 'single' }"
        >
          单次评估
        </button>
        <button 
          @click="evaluationMode = 'batch'" 
          class="mode-button" 
          :class="{ 'active': evaluationMode === 'batch' }"
        >
          批量评估
        </button>
      </div>
    </div>
    
    <!-- 文件上传 (批量模式) -->
    <div class="control-item" v-if="evaluationMode === 'batch'">
      <div class="control-label">评估数据</div>
      <div class="file-upload">
        <input 
          type="file" 
          id="evaluation-file" 
          @change="handleFileChange" 
          accept=".csv,.xlsx,.json"
          class="file-input"
        />
        <label for="evaluation-file" class="file-label">
          <span class="file-button">选择文件</span>
          <span class="file-name">{{ fileName || '未选择文件' }}</span>
        </label>
      </div>
    </div>
    
    <!-- 单次评估输入 (单次模式) -->
    <div class="control-item" v-if="evaluationMode === 'single'">
      <div class="control-label">评估内容</div>
      <textarea 
        v-model="evaluationText" 
        class="evaluation-input" 
        placeholder="输入需要评估的内容..."
        rows="5"
      ></textarea>
    </div>
    
    <!-- 评估维度 -->
    <div class="control-item">
      <div class="control-label">评估维度</div>
      <div class="dimensions-group">
        <div class="dimension-item" v-for="(dimension, index) in dimensions" :key="index">
          <input 
            type="checkbox" 
            :id="`dimension-${index}`" 
            v-model="dimension.enabled" 
          />
          <label :for="`dimension-${index}`">{{ dimension.name }}</label>
        </div>
        <button @click="addCustomDimension" class="add-dimension-button">
          + 添加自定义维度
        </button>
        <div class="custom-dimension" v-if="showCustomDimension">
          <input 
            type="text" 
            v-model="customDimensionName" 
            class="custom-dimension-input" 
            placeholder="输入维度名称"
          />
          <button @click="confirmAddDimension" class="confirm-button">确认</button>
          <button @click="cancelAddDimension" class="cancel-button">取消</button>
        </div>
      </div>
    </div>
    
    <!-- 评估模型 -->
    <div class="control-item">
      <div class="control-label">评估模型</div>
      <select v-model="selectedModel" class="model-select">
        <option value="gpt-3.5">GPT-3.5</option>
        <option value="gpt-4">GPT-4</option>
        <option value="claude">Claude</option>
      </select>
    </div>
    
    <!-- 操作按钮 -->
    <div class="control-actions">
      <button @click="startEvaluation" class="action-button primary" :disabled="!canEvaluate">
        开始评估
      </button>
      <button @click="resetEvaluation" class="action-button secondary">
        重置
      </button>
    </div>
    
    <!-- 报告管理 (仅在评估中心频道) -->
    <div class="control-item report-management">
      <div class="control-label">报告管理</div>
      <div class="report-actions">
        <button @click="viewReport" class="report-button" :disabled="!hasReport">
          查看报告
        </button>
        <button @click="compareReports" class="report-button" :disabled="!hasMultipleReports">
          对比报告
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, defineEmits } from 'vue'

const emit = defineEmits(['update:status'])

// 状态变量
const evaluationMode = ref('single')
const fileName = ref('')
const selectedFile = ref(null)
const evaluationText = ref('')
const selectedModel = ref('gpt-4')
const isEvaluating = ref(false)
const hasReport = ref(false)
const hasMultipleReports = ref(false)

// 评估维度
const dimensions = ref([
  { name: '准确性', enabled: true },
  { name: '相关性', enabled: true },
  { name: '完整性', enabled: true },
  { name: '创新性', enabled: false },
  { name: '有害性', enabled: true }
])

// 自定义维度
const showCustomDimension = ref(false)
const customDimensionName = ref('')

// 计算属性
const canEvaluate = computed(() => {
  if (isEvaluating.value) return false
  
  if (evaluationMode.value === 'batch') {
    return selectedFile.value !== null
  } else {
    return evaluationText.value.trim() !== ''
  }
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

// 添加自定义维度
const addCustomDimension = () => {
  showCustomDimension.value = true
}

const confirmAddDimension = () => {
  if (customDimensionName.value.trim()) {
    dimensions.value.push({
      name: customDimensionName.value.trim(),
      enabled: true
    })
    customDimensionName.value = ''
    showCustomDimension.value = false
  }
}

const cancelAddDimension = () => {
  customDimensionName.value = ''
  showCustomDimension.value = false
}

// 开始评估
const startEvaluation = () => {
  if (!canEvaluate.value) return
  
  isEvaluating.value = true
  emit('update:status', '正在进行评估...')
  
  // 这里应该调用父组件的方法来实际开始评估过程
  // 在实际应用中，你可能需要使用 refs 或事件来与父组件通信
  
  // 模拟评估完成
  setTimeout(() => {
    isEvaluating.value = false
    hasReport.value = true
    hasMultipleReports.value = true // 假设已有多个报告
    emit('update:status', '评估完成')
  }, 3000)
}

// 重置评估
const resetEvaluation = () => {
  if (evaluationMode.value === 'batch') {
    selectedFile.value = null
    fileName.value = ''
  } else {
    evaluationText.value = ''
  }
  isEvaluating.value = false
  emit('update:status', '系统就绪')
}

// 查看报告
const viewReport = () => {
  if (!hasReport.value) return
  emit('update:status', '正在加载报告...')
  // 这里应该触发父组件切换到报告视图
}

// 对比报告
const compareReports = () => {
  if (!hasMultipleReports.value) return
  emit('update:status', '正在加载报告对比...')
  // 这里应该触发父组件切换到报告对比视图
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

.mode-selector {
  display: flex;
  gap: 10px;
}

.mode-button {
  flex: 1;
  background-color: #1a1a2a;
  border: 1px solid #3a3a4a;
  border-radius: 5px;
  padding: 8px 12px;
  color: #e0e0e0;
  cursor: pointer;
  transition: all 0.3s ease;
}

.mode-button.active {
  background-color: rgba(68, 255, 68, 0.2);
  border-color: #44ff44;
  color: #44ff44;
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

.evaluation-input {
  width: 100%;
  background-color: #1a1a2a;
  border: 1px solid #3a3a4a;
  border-radius: 5px;
  padding: 8px;
  color: #e0e0e0;
  resize: vertical;
  font-family: inherit;
}

.evaluation-input:focus {
  outline: none;
  border-color: #44ff44;
}

.dimensions-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.dimension-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.dimension-item input[type="checkbox"] {
  appearance: none;
  width: 16px;
  height: 16px;
  border: 1px solid #3a3a4a;
  border-radius: 3px;
  background-color: #1a1a2a;
  cursor: pointer;
  position: relative;
}

.dimension-item input[type="checkbox"]:checked {
  background-color: #44ff44;
  border-color: #44ff44;
}

.dimension-item input[type="checkbox"]:checked::after {
  content: '✓';
  position: absolute;
  color: #000;
  font-size: 12px;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

.dimension-item label {
  cursor: pointer;
}

.add-dimension-button {
  background: none;
  border: none;
  color: #44ff44;
  cursor: pointer;
  padding: 5px 0;
  text-align: left;
  font-size: 0.9rem;
}

.add-dimension-button:hover {
  text-decoration: underline;
}

.custom-dimension {
  display: flex;
  gap: 5px;
  margin-top: 5px;
}

.custom-dimension-input {
  flex: 1;
  background-color: #1a1a2a;
  border: 1px solid #3a3a4a;
  border-radius: 5px;
  padding: 5px 8px;
  color: #e0e0e0;
}

.custom-dimension-input:focus {
  outline: none;
  border-color: #44ff44;
}

.confirm-button, .cancel-button {
  background-color: #1a1a2a;
  border: 1px solid #3a3a4a;
  border-radius: 5px;
  padding: 5px 8px;
  cursor: pointer;
}

.confirm-button {
  color: #44ff44;
}

.confirm-button:hover {
  background-color: rgba(68, 255, 68, 0.2);
}

.cancel-button {
  color: #ff4444;
}

.cancel-button:hover {
  background-color: rgba(255, 68, 68, 0.2);
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

.report-management {
  margin-top: 10px;
  padding-top: 15px;
  border-top: 1px solid #3a3a4a;
}

.report-actions {
  display: flex;
  gap: 10px;
}

.report-button {
  flex: 1;
  background-color: #1a1a2a;
  border: 1px solid #3a3a4a;
  border-radius: 5px;
  padding: 8px 12px;
  color: #e0e0e0;
  cursor: pointer;
  transition: all 0.3s ease;
}

.report-button:hover:not(:disabled) {
  background-color: #2a2a3a;
  border-color: #44ff44;
  color: #44ff44;
}

.report-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style> 