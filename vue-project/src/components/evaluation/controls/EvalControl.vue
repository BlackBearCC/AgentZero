<template>
  <div class="control-group">
    <!-- 评估类型选择 -->
    <div class="control-label">MODE</div>
    <div class="mode-selector">
      <button 
        @click="setEvalType('dialogue')" 
        class="control-button mode-btn" 
        :class="{ 'active': selectedEvalType === 'dialogue' }"
        :disabled="!isPoweredOn"
      >
        对话评估
      </button>
      <button 
        @click="setEvalType('memory')" 
        class="control-button mode-btn" 
        :class="{ 'active': selectedEvalType === 'memory' }"
        :disabled="!isPoweredOn"
      >
        记忆评估
      </button>
    </div>

    <!-- 人设信息输入 -->
    <div class="role-info-section">
      <div class="control-label">人设信息</div>
      <textarea 
        v-model="roleInfo" 
        placeholder="输入角色人设信息（可选）"
        class="role-info-input"
        rows="4"
        :disabled="!isPoweredOn"
      ></textarea>
    </div>

    <!-- 开始评估按钮 -->
    <div class="eval-actions">
      <div class="control-label">OPERATION</div>
      <button 
        @click="handleStartEvaluation"
        :disabled="!canStartEvaluation"
        class="control-button start-btn"
      >
        <div class="button-face">
          <span>开始评估</span>
          <div class="operation-indicator" :class="{ 'active': isEvaluating }"></div>
        </div>
      </button>
    </div>

    <!-- 进度条 -->
    <div v-if="isEvaluating" class="progress-section">
      <div class="control-label">PROGRESS</div>
      <div class="progress-bar">
        <div class="progress-fill" :style="progressStyle"></div>
      </div>
      <div class="progress-text">{{ processed }}/{{ total }}</div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { storeToRefs } from 'pinia'
import { useEvaluationStore } from '@/stores/evaluation'

const store = useEvaluationStore()
const { 
  selectedFile,
  fieldsConfirmed,
  isPoweredOn,
  evaluationInProgress,
  selectedEvalType,
  roleInfo,
  isEvaluating,
  processed,
  total
} = storeToRefs(store)

// 计算属性
const progressStyle = computed(() => ({
  width: `${(processed.value / total.value) * 100}%`
}))

const canStartEvaluation = computed(() => 
  isPoweredOn.value && 
  selectedFile.value && 
  fieldsConfirmed.value && 
  !evaluationInProgress.value
)

// 方法
const setEvalType = (type) => {
  selectedEvalType.value = type
}

const handleStartEvaluation = () => {
  if (!canStartEvaluation.value) return
  store.startEvaluation(selectedFile.value, store.selectedFields)
}

const startEvaluation = async (evaluationData) => {
  try {
    const formData = new FormData()
    formData.append('file', evaluationData.file)
    formData.append('fields', JSON.stringify(evaluationData.fields))
    formData.append('evaluationCode', evaluationData.evaluationCode)
    formData.append('evalType', evaluationData.evalType)
    formData.append('roleInfo', evaluationData.roleInfo || '')
    
    const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/start`, {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) throw new Error('评估请求失败')
    
    return await response.json()
  } catch (error) {
    throw new Error(`评估失败: ${error.message}`)
  }
}

// 事件
const emit = defineEmits(['start-evaluation', 'evaluation-progress'])

// 导出方法供父组件调用
defineExpose({
  setProgress: ({ processed: p, total: t }) => {
    processed.value = p
    total.value = t
  },
  setEvaluating: (value) => {
    isEvaluating.value = value
  },
  startEvaluation
})
</script>

<style scoped>
.control-group {
  margin-bottom: 1rem;
  padding: 0.5rem;
  border: 1px solid rgba(68, 255, 68, 0.3);
  background: rgba(0, 0, 0, 0.3);
  border-radius: 4px;
}

.control-label {
  font-size: 0.8rem;
  color: #44ff44;
  margin-bottom: 0.5rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  text-shadow: 0 0 5px rgba(68, 255, 68, 0.5);
}

.mode-selector {
  display: flex;
  gap: 0.5rem;
}

.mode-btn {
  flex: 1;
  padding: 0.5rem;
}

.mode-btn.active {
  background: rgba(var(--primary-color-rgb), 0.2);
  box-shadow: 0 0 15px var(--shadow-color);
}

.role-info-section {
  margin-top: 1rem;
}

.role-info-input {
  width: 100%;
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid var(--border-color);
  color: var(--primary-color);
  padding: 0.5rem;
  border-radius: 4px;
  resize: vertical;
  font-family: monospace;
}

.role-info-input:focus {
  outline: none;
  border-color: var(--primary-color);
  box-shadow: 0 0 5px var(--shadow-color);
}

.role-info-input:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.eval-actions {
  margin-top: 1rem;
}

.start-btn {
  width: 100%;
}

.operation-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #333;
  transition: all 0.3s ease;
}

.operation-indicator.active {
  background-color: #44ff44;
  box-shadow: 0 0 10px #44ff44;
  animation: pulse 2s infinite;
}

.progress-section {
  margin-top: 1rem;
}

.progress-bar {
  height: 10px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 5px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--primary-color), rgba(var(--primary-color-rgb), 0.7));
  border-radius: 5px;
  transition: width 0.3s ease;
}

.progress-text {
  text-align: center;
  font-size: 0.8rem;
  color: var(--text-secondary);
}

@keyframes pulse {
  0% { box-shadow: 0 0 5px #44ff44; }
  50% { box-shadow: 0 0 15px #44ff44; }
  100% { box-shadow: 0 0 5px #44ff44; }
}
</style> 