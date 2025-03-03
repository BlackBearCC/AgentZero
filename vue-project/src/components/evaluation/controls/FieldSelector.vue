<template>
  <div class="control-group field-selector" v-if="availableFields.length > 0">
    <div class="control-label">
      FIELDS 
      <span class="field-count">{{ selectedFields.length }}/{{ availableFields.length }}</span>
    </div>
    
    <!-- 字段列表 - 垂直排列 -->
    <div class="field-list">
      <div v-for="field in availableFields" :key="field" class="field-item">
        <label class="field-label">
          <input 
            type="checkbox" 
            v-model="selectedFieldsLocal" 
            :value="field"
            @change="handleFieldChange"
          >
          <span class="field-name">{{ field }}</span>
        </label>
      </div>
    </div>
    
    <!-- 操作按钮 -->
    <div class="field-actions">
      <button 
        @click="confirmFields" 
        class="control-button confirm-fields-btn"
        :disabled="selectedFieldsLocal.length === 0"
      >
        <div class="button-face">
          <span>确认字段</span>
          <div v-if="fieldsConfirmed" class="confirm-indicator">✓</div>
        </div>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { useEvaluationStore } from '@/stores/evaluation'

const store = useEvaluationStore()
const { availableFields, selectedFields, fieldsConfirmed } = storeToRefs(store)

// 本地状态
const selectedFieldsLocal = ref([])

// 监听 store 中的 selectedFields 变化
watch(() => selectedFields.value, (newValue) => {
  selectedFieldsLocal.value = [...newValue]
}, { immediate: true })

// 处理字段选择变化
const handleFieldChange = () => {
  store.setSelectedFields([...selectedFieldsLocal.value])
}

// 确认字段选择
const confirmFields = () => {
  if (selectedFieldsLocal.value.length > 0) {
    store.setFieldsConfirmed(true)
  }
}

// 导出组件事件
defineEmits(['fields-confirmed'])
</script>

<style scoped>
.field-selector {
  margin-bottom: 1rem;
}

.field-count {
  font-size: 0.7rem;
  color: rgba(var(--primary-color-rgb), 0.7);
  margin-left: 0.5rem;
}

.field-list {
  max-height: 200px;
  overflow-y: auto;
  margin: 0.5rem 0;
  padding-right: 0.5rem;
  scrollbar-width: thin;
  scrollbar-color: #44ff44 rgba(0, 0, 0, 0.3);
}

.field-list::-webkit-scrollbar {
  width: 6px;
}

.field-list::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.3);
  border-radius: 3px;
}

.field-list::-webkit-scrollbar-thumb {
  background-color: #44ff44;
  border-radius: 3px;
}

.field-item {
  margin-bottom: 0.3rem;
}

.field-label {
  display: flex;
  align-items: center;
  padding: 0.4rem;
  background: rgba(40, 40, 50, 0.8);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
  width: 100%;
}

.field-label:hover {
  background: rgba(50, 50, 60, 0.8);
}

.field-label input[type="checkbox"] {
  appearance: none;
  -webkit-appearance: none;
  width: 16px;
  height: 16px;
  background: #2a2a3a;
  border: 1px solid var(--primary-color);
  border-radius: 3px;
  margin-right: 0.5rem;
  position: relative;
  cursor: pointer;
  flex-shrink: 0;
}

.field-label input[type="checkbox"]:checked {
  background: var(--primary-color);
}

.field-label input[type="checkbox"]:checked::after {
  content: '✓';
  position: absolute;
  color: var(--bg-darker);
  font-size: 12px;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

.field-name {
  color: var(--text-color);
  font-size: 0.85rem;
  word-break: break-all;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.field-actions {
  margin-top: 1rem;
  text-align: right;
}

.confirm-fields-btn {
  min-width: 100px;
}

.confirm-indicator {
  display: inline-block;
  margin-left: 0.5rem;
  color: var(--primary-color);
  font-weight: bold;
}

/* 继承控制组样式 */
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

.control-button {
  width: 100%;
  background: none;
  border: 1px solid #44ff44;
  color: #44ff44;
  padding: 0.5rem;
  cursor: pointer;
  transition: all 0.3s ease;
  border-radius: 4px;
}

.control-button:hover:not(:disabled) {
  background: rgba(68, 255, 68, 0.1);
  box-shadow: 0 0 10px rgba(68, 255, 68, 0.2);
}

.control-button:active:not(:disabled) {
  transform: scale(0.98);
}

.control-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  border-color: rgba(68, 255, 68, 0.3);
}
</style> 