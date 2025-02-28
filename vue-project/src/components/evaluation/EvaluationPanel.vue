<template>
  <div class="tv-container crt-effect" v-if="isPoweredOn">
    <!-- 左侧控制面板 -->
    <div class="control-panel">
      <div class="panel-title">控制中心</div>
      
      <!-- 控制组件 -->
      <PowerControl 
        :is-powered-on="isPoweredOn"
        @power-change="handlePowerChange" 
      />
      
      <ChannelControl 
        :active-channel="activeChannel"
        :is-changing="isChangingChannel"
        @channel-change="handleChannelChange" 
      />
      
      <FileControl 
        ref="fileControlRef"
        :system-message="systemMessage"
        @file-selected="handleFileSelected"
      />
      
      <FieldSelector 
        v-if="availableFields.length > 0"
        :available-fields="availableFields"
        :selected-fields="selectedFields"
        :fields-confirmed="fieldsConfirmed"
        @fields-updated="handleFieldsUpdated"
        @fields-confirmed="handleFieldsConfirmed"
      />
      
      <EvalControl 
        ref="evalControlRef"
        :can-start-evaluation="canStartEvaluation"
        :is-evaluating="isScanning"
        @evaluation-start="handleEvaluationStart"
        @evaluation-complete="handleEvaluationComplete"
      />
    </div>

    <!-- 右侧显示区域 -->
    <div class="display-panel">
      <Screen 
        :is-scanning="isScanning"
        :is-changing-channel="isChangingChannel"
      >
        <EvaluationView 
          v-if="activeChannel === 1"
          :system-message="systemMessage"
          :evaluation-text="evaluationText"
        />
        <ReportView 
          v-else-if="activeChannel === 2"
          :evaluation-stats="currentReport"
          @export-report="handleExportReport"
        />
        <ComparisonView 
          v-else-if="activeChannel === 3"
          :evaluation-history="evaluationHistory"
          :selected-reports="selectedReportsForComparison"
          @report-selected="handleReportSelection"
          @export-comparison="handleExportComparison"
        />
      </Screen>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { storeToRefs } from 'pinia'
import { useEvaluationStore } from '@/stores/evaluation'

// 导入子组件
import PowerControl from './controls/PowerControl.vue'
import ChannelControl from './controls/ChannelControl.vue'
import FileControl from './controls/FileControl.vue'
import FieldSelector from './controls/FieldSelector.vue'
import EvalControl from './controls/EvalControl.vue'
import Screen from './display/Screen.vue'
import EvaluationView from './display/EvaluationView.vue'
import ReportView from './display/ReportView.vue'
import ComparisonView from './display/ComparisonView.vue'

// 初始化 store
const store = useEvaluationStore()

// 使用 storeToRefs 解构状态，保持响应性
const {
  isPoweredOn,
  activeChannel,
  isChangingChannel,
  isScanning,
  availableFields,
  selectedFields,
  fieldsConfirmed,
  systemMessage,
  evaluationHistory,
  selectedReportsForComparison
} = storeToRefs(store)

// 组件引用
const fileControlRef = ref(null)
const evalControlRef = ref(null)

// 计算属性
const canStartEvaluation = computed(() => 
  isPoweredOn.value && 
  selectedFile.value && // 这里的 selectedFile 是从 store 获取的，但 store 中并没有这个状态
  fieldsConfirmed.value && 
  !evaluationInProgress.value
)

const currentReport = computed(() => {
  return evaluationHistory.value[evaluationHistory.value.length - 1]
})

// 事件处理函数
const handlePowerChange = (status) => {
  if (status) {
    store.initializeSystem()
  } else {
    store.resetAllStates()
  }
}

const handleChannelChange = (channel) => {
  store.setChannelChanging(true)
  setTimeout(() => {
    store.setActiveChannel(channel)
    store.setChannelChanging(false)
  }, 1000)
}

const handleFileSelected = (fields) => {
  store.setAvailableFields(fields)
}

const handleFieldsUpdated = (fields) => {
  store.setSelectedFields(fields)
}

const handleFieldsConfirmed = () => {
  store.setFieldsConfirmed(true)
}

const handleEvaluationStart = async () => {
  if (!canStartEvaluation.value) return
  
  store.setSystemMessage('评估开始...')
  store.setScanning(true)
  
  try {
    const evaluationData = {
      file: fileControlRef.value.selectedFile,
      fields: selectedFields.value,
      evaluationCode: fileControlRef.value.evaluationCode,
      evalType: selectedEvalType.value,
      roleInfo: roleInfo.value
    }
    
    const result = await evalControlRef.value.startEvaluation(evaluationData)
    
    store.saveReport(result)
    store.setSystemMessage('评估完成')
    handleChannelChange(2) // 自动切换到报告视图
  } catch (error) {
    store.setSystemMessage('评估失败：' + error.message)
  } finally {
    store.setScanning(false)
  }
}

const handleEvaluationComplete = () => {
  store.setScanning(false)
}

const handleExportReport = () => {
  if (currentReport.value) {
    evalControlRef.value.exportReport(currentReport.value)
  }
}

const handleReportSelection = (reportId) => {
  store.toggleReportSelection(reportId)
}

const handleExportComparison = () => {
  if (selectedReportsForComparison.value.length >= 2) {
    evalControlRef.value.exportComparison(
      selectedReportsForComparison.value.map(id => 
        evaluationHistory.value.find(report => report.id === id)
      )
    )
  }
}
</script>

<style scoped>
/* 布局相关样式 */
.tv-container {
  display: flex;
  gap: 2rem;
  padding: 2rem;
  background: #1a1a1a;
  min-height: 100vh;
  width: 100vw;
  color: #44ff44;
  position: fixed;
  top: 0;
  left: 0;
  box-sizing: border-box;
}

.control-panel {
  width: 300px;
  min-width: 300px;
  background: rgba(0, 0, 0, 0.5);
  padding: 1.5rem;
  border-radius: 10px;
  border: 1px solid rgba(68, 255, 68, 0.3);
  box-shadow: 0 0 20px rgba(68, 255, 68, 0.1);
  height: calc(100vh - 4rem);
  overflow-y: auto;
}

.panel-title {
  font-size: 1.2rem;
  text-align: center;
  margin-bottom: 2rem;
  text-transform: uppercase;
  letter-spacing: 0.2em;
  color: #44ff44;
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.5);
}

.display-panel {
  flex: 1;
  min-height: calc(100vh - 4rem);
  display: flex;
  flex-direction: column;
  position: relative;
  overflow: hidden;
}
</style>