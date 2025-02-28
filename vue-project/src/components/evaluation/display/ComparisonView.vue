<template>
  <div class="chat-window report-comparison">
    <!-- 无数据时显示无信号 -->
    <div v-if="!savedReports.length" class="no-signal">
      <div class="static-effect"></div>
      <div class="no-signal-text">NO SIGNAL</div>
    </div>

    <!-- 报告列表 -->
    <div v-else class="reports-container">
      <div class="reports-list">
        <div 
          v-for="report in savedReports" 
          :key="report.id"
          class="report-item"
          :class="{ 'selected': selectedReports.includes(report.id) }"
          @click="toggleReportSelection(report.id)"
        >
          <div class="report-info">
            <div class="report-code">{{ report.evaluation_code }}</div>
            <div class="report-time">{{ formatTime(report.timestamp) }}</div>
          </div>
          <div class="report-score">
            {{ report.stats.overall_scores.final_score }}
          </div>
        </div>
      </div>

      <!-- 对比视图 -->
      <div v-if="selectedReports.length >= 2" class="comparison-view">
        <h3>评分对比</h3>
        
        <!-- 总体评分对比 -->
        <div class="comparison-section">
          <h4>总体评分</h4>
          <div class="comparison-bars">
            <div 
              v-for="reportId in selectedReports" 
              :key="reportId"
              class="comparison-bar-item"
            >
              <div class="comparison-label">{{ getReportById(reportId).evaluation_code }}</div>
              <div class="comparison-bar-container">
                <div 
                  class="comparison-bar" 
                  :style="{ 
                    width: `${getReportById(reportId).stats.overall_scores.final_score}%`,
                    backgroundColor: getReportColor(reportId)
                  }"
                ></div>
              </div>
              <div class="comparison-value">
                {{ getReportById(reportId).stats.overall_scores.final_score }}
              </div>
            </div>
          </div>
        </div>

        <!-- 角色扮演评分对比 -->
        <div class="comparison-section">
          <h4>角色扮演评分</h4>
          <div class="dimension-tabs">
            <button 
              v-for="(item, key) in rolePlayItems" 
              :key="key"
              @click="activeRoleTab = key"
              class="dimension-tab"
              :class="{ 'active': activeRoleTab === key }"
            >
              {{ item.label }}
            </button>
          </div>
          <div class="comparison-bars">
            <div 
              v-for="reportId in selectedReports" 
              :key="`role-${reportId}`"
              class="comparison-bar-item"
            >
              <div class="comparison-label">{{ getReportById(reportId).evaluation_code }}</div>
              <div class="comparison-bar-container">
                <div 
                  class="comparison-bar" 
                  :style="{ 
                    width: `${getReportDimensionScore(reportId, 'role_play', activeRoleTab)}%`,
                    backgroundColor: getReportColor(reportId)
                  }"
                ></div>
              </div>
              <div class="comparison-value">
                {{ getReportDimensionScore(reportId, 'role_play', activeRoleTab) }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { storeToRefs } from 'pinia'
import { useEvaluationStore } from '@/stores/evaluation'

const store = useEvaluationStore()
const { savedReports, selectedReports } = storeToRefs(store)

const activeRoleTab = ref('consistency')

const rolePlayItems = {
  consistency: { label: '角色一致性' },
  personality: { label: '性格特征' },
  background: { label: '背景知识' },
  language: { label: '语言风格' }
}

// 获取报告颜色
const reportColors = ['#44ff44', '#ff4444', '#4444ff']
const getReportColor = (reportId) => {
  const index = selectedReports.value.indexOf(reportId)
  return reportColors[index] || reportColors[0]
}

// 格式化时间
const formatTime = (timestamp) => {
  return new Date(timestamp).toLocaleString()
}

// 根据ID获取报告
const getReportById = (id) => {
  return savedReports.value.find(report => report.id === id)
}

// 获取维度分数
const getReportDimensionScore = (reportId, category, dimension) => {
  const report = getReportById(reportId)
  return report?.stats[category][dimension]?.avg || 0
}

// 切换报告选择
const toggleReportSelection = (reportId) => {
  const index = selectedReports.value.indexOf(reportId)
  if (index === -1 && selectedReports.value.length < 3) {
    selectedReports.value.push(reportId)
  } else if (index !== -1) {
    selectedReports.value.splice(index, 1)
  }
}
</script>

<style scoped>
.report-comparison {
  height: 100%;
  overflow-y: auto;
  padding: 1rem;
}

.reports-container {
  display: flex;
  gap: 2rem;
  height: 100%;
}

.reports-list {
  width: 250px;
  border-right: 1px solid rgba(68, 255, 68, 0.2);
  padding-right: 1rem;
}

.report-item {
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 4px;
  padding: 0.8rem;
  margin-bottom: 0.8rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.report-item:hover {
  background: rgba(68, 255, 68, 0.1);
}

.report-item.selected {
  background: rgba(68, 255, 68, 0.2);
  border-color: #44ff44;
  box-shadow: 0 0 10px rgba(68, 255, 68, 0.3);
}

.report-info {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.report-code {
  color: #44ff44;
  font-weight: bold;
}

.report-time {
  color: #8a8a9a;
  font-size: 0.8rem;
}

.report-score {
  font-size: 1.5rem;
  color: #44ff44;
  text-align: center;
  text-shadow: 0 0 5px rgba(68, 255, 68, 0.5);
}

.comparison-view {
  flex: 1;
  padding: 1rem;
}

.comparison-section {
  margin-bottom: 2rem;
}

h3, h4 {
  color: #44ff44;
  margin-bottom: 1rem;
  text-shadow: 0 0 5px rgba(68, 255, 68, 0.3);
}

.dimension-tabs {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}

.dimension-tab {
  background: none;
  border: 1px solid rgba(68, 255, 68, 0.3);
  color: #44ff44;
  padding: 0.5rem 1rem;
  cursor: pointer;
  border-radius: 4px;
  transition: all 0.3s ease;
}

.dimension-tab:hover {
  background: rgba(68, 255, 68, 0.1);
}

.dimension-tab.active {
  background: rgba(68, 255, 68, 0.2);
  border-color: #44ff44;
  box-shadow: 0 0 10px rgba(68, 255, 68, 0.3);
}

.comparison-bars {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.comparison-bar-item {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.comparison-label {
  width: 100px;
  color: #a0a0a0;
  text-align: right;
}

.comparison-bar-container {
  flex: 1;
  height: 20px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 10px;
  overflow: hidden;
}

.comparison-bar {
  height: 100%;
  transition: width 0.3s ease;
}

.comparison-value {
  width: 50px;
  color: #44ff44;
  text-align: right;
}

.static-effect {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: repeating-radial-gradient(
    circle at 50% 50%,
    rgba(32, 32, 32, 0.98),
    rgba(32, 32, 32, 0.98) 2px,
    rgba(48, 48, 48, 0.98) 3px,
    rgba(48, 48, 48, 0.98) 4px
  );
  opacity: 0.15;
  animation: static 0.2s steps(4) infinite;
}

@keyframes static {
  0% { transform: translate(0, 0); }
  25% { transform: translate(-1px, 1px); }
  50% { transform: translate(1px, -1px); }
  75% { transform: translate(-1px, -1px); }
  100% { transform: translate(1px, 1px); }
}
</style> 