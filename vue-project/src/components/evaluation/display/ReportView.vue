<template>
  <div class="chat-window report-view">
    <!-- 无数据时显示无信号 -->
    <div v-if="!evaluationResults" class="no-signal">
      <div class="static-effect"></div>
      <div class="no-signal-text">NO SIGNAL</div>
    </div>
    
    <!-- 有数据时显示报告 -->
    <div v-else class="report-container">
      <div class="report-header">
        <h2 class="report-title">评估报告</h2>
        <div class="report-actions">
          <button @click="exportReportCSV" class="crt-button export-btn">
            <span class="button-text">[ 导出报告(CSV) ]</span>
            <div class="button-icon">📊</div>
          </button>
        </div>
      </div>
      
      <!-- 总体评分 -->
      <div class="score-overview">
        <div class="score-card">
          <div class="score-value">{{ evaluationResults.overall_scores.final_score }}</div>
          <div class="score-label">总体评分</div>
        </div>
        <div class="score-card">
          <div class="score-value">{{ evaluationResults.overall_scores.role_score }}</div>
          <div class="score-label">角色评分</div>
        </div>
        <div class="score-card">
          <div class="score-value">{{ evaluationResults.overall_scores.dialogue_score }}</div>
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
        
        <!-- 角色扮演关键词词云 -->
        <div class="keywords-section">
          <h4>角色扮演关键词分析</h4>
          <div class="keywords-tabs">
            <button 
              v-for="(item, key) in rolePlayItems" 
              :key="`role-${key}`"
              @click="activeRoleKeywordTab = key"
              class="keyword-tab"
              :class="{ 'active': activeRoleKeywordTab === key }"
            >
              {{ item.label }}
            </button>
          </div>
          <div class="keywords-cloud">
            <!-- 这里可以集成词云组件 -->
            <div class="keywords-list">
              <div 
                v-for="(weight, word) in getKeywords('role_play', activeRoleKeywordTab)" 
                :key="word"
                class="keyword-item"
                :style="{ fontSize: `${Math.max(0.8, Math.min(2, weight * 0.1))}em` }"
              >
                {{ word }}
              </div>
            </div>
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
        
        <!-- 对话体验关键词词云 -->
        <div class="keywords-section">
          <h4>对话体验关键词分析</h4>
          <div class="keywords-tabs">
            <button 
              v-for="(item, key) in dialogueItems" 
              :key="`dialogue-${key}`"
              @click="activeDialogueKeywordTab = key"
              class="keyword-tab"
              :class="{ 'active': activeDialogueKeywordTab === key }"
            >
              {{ item.label }}
            </button>
          </div>
          <div class="keywords-cloud">
            <div class="keywords-list">
              <div 
                v-for="(weight, word) in getKeywords('dialogue_experience', activeDialogueKeywordTab)" 
                :key="word"
                class="keyword-item"
                :style="{ fontSize: `${Math.max(0.8, Math.min(2, weight * 0.1))}em` }"
              >
                {{ word }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { storeToRefs } from 'pinia'
import { useEvaluationStore } from '@/stores/evaluation'

const store = useEvaluationStore()
const { evaluationResults } = storeToRefs(store)

// 评估维度配置
const rolePlayItems = {
  role_consistency: { label: '角色一致性' },
  knowledge_application: { label: '知识应用' },
  personality_expression: { label: '性格表达' },
  interaction_quality: { label: '互动质量' }
}

const dialogueItems = {
  response_relevance: { label: '回应相关性' },
  language_quality: { label: '语言质量' },
  emotional_expression: { label: '情感表达' },
  conversation_flow: { label: '对话流畅度' }
}

// 关键词标签页状态
const activeRoleKeywordTab = ref('role_consistency')
const activeDialogueKeywordTab = ref('response_relevance')

// 获取评分值
const getScoreValue = (key, category) => {
  if (!evaluationResults.value || !evaluationResults.value[category] || !evaluationResults.value[category][key]) {
    return 0
  }
  return evaluationResults.value[category][key].avg
}

// 获取关键词
const getKeywords = (category, dimension) => {
  if (!evaluationResults.value || !evaluationResults.value[category] || !evaluationResults.value[category][dimension]) {
    return {}
  }
  return evaluationResults.value[category][dimension].keywords || {}
}

// 导出报告为CSV
const exportReportCSV = () => {
  if (!evaluationResults.value) return
  
  // 创建CSV内容
  const csvContent = [
    ['评估项目', '得分'],
    ['总体评分', evaluationResults.value.overall_scores.final_score],
    ['角色评分', evaluationResults.value.overall_scores.role_score],
    ['对话评分', evaluationResults.value.overall_scores.dialogue_score],
    [''],
    ['角色扮演评估'],
    ...Object.entries(rolePlayItems).map(([key, item]) => [
      item.label,
      getScoreValue(key, 'role_play')
    ]),
    [''],
    ['对话体验评估'],
    ...Object.entries(dialogueItems).map(([key, item]) => [
      item.label,
      getScoreValue(key, 'dialogue_experience')
    ])
  ].map(row => row.join(',')).join('\n')

  // 创建并下载文件
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = `evaluation_report_${new Date().toISOString()}.csv`
  link.click()
}
</script>

<style scoped>
.report-view {
  height: 100%;
  background: rgba(0, 0, 0, 0.8);
  border-radius: 10px;
  overflow: auto;
  padding: 1rem;
  position: relative;
}

.report-container {
  padding: 1rem;
}

.report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
}

.report-title {
  color: #44ff44;
  font-size: 1.5rem;
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.5);
}

.crt-button {
  background: none;
  border: 1px solid #44ff44;
  color: #44ff44;
  padding: 0.5rem 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  border-radius: 4px;
}

.crt-button:hover {
  background: rgba(68, 255, 68, 0.1);
  box-shadow: 0 0 10px rgba(68, 255, 68, 0.2);
}

.button-icon {
  font-size: 1.2em;
}

.score-overview {
  display: flex;
  justify-content: space-around;
  margin-bottom: 2rem;
  gap: 1rem;
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

.assessment-section h4 {
  color: #44ff44;
  margin-bottom: 1rem;
  font-size: 1.1rem;
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
  height: 8px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 4px;
  overflow: hidden;
}

.score-bar {
  height: 100%;
  background: linear-gradient(90deg, #44ff44, #7cff7c);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.score-bar-value {
  width: 40px;
  text-align: right;
  font-size: 0.9rem;
  color: #44ff44;
}

.keywords-section {
  margin-top: 1.5rem;
  background: rgba(20, 20, 30, 0.5);
  border-radius: 8px;
  padding: 1rem;
  border: 1px solid rgba(68, 255, 68, 0.2);
}

.keywords-tabs {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}

.keyword-tab {
  background: none;
  border: 1px solid rgba(68, 255, 68, 0.3);
  color: #44ff44;
  padding: 0.3rem 0.8rem;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 0.9rem;
}

.keyword-tab:hover {
  background: rgba(68, 255, 68, 0.1);
}

.keyword-tab.active {
  background: rgba(68, 255, 68, 0.2);
  box-shadow: 0 0 10px rgba(68, 255, 68, 0.2);
}

.keywords-cloud {
  min-height: 150px;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
}

.keywords-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
  justify-content: center;
  align-items: center;
}

.keyword-item {
  color: #44ff44;
  transition: all 0.3s ease;
  cursor: default;
}

.keyword-item:hover {
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.8);
}

.no-signal {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background: #000;
}

.static-effect {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: 
    repeating-radial-gradient(
      circle at 50% 50%,
      rgba(32, 32, 32, 0.98),
      rgba(32, 32, 32, 0.98) 2px,
      rgba(48, 48, 48, 0.98) 3px,
      rgba(48, 48, 48, 0.98) 4px
    );
  opacity: 0.15;
  animation: static 0.2s steps(4) infinite;
}

.no-signal-text {
  font-size: 2rem;
  font-weight: bold;
  color: rgba(255, 255, 255, 0.8);
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
  animation: flicker 0.3s ease infinite;
  z-index: 1;
}

@keyframes static {
  0% { transform: translate(0, 0); }
  25% { transform: translate(-1px, 1px); }
  50% { transform: translate(1px, -1px); }
  75% { transform: translate(-1px, -1px); }
  100% { transform: translate(1px, 1px); }
}

@keyframes flicker {
  0% { opacity: 1; }
  50% { opacity: 0.8; }
  100% { opacity: 1; }
}
</style> 