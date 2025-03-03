<template>
  <div class="evaluation-center">
    <!-- 评估子频道导航 -->
    <div class="subchannel-nav">
      <button 
        @click="activeSubchannel = 'process'" 
        class="subchannel-btn" 
        :class="{ 'active': activeSubchannel === 'process' }"
      >
        评估处理
      </button>
      <button 
        @click="activeSubchannel = 'report'" 
        class="subchannel-btn" 
        :class="{ 'active': activeSubchannel === 'report' }"
      >
        评估报告
      </button>
      <button 
        @click="activeSubchannel = 'compare'" 
        class="subchannel-btn" 
        :class="{ 'active': activeSubchannel === 'compare' }"
      >
        报告对比
      </button>
    </div>
    
    <!-- 评估处理子频道 -->
    <div v-if="activeSubchannel === 'process'" class="eval-subchannel">
      <div v-if="!evaluationText && !systemMessage" class="empty-state">
        <div class="tv-logo">AI EVALUATOR</div>
        <div class="channel-info">频道 3</div>
        <div class="instruction-text">请使用左侧控制面板上传评估文件并开始处理</div>
      </div>
      
      <div v-else-if="systemMessage && !evaluationText" class="standby-state">
        <div class="tv-logo">AI EVALUATOR</div>
        <div class="standby-message">{{ systemMessage }}</div>
        <div class="standby-animation"></div>
      </div>
      
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
            <pre class="typewriter-text">{{ evaluationText }}<span class="cursor" :class="{ 'blink': !isTyping }">|</span></pre>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 评估报告子频道 -->
    <div v-if="activeSubchannel === 'report'" class="eval-subchannel report-view">
      <div v-if="!evaluationStats" class="empty-state">
        <div class="tv-logo">EVALUATION REPORT</div>
        <div class="instruction-text">请先完成评估处理以生成报告</div>
      </div>
      
      <div v-else class="report-container">
        <div class="report-header">
          <h2>评估报告</h2>
          <div class="report-actions">
            <button @click="exportReport" class="crt-button">
              <span class="button-text">[ 导出报告 ]</span>
            </button>
          </div>
        </div>
        
        <div class="report-content">
          <div class="report-section">
            <h3>总体评分</h3>
            <div class="score-display">
              <div class="score-value">{{ evaluationStats.overall.score }}</div>
              <div class="score-bar-container">
                <div class="score-bar" :style="{ width: `${evaluationStats.overall.score}%` }"></div>
              </div>
            </div>
          </div>
          
          <div class="report-section">
            <h3>维度评分</h3>
            <div class="dimensions-list">
              <div v-for="(dimension, index) in evaluationStats.dimensions" :key="index" class="dimension-item">
                <div class="dimension-name">{{ dimension.name }}</div>
                <div class="dimension-score">{{ dimension.score }}</div>
                <div class="dimension-bar-container">
                  <div class="dimension-bar" :style="{ width: `${dimension.score}%` }"></div>
                </div>
              </div>
            </div>
          </div>
          
          <div class="report-section">
            <h3>评估摘要</h3>
            <div class="summary-content">
              <p>{{ evaluationStats.summary }}</p>
            </div>
          </div>
          
          <div class="report-section">
            <h3>改进建议</h3>
            <div class="suggestions-list">
              <div v-for="(suggestion, index) in evaluationStats.suggestions" :key="index" class="suggestion-item">
                <div class="suggestion-number">{{ index + 1 }}</div>
                <div class="suggestion-content">{{ suggestion }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 报告对比子频道 -->
    <div v-if="activeSubchannel === 'compare'" class="eval-subchannel compare-view">
      <div v-if="!comparisonData.length" class="empty-state">
        <div class="tv-logo">REPORT COMPARISON</div>
        <div class="instruction-text">请使用左侧控制面板选择要对比的报告</div>
      </div>
      
      <div v-else class="comparison-container">
        <div class="comparison-header">
          <h2>报告对比</h2>
          <div class="comparison-actions">
            <button @click="exportComparison" class="crt-button">
              <span class="button-text">[ 导出对比结果 ]</span>
            </button>
          </div>
        </div>
        
        <div class="comparison-content">
          <div class="comparison-section">
            <h3>总体评分对比</h3>
            <div class="comparison-chart">
              <div v-for="(report, index) in comparisonData" :key="index" class="comparison-item">
                <div class="comparison-label">{{ report.name }}</div>
                <div class="comparison-score">{{ report.overall.score }}</div>
                <div class="comparison-bar-container">
                  <div class="comparison-bar" :style="{ width: `${report.overall.score}%`, backgroundColor: getReportColor(index) }"></div>
                </div>
              </div>
            </div>
          </div>
          
          <div class="comparison-section">
            <h3>维度评分对比</h3>
            <div class="dimensions-comparison">
              <table class="comparison-table">
                <thead>
                  <tr>
                    <th>维度</th>
                    <th v-for="(report, index) in comparisonData" :key="index">
                      {{ report.name }}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(dimension, dimIndex) in getDimensionNames()" :key="dimIndex">
                    <td>{{ dimension }}</td>
                    <td v-for="(report, repIndex) in comparisonData" :key="repIndex">
                      <div class="table-score-container">
                        <span class="table-score">{{ getDimensionScore(report, dimension) }}</span>
                        <div class="table-bar-container">
                          <div class="table-bar" 
                               :style="{ width: `${getDimensionScore(report, dimension)}%`, backgroundColor: getReportColor(repIndex) }">
                          </div>
                        </div>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
          
          <div class="comparison-section">
            <h3>差异分析</h3>
            <div class="difference-analysis">
              <p>{{ differenceAnalysis }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, defineEmits } from 'vue'

const emit = defineEmits(['scanning:start', 'scanning:stop'])

// 子频道状态
const activeSubchannel = ref('process')

// 评估处理状态
const systemMessage = ref('')
const evaluationText = ref('')
const isTyping = ref(false)

// 评估报告状态
const evaluationStats = ref(null)

// 报告对比状态
const comparisonData = ref([])
const differenceAnalysis = ref('')

// 报告颜色
const reportColors = ['#44ff44', '#44aaff', '#ff44aa', '#ffaa44']

// 获取报告颜色
const getReportColor = (index) => {
  return reportColors[index % reportColors.length]
}

// 获取所有维度名称
const getDimensionNames = () => {
  if (!comparisonData.value.length) return []
  
  // 收集所有报告中的维度名称
  const dimensionSet = new Set()
  comparisonData.value.forEach(report => {
    report.dimensions.forEach(dim => {
      dimensionSet.add(dim.name)
    })
  })
  
  return Array.from(dimensionSet)
}

// 获取特定报告中特定维度的分数
const getDimensionScore = (report, dimensionName) => {
  const dimension = report.dimensions.find(dim => dim.name === dimensionName)
  return dimension ? dimension.score : 0
}

// 导出报告
const exportReport = () => {
  if (!evaluationStats.value) return
  
  const reportData = JSON.stringify(evaluationStats.value, null, 2)
  const blob = new Blob([reportData], { type: 'application/json' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = `evaluation_report_${new Date().toISOString().slice(0,10)}.json`
  link.click()
  URL.revokeObjectURL(link.href)
}

// 导出对比结果
const exportComparison = () => {
  if (!comparisonData.value.length) return
  
  const comparisonReport = {
    reports: comparisonData.value,
    analysis: differenceAnalysis.value,
    timestamp: new Date().toISOString()
  }
  
  const reportData = JSON.stringify(comparisonReport, null, 2)
  const blob = new Blob([reportData], { type: 'application/json' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = `comparison_report_${new Date().toISOString().slice(0,10)}.json`
  link.click()
  URL.revokeObjectURL(link.href)
}

// 暴露给父组件的方法
defineExpose({
  // 开始评估处理
  startEvaluation(text) {
    activeSubchannel.value = 'process'
    systemMessage.value = '正在处理评估...'
    evaluationText.value = ''
    emit('scanning:start')
    
    // 模拟评估处理
    setTimeout(() => {
      isTyping.value = true
      
      // 模拟打字效果
      const fullText = `评估结果：

该回复整体表现良好，但存在一些需要改进的地方。

优点：
1. 回答了用户的核心问题
2. 提供了详细的解释和背景信息
3. 语言表达清晰，逻辑结构合理

不足：
1. 部分专业术语未做解释，可能影响用户理解
2. 缺少具体的实例来支持观点
3. 回复过长，核心信息可以更加突出

建议：
- 增加实例说明，使抽象概念更容易理解
- 精简非核心内容，突出重点信息
- 为专业术语提供简明解释`
      
      let currentIndex = 0
      const typingInterval = setInterval(() => {
        if (currentIndex < fullText.length) {
          evaluationText.value = fullText.substring(0, currentIndex + 1)
          currentIndex++
        } else {
          clearInterval(typingInterval)
          isTyping.value = false
          
          // 生成评估报告数据
          setTimeout(() => {
            evaluationStats.value = {
              overall: { score: 78 },
              dimensions: [
                { name: '准确性', score: 85 },
                { name: '清晰度', score: 75 },
                { name: '相关性', score: 90 },
                { name: '完整性', score: 70 },
                { name: '简洁性', score: 65 }
              ],
              summary: '该回复整体表现良好，在相关性和准确性方面表现突出，但在简洁性和完整性方面有待提高。',
              suggestions: [
                '增加实例说明，使抽象概念更容易理解',
                '精简非核心内容，突出重点信息',
                '为专业术语提供简明解释',
                '考虑添加视觉元素如图表或列表来提高信息传达效率'
              ]
            }
            
            emit('scanning:stop')
          }, 1000)
        }
      }, 20)
    }, 1500)
  },
  
  // 添加对比报告
  addComparisonReport(reportName) {
    activeSubchannel.value = 'compare'
    emit('scanning:start')
    
    // 模拟添加报告
    setTimeout(() => {
      // 生成随机报告数据
      const newReport = {
        name: reportName,
        overall: { score: Math.floor(Math.random() * 30) + 60 },
        dimensions: [
          { name: '准确性', score: Math.floor(Math.random() * 30) + 60 },
          { name: '清晰度', score: Math.floor(Math.random() * 30) + 60 },
          { name: '相关性', score: Math.floor(Math.random() * 30) + 60 },
          { name: '完整性', score: Math.floor(Math.random() * 30) + 60 },
          { name: '简洁性', score: Math.floor(Math.random() * 30) + 60 }
        ]
      }
      
      comparisonData.value.push(newReport)
      
      // 如果有多个报告，生成差异分析
      if (comparisonData.value.length > 1) {
        differenceAnalysis.value = `对比分析显示，${comparisonData.value.map(r => r.name).join('和')}在评估维度上存在明显差异。
        
${comparisonData.value[0].name}在${getHighestDimension(0)}方面表现最佳，而${comparisonData.value[1].name}在${getHighestDimension(1)}方面表现突出。

总体而言，${comparisonData.value[0].overall.score > comparisonData.value[1].overall.score ? comparisonData.value[0].name : comparisonData.value[1].name}的整体评分更高，建议重点关注${getLowestDimension(comparisonData.value[0].overall.score > comparisonData.value[1].overall.score ? 0 : 1)}方面的提升。`
      }
      
      emit('scanning:stop')
    }, 1500)
  },
  
  // 重置评估
  reset() {
    activeSubchannel.value = 'process'
    systemMessage.value = ''
    evaluationText.value = ''
    evaluationStats.value = null
    comparisonData.value = []
    differenceAnalysis.value = ''
  }
})

// 获取报告中得分最高的维度
const getHighestDimension = (reportIndex) => {
  if (!comparisonData.value[reportIndex]) return ''
  
  const dimensions = comparisonData.value[reportIndex].dimensions
  let highest = dimensions[0]
  
  dimensions.forEach(dim => {
    if (dim.score > highest.score) {
      highest = dim
    }
  })
  
  return highest.name
}

// 获取报告中得分最低的维度
const getLowestDimension = (reportIndex) => {
  if (!comparisonData.value[reportIndex]) return ''
  
  const dimensions = comparisonData.value[reportIndex].dimensions
  let lowest = dimensions[0]
  
  dimensions.forEach(dim => {
    if (dim.score < lowest.score) {
      lowest = dim
    }
  })
  
  return lowest.name
}
</script>

<style scoped>
.evaluation-center {
  height: 100%;
  padding: 20px;
  overflow-y: auto;
  color: #e0e0e0;
  display: flex;
  flex-direction: column;
}

/* 子频道导航 */
.subchannel-nav {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  background: rgba(0, 0, 0, 0.5);
  padding: 0.5rem;
  border-radius: 5px;
}

.subchannel-btn {
  background: #2a2a3a;
  border: 1px solid #3a3a4a;
  color: #8a8a9a;
  padding: 0.5rem 1rem;
  border-radius: 3px;
  cursor: pointer;
  transition: all 0.3s ease;
  flex: 1;
  text-align: center;
}

.subchannel-btn.active {
  background: rgba(68, 255, 68, 0.2);
  border: 1px solid #44ff44;
  color: #44ff44;
  text-shadow: 0 0 5px #44ff44;
}

.eval-subchannel {
  flex: 1;
  overflow-y: auto;
}

/* 空状态 */
.empty-state, .standby-state {
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

.instruction-text, .standby-message {
  font-size: 1.2rem;
  color: #a0a0a0;
  max-width: 80%;
  line-height: 1.6;
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

.standby-animation {
  width: 100px;
  height: 100px;
  border: 5px solid rgba(68, 255, 68, 0.3);
  border-top: 5px solid #44ff44;
  border-radius: 50%;
  animation: spin 1.5s linear infinite;
  margin-top: 2rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* 评估内容 */
.evaluation-content {
  padding: 1rem;
}

.message {
  margin-bottom: 1.5rem;
  padding: 1rem;
  border-radius: 5px;
}

.system-message {
  background: rgba(255, 255, 255, 0.1);
  border-left: 3px solid #8a8a9a;
}

.ai-message {
  background: rgba(68, 255, 68, 0.1);
  border-left: 3px solid #44ff44;
}

.message-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.8rem;
}

.ai-badge {
  background: #44ff44;
  color: #000;
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
  font-size: 0.8rem;
  font-weight: bold;
}

.typewriter-text {
  font-family: 'Courier New', monospace;
  white-space: pre-wrap;
  margin: 0;
  line-height: 1.6;
}

.cursor {
  display: inline-block;
  width: 0.5rem;
  height: 1.2rem;
  background-color: #44ff44;
  vertical-align: middle;
}

.cursor.blink {
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

/* 报告视图 */
.report-container, .comparison-container {
  padding: 1rem;
}

.report-header, .comparison-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.report-header h2, .comparison-header h2 {
  color: #44ff44;
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.5);
}

.crt-button {
  background: #2a2a3a;
  border: 1px solid #3a3a4a;
  color: #e0e0e0;
  padding: 0.5rem 1rem;
  border-radius: 3px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.crt-button:hover {
  background: #3a3a4a;
  border-color: #44ff44;
  color: #44ff44;
  text-shadow: 0 0 5px rgba(68, 255, 68, 0.5);
}

.report-section, .comparison-section {
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 10px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.report-section h3, .comparison-section h3 {
  color: #44ff44;
  margin-bottom: 1rem;
  font-size: 1.2rem;
}

/* 评分显示 */
.score-display {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.score-value {
  font-size: 2rem;
  font-weight: bold;
  color: #44ff44;
  min-width: 60px;
  text-align: center;
}

.score-bar-container {
  flex: 1;
  height: 20px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  overflow: hidden;
}

.score-bar {
  height: 100%;
  background: linear-gradient(to right, #44ff44, #88ff88);
  border-radius: 10px;
  transition: width 1s ease;
}

/* 维度列表 */
.dimensions-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.dimension-item {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.dimension-name {
  width: 100px;
  font-size: 1rem;
}

.dimension-score {
  min-width: 40px;
  text-align: center;
  font-weight: bold;
}

.dimension-bar-container {
  flex: 1;
  height: 15px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 7px;
  overflow: hidden;
}

.dimension-bar {
  height: 100%;
  background: linear-gradient(to right, #44ff44, #88ff88);
  border-radius: 7px;
  transition: width 1s ease;
}

/* 摘要内容 */
.summary-content {
  line-height: 1.6;
}

/* 建议列表 */
.suggestions-list {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.suggestion-item {
  display: flex;
  gap: 1rem;
}

.suggestion-number {
  width: 25px;
  height: 25px;
  background: rgba(68, 255, 68, 0.2);
  border: 1px solid rgba(68, 255, 68, 0.5);
  color: #44ff44;
  border-radius: 50%;
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 0.8rem;
}

.suggestion-content {
  flex: 1;
  line-height: 1.6;
}

/* 对比图表 */
.comparison-chart {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.comparison-item {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.comparison-label {
  width: 120px;
  font-size: 1rem;
}

.comparison-score {
  min-width: 40px;
  text-align: center;
  font-weight: bold;
}

.comparison-bar-container {
  flex: 1;
  height: 20px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  overflow: hidden;
}

.comparison-bar {
  height: 100%;
  border-radius: 10px;
  transition: width 1s ease;
}

/* 对比表格 */
.comparison-table {
  width: 100%;
  border-collapse: collapse;
}

.comparison-table th, .comparison-table td {
  padding: 0.8rem;
  text-align: left;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.comparison-table th {
  color: #44ff44;
  font-weight: normal;
}

.table-score-container {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.table-score {
  min-width: 30px;
  font-weight: bold;
}

.table-bar-container {
  width: 100px;
  height: 10px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 5px;
  overflow: hidden;
}

.table-bar {
  height: 100%;
  border-radius: 5px;
  transition: width 1s ease;
}

/* 差异分析 */
.difference-analysis {
  line-height: 1.6;
}
</style> 