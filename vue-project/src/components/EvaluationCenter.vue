<template>
    <div class="evaluation-center">
      <!-- 未上传文件时的空状态 -->
      <div v-if="!selectedFile && !isEvaluating && !evaluationStats" class="empty-state">
        <div class="tv-logo">EVALUATION CENTER</div>
        <div class="channel-info">频道 3</div>
        <div class="instruction-text">请上传评估数据文件开始分析</div>
        
        <!-- 控制区域整合到屏幕中 -->
        <div class="screen-controls">
          <div class="file-control">
            <input 
              type="file" 
              id="evaluation-file" 
              @change="handleFileUpload" 
              accept=".csv,.xls,.xlsx,.json"
              class="file-input"
            />
            <label for="evaluation-file" class="tv-button">
              <span class="button-text">[ 选择评估文件 ]</span>
            </label>
            <div class="file-name">{{ selectedFile ? selectedFile.name : '未选择文件' }}</div>
          </div>
          
          <!-- 字段选择 - 仅在有可用字段时显示 -->
          <div class="field-selector" v-if="availableFields.length > 0">
            <div class="field-selector-header">
              <div class="field-label">可用字段 <span class="field-count">{{ selectedFields.length }}/{{ availableFields.length }}</span></div>
              <button @click="selectAllFields" class="tv-button small">全选</button>
            </div>
            
            <div class="field-list">
              <div v-for="field in availableFields" :key="field" class="field-item">
                <label class="field-label">
                  <input type="checkbox" v-model="selectedFields" :value="field">
                  <span class="field-name">{{ field }}</span>
                </label>
              </div>
            </div>
            
            <button 
              @click="confirmFields" 
              class="tv-button confirm"
              :disabled="selectedFields.length === 0"
            >
              确认字段
              <span v-if="fieldsConfirmed" class="confirm-indicator">✓</span>
            </button>
          </div>
          
          <!-- 评估类型选择 -->
          <div class="eval-type-selector" v-if="selectedFile">
            <div class="selector-label">评估类型</div>
            <div class="type-buttons">
              <button 
                @click="selectedEvalType = 'dialogue'" 
                class="tv-button" 
                :class="{ 'active': selectedEvalType === 'dialogue' }"
              >
                对话评估
              </button>
              <button 
                @click="selectedEvalType = 'memory'" 
                class="tv-button" 
                :class="{ 'active': selectedEvalType === 'memory' }"
              >
                记忆评估
              </button>
            </div>
          </div>
          
          <!-- 评估代号设置 -->
          <div class="eval-code-group" v-if="selectedFile">
            <div class="selector-label">评估代号</div>
            <div class="eval-code-input">
              <input 
                type="text" 
                v-model="evaluationCode" 
                placeholder="评估代号"
                class="retro-input"
              >
              <button @click="generateRandomCode" class="tv-button small">
                重新生成
              </button>
            </div>
          </div>
          
          <!-- 人设信息输入 -->
          <div class="role-info-group" v-if="selectedFile">
            <div class="selector-label">人设信息</div>
            <textarea 
              v-model="roleInfo" 
              placeholder="输入角色人设信息（可选）"
              class="retro-textarea"
              rows="4"
            ></textarea>
          </div>
          
          <!-- 开始评估按钮 -->
          <div class="eval-actions" v-if="selectedFile">
            <button 
              @click="startEvaluation" 
              class="tv-button primary" 
              :disabled="!fieldsConfirmed || isEvaluating"
            >
              {{ isEvaluating ? '评估中...' : '开始评估' }}
              <div class="button-indicator" :class="{ 'active': isEvaluating }"></div>
            </button>
          </div>
        </div>
      </div>
      
      <!-- 评估过程屏幕 -->
      <div v-else-if="isEvaluating || (systemMessage && !evaluationStats)" class="eval-process">
        <!-- 进度条 - 仅在评估过程中显示 -->
        <div class="progress-container" v-if="isEvaluating">
          <div class="progress-bar">
            <div class="progress-fill" :style="progressStyle"></div>
          </div>
          <div class="progress-text">{{ processed }}/{{ total }}</div>
        </div>
        
        <div class="standby-screen" v-if="systemMessage && !evaluationText">
          <div class="tv-logo">AI EVALUATOR</div>
          <div class="standby-message">{{ systemMessage }}</div>
          <div class="standby-animation"></div>
        </div>
        
        <!-- 有评估数据时显示内容 -->
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
              <pre class="typewriter-text">{{ evaluationText }}<span class="cursor" :class="{ 'blink': !isScanning }">|</span></pre>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 评估报告屏幕 -->
      <div v-else-if="evaluationStats" class="eval-report">
        <div class="report-header">
          <h2 class="report-title">评估报告</h2>
          <div class="report-tabs">
            <button 
              @click="activeReportTab = 'details'" 
              class="tab-button" 
              :class="{ 'active': activeReportTab === 'details' }"
            >
              详细报告
            </button>
            <button 
              @click="activeReportTab = 'compare'" 
              class="tab-button" 
              :class="{ 'active': activeReportTab === 'compare' }"
            >
              报告对比
            </button>
          </div>
          <div class="report-actions">
            <button @click="exportReportCSV" class="tv-button">
              [ 导出报告(CSV) ]
            </button>
            <button @click="resetEvaluation" class="tv-button">
              [ 重新评估 ]
            </button>
          </div>
        </div>
        
        <!-- 详细报告标签页 -->
        <div v-if="activeReportTab === 'details'" class="report-tab-content">
          <!-- 总体评分 -->
          <div class="score-overview">
            <div class="score-card">
              <div class="score-value">{{ evaluationStats.overall_scores.final_score }}</div>
              <div class="score-label">总体评分</div>
            </div>
            <div class="score-card">
              <div class="score-value">{{ evaluationStats.overall_scores.role_score }}</div>
              <div class="score-label">角色评分</div>
            </div>
            <div class="score-card">
              <div class="score-value">{{ evaluationStats.overall_scores.dialogue_score }}</div>
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
              <div class="retro-keyword-cloud" :key="`role-cloud-${activeRoleKeywordTab}`">
                <div class="scanlines"></div>
                <div class="glow-container">
                  <div 
                    v-for="(keyword, index) in getFormattedKeywords('role_play', activeRoleKeywordTab)" 
                    :key="`role-keyword-${keyword.text}-${index}`"
                    class="retro-keyword-tag"
                    :style="keyword.style"
                  >
                    {{ keyword.text }}
                    <span class="keyword-count">{{ keyword.count }}</span>
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
              <div class="retro-keyword-cloud" :key="`dialogue-cloud-${activeDialogueKeywordTab}`">
                <div class="scanlines"></div>
                <div class="glow-container">
                  <div 
                    v-for="(keyword, index) in getFormattedKeywords('dialogue_experience', activeDialogueKeywordTab)" 
                    :key="`dialogue-keyword-${keyword.text}-${index}`"
                    class="retro-keyword-tag"
                    :style="keyword.style"
                  >
                    {{ keyword.text }}
                    <span class="keyword-count">{{ keyword.count }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 报告对比标签页 -->
        <div v-if="activeReportTab === 'compare'" class="report-tab-content">
          <div class="comparison-header">
            <h3>报告对比</h3>
            <!-- 添加文件上传按钮 -->
            <label class="tv-button upload-button">
              [ 上传报告文件 ]
              <input 
                type="file" 
                @change="handleReportFileUpload" 
                accept=".json"
                multiple
                class="hidden-file-input"
              />
            </label>
          </div>
  
          <!-- 无保存报告时显示引导信息 -->
          <div v-if="savedReports.length === 0" class="no-reports">
            <div class="info-icon">i</div>
            <div class="no-reports-text">
              请上传已保存的评估报告JSON文件进行对比分析。<br>
              您可以上传多个报告文件同时进行对比。
            </div>
          </div>
          
          <div v-else class="reports-container">
            <!-- 报告列表 -->
            <div class="reports-list">
              <div class="reports-list-header">
                <div class="header-text">已加载报告 ({{ savedReports.length }})</div>
                <div class="clear-button" @click="clearReports">清空</div>
              </div>
              <div v-for="report in savedReports" :key="report.id" class="report-item">
                <div class="report-checkbox">
                  <input type="checkbox" v-model="selectedReports" :value="report.id" :id="`report-${report.id}`">
                  <label :for="`report-${report.id}`"></label>
                </div>
                <div class="report-info">
                  <div class="report-code" :style="{ color: getReportColor(report.id) }">
                    {{ report.evaluation_code }}
                  </div>
                  <div class="report-meta">
                    得分: {{ report.overall_scores.final_score }} | 日期: {{ formatDate(report.evaluation_date) }}
                  </div>
                </div>
              </div>
            </div>
            
            <!-- 对比视图 -->
            <div v-if="selectedReports.length >= 2" class="comparison-view">
              <!-- 对比标签页 -->
              <div class="comparison-tabs">
                <button 
                  v-for="(item, key) in rolePlayItems" 
                  :key="`comp-${key}`"
                  @click="activeComparisonTab = key"
                  class="comparison-tab"
                  :class="{ 'active': activeComparisonTab === key }"
                >
                  {{ item.label }}
                </button>
              </div>
              
              <!-- 对比图表 -->
              <div class="comparison-section">
                <h4>角色扮演对比 - {{ activeComparisonTab ? rolePlayItems[activeComparisonTab].label : '' }}</h4>
                <div class="comparison-bars" v-if="activeComparisonTab">
                  <div 
                    v-for="reportId in selectedReports" 
                    :key="`bar-${reportId}-${activeComparisonTab}`"
                    class="comparison-bar-item"
                  >
                    <div class="comparison-label">{{ getReportById(reportId).evaluation_code }}</div>
                    <div class="comparison-bar-container">
                      <div 
                        class="comparison-bar" 
                        :style="{ 
                          width: `${getDimensionScore(reportId, 'role_play', activeComparisonTab)}%`,
                          backgroundColor: getReportColor(reportId)
                        }"
                      ></div>
                    </div>
                    <div class="comparison-value">{{ getDimensionScore(reportId, 'role_play', activeComparisonTab) }}</div>
                  </div>
                </div>
              </div>
            </div>
            <div v-if="savedReports.length > 0" class="report-actions bottom">
              <button @click="exportComparisonCSV" class="tv-button export-btn">
                [ 导出对比报告(CSV) ]
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script setup>
  import { ref, computed, watch } from 'vue'
  
  const API_BASE_URL = 'http://localhost:8000' // 修改为你的后端地址
  
  // 文件和字段相关状态
  const selectedFile = ref(null)
  const availableFields = ref([])
  const selectedFields = ref([])
  const fieldsConfirmed = ref(false)
  
  // 评估类型和状态
  const selectedEvalType = ref('dialogue')
  const isEvaluating = ref(false)
  const isScanning = ref(false)
  const evaluationCode = ref('')
  const roleInfo = ref('')
  
  // 进度相关
  const processed = ref(0)
  const total = ref(0)
  
  // 消息和结果相关
  const systemMessage = ref('我是评估助手，请上传文件开始评估。')
  const evaluationText = ref('')
  const evaluationStats = ref(null)
  
  // 报告标签页状态
  const activeReportTab = ref('details')
  const activeRoleKeywordTab = ref('consistency')
  const activeDialogueKeywordTab = ref('response_quality')
  const activeComparisonTab = ref('consistency')
  
  // 报告对比相关
  const savedReports = ref([])
  const selectedReports = ref([])
  
  // 词组库
  const wordLists = {
    games: ['魂斗罗', '双截龙', '坦克大战', '忍者龙剑传', '洛克人', '恶魔城', '冒险岛', '赤色要塞', 
            '超级马里奥', '塞尔达传说', '银河战士', '最终幻想', '勇者斗恶龙', '街头霸王', '快打旋风', 
            '魔界村', '绿色兵团', '沙罗曼蛇', '赤影战士', '忍者神龟', '超级魂斗罗', '热血物语', '热血格斗', 
            '热血篮球', '热血足球', '热血新纪录', '吞食天地', '重装机兵', '梦幻模拟战', '火焰之纹章', 
            '大航海时代', '三国志', '信长之野望', '炸弹人', '泡泡龙', '俄罗斯方块', '打砖块', '小蜜蜂', 
            '大金刚', '吃豆人', '功夫', '影子传说', '淘金者', '越野机车', '马戏团', '南极大冒险', 
            '高桥名人的冒险岛', '圣斗士星矢', '北斗神拳', '七龙珠', '幽游白书'],
    suffixes: ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 
               'EX', 'DX', 'PLUS', 'ULTRA', 'SPECIAL', 'TURBO', 'CHAMPION', 'MASTER', 'LEGEND', 'FINAL']
  }
  
  // 预定义评估维度选项
  const rolePlayItems = {
    consistency: { label: '角色一致性', color: '#44ff44' },
    knowledge: { label: '知识准确性', color: '#44ffff' },
    reaction: { label: '反应合理性', color: '#ff44ff' },
    creativity: { label: '角色创造力', color: '#ffff44' }
  }
  
  const dialogueItems = {
    response_quality: { label: '回复质量', color: '#44ff44' },
    helpfulness: { label: '帮助程度', color: '#44ffff' },
    complexity: { label: '回复复杂度', color: '#ff44ff' },
    harmfulness: { label: '无害程度', color: '#ffff44' }
  }
  
  // 进度条样式
  const progressStyle = computed(() => ({
    width: `${(processed.value / total.value) * 100}%`
  }))
  
  // 生成随机评估代号
  const generateRandomCode = () => {
    const randomGame = wordLists.games[Math.floor(Math.random() * wordLists.games.length)]
    const randomSuffix = wordLists.suffixes[Math.floor(Math.random() * wordLists.suffixes.length)]
    evaluationCode.value = `${randomGame}${randomSuffix}`
  }
  
  // 处理文件上传
  const handleFileUpload = async (event) => {
    const file = event.target.files[0]
    if (!file) return
    
    selectedFile.value = file
    fieldsConfirmed.value = false
    selectedFields.value = []
    
    // 自动生成评估代号
    generateRandomCode()
    
    try {
      const formData = new FormData()
      formData.append('file', file)
      
      const response = await fetch(`${API_BASE_URL}/api/v1/file/columns`, {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) throw new Error('获取列名失败')
      
      const data = await response.json()
      availableFields.value = data.columns
    } catch (error) {
      console.error('Error:', error)
      systemMessage.value = '文件处理失败'
    }
  }
  
  // 全选字段
  const selectAllFields = () => {
    selectedFields.value = [...availableFields.value]
  }
  
  // 确认字段
  const confirmFields = () => {
    if (selectedFields.value.length === 0) return
    fieldsConfirmed.value = true
    
    // 添加确认提示
    const originalMessage = systemMessage.value
    systemMessage.value = '字段选择已确认 ✓'
    setTimeout(() => {
      systemMessage.value = originalMessage
    }, 2000)
  }
  
  // 开始评估
  const startEvaluation = async () => {
    if (!selectedFile.value || !fieldsConfirmed.value) return
    
    try {
      isEvaluating.value = true
      evaluationStats.value = null // 清空之前的报告数据
      systemMessage.value = '正在准备评估...'
      
      const formData = new FormData()
      formData.append('file', selectedFile.value)
      formData.append('eval_type', selectedEvalType.value)
      formData.append('user_id', 'user123') // 可以使用实际用户ID
      formData.append('selected_fields', JSON.stringify(selectedFields.value))
      
      // 添加评估代号
      formData.append('evaluation_code', evaluationCode.value || `评估${new Date().toISOString().slice(0,10)}`)
      
      // 添加人设信息
      if (roleInfo.value && roleInfo.value.trim()) {
        formData.append('role_info', roleInfo.value.trim())
      }
      
      systemMessage.value = '正在连接评估服务...'
      
      const response = await fetch(`${API_BASE_URL}/api/v1/evaluate`, {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) throw new Error('评估请求失败')
      
      // 开始接收数据时启动扫描
      isScanning.value = true
      systemMessage.value = '评估进行中...'
  
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
  
      let currentEvaluation = {
        index: null,
        content: '',
        originalData: ''
      }
      
      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          // 数据接收完成，停止扫描
          isScanning.value = false
          break
        }
  
        buffer += decoder.decode(value)
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        
        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data: ')) continue
          
          try {
            const data = JSON.parse(line.slice(5))
            
            if (data.total) {
              total.value = data.total
              continue
            }
  
            switch (data.type) {
              case 'start':
                if (currentEvaluation.index !== null && evaluationText.value) {
                  evaluationText.value += '\n---\n'
                }
                currentEvaluation = {
                  index: data.index,
                  content: '',
                  originalData: data.original_data
                }
                evaluationText.value += `评估项 ${data.index}:\n原始数据:\n${data.original_data}\n\n评估结果:\n`
                break
                
              case 'chunk':
                currentEvaluation.content += data.content
                evaluationText.value += data.content
                // 打字声音效果
                playTypeSound()
                break
                
              case 'end':
                processed.value = data.processed
                // 评估完成
                if (data.processed === data.total) {
                  systemMessage.value = '评估完成!'
                  
                  // 获取统计报告
                  if (data.stats) {
                    evaluationStats.value = data.stats
                    saveReportToLocalStorage(data.stats)
                    // 自动添加到对比列表
                    savedReports.value.push(data.stats)
                  }
                }
                break
            }
          } catch (e) {
            console.error('Error parsing data:', e, line)
          }
        }
      }
      
      isEvaluating.value = false
      
    } catch (error) {
      console.error('Error:', error)
      systemMessage.value = '评估过程中出现错误'
      isEvaluating.value = false
      isScanning.value = false
    }
  }
  
  // 重置评估
  const resetEvaluation = () => {
    evaluationStats.value = null
    evaluationText.value = ''
    processed.value = 0
    total.value = 0
    systemMessage.value = '我是评估助手，请上传文件开始评估。'
  }
  
  // 处理报告上传
  const handleReportFileUpload = async (event) => {
    const files = event.target.files
    if (!files.length) return
    
    for (const file of files) {
      try {
        const reader = new FileReader()
        reader.onload = (e) => {
          try {
            const reportData = JSON.parse(e.target.result)
            // 确保没有重复的报告
            if (!savedReports.value.some(r => r.id === reportData.id)) {
              savedReports.value.push(reportData)
            }
          } catch (error) {
            console.error('解析报告文件失败:', error)
          }
        }
        reader.readAsText(file)
      } catch (error) {
        console.error('读取文件失败:', error)
      }
    }
  }
  
  // 保存报告到本地存储
  const saveReportToLocalStorage = (report) => {
    try {
      // 给报告添加唯一ID
      report.id = Date.now().toString()
      
      // 尝试从localStorage中获取已保存的报告
      const savedReportsStr = localStorage.getItem('evaluation_reports')
      let savedReportsArr = []
      
      if (savedReportsStr) {
        savedReportsArr = JSON.parse(savedReportsStr)
      }
      
      // 添加新报告
      savedReportsArr.push(report)
      
      // 保存回localStorage
      localStorage.setItem('evaluation_reports', JSON.stringify(savedReportsArr))
    } catch (error) {
      console.error('保存报告到本地存储失败:', error)
    }
  }
  
  // 清空报告
  const clearReports = () => {
    savedReports.value = []
    selectedReports.value = []
  }
  
  // 获取报告分数
  const getScoreValue = (dimension, category) => {
    if (!evaluationStats.value || !evaluationStats.value[category]) return 0
    return evaluationStats.value[category][dimension] || 0
  }
  
  // 获取维度分数(对比报告)
  const getDimensionScore = (reportId, category, dimension) => {
    const report = getReportById(reportId)
    if (!report || !report[category]) return 0
    return report[category][dimension] || 0
  }
  
  // 获取报告颜色
  const getReportColor = (reportId) => {
    // 为每个报告分配一个固定颜色
    const colors = ['#44ff44', '#44ffff', '#ff44ff', '#ffff44', '#ff4444', '#4444ff']
    const index = selectedReports.value.indexOf(reportId)
    return colors[index % colors.length]
  }
  
  // 根据ID获取报告
  const getReportById = (id) => {
    return savedReports.value.find(report => report.id === id) || {}
  }
  
  // 格式化关键词数据
  const getFormattedKeywords = (category, dimension) => {
    if (!evaluationStats.value || 
        !evaluationStats.value.keywords || 
        !evaluationStats.value.keywords[category] || 
        !evaluationStats.value.keywords[category][dimension]) {
      return []
    }
    
    const keywords = evaluationStats.value.keywords[category][dimension]
    const maxCount = Math.max(...keywords.map(k => k.count))
    
    return keywords.map(keyword => {
      const size = 0.7 + (keyword.count / maxCount) * 0.8
      const randomX = Math.floor(Math.random() * 70)
      const randomY = Math.floor(Math.random() * 70)
      
      return {
        text: keyword.text,
        count: keyword.count,
        style: {
          fontSize: `${size}em`,
          color: rolePlayItems[dimension]?.color || '#44ff44',
          position: 'absolute',
          left: `${randomX}%`,
          top: `${randomY}%`,
          transform: 'translate(-50%, -50%)',
          opacity: 0.5 + (keyword.count / maxCount) * 0.5
        }
      }
    })
  }
  
  // 格式化日期
  const formatDate = (dateStr) => {
    try {
      const date = new Date(dateStr)
      return date.toLocaleDateString()
    } catch (e) {
      return dateStr
    }
  }
  
  // 导出CSV报告
  const exportReportCSV = () => {
    if (!evaluationStats.value) return
    
    const headers = ['维度', '得分']
    let csvContent = `${headers.join(',')}\n`
    
    // 添加角色扮演评分
    csvContent += `总体评分,${evaluationStats.value.overall_scores.final_score}\n`
    csvContent += `角色评分,${evaluationStats.value.overall_scores.role_score}\n`
    csvContent += `对话评分,${evaluationStats.value.overall_scores.dialogue_score}\n\n`
    
    csvContent += `角色扮演维度,得分\n`
    for (const [key, item] of Object.entries(rolePlayItems)) {
      csvContent += `${item.label},${getScoreValue(key, 'role_play')}\n`
    }
    
    csvContent += `\n对话体验维度,得分\n`
    for (const [key, item] of Object.entries(dialogueItems)) {
      csvContent += `${item.label},${getScoreValue(key, 'dialogue_experience')}\n`
    }
    
    // 创建并下载CSV文件
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.setAttribute('href', url)
    link.setAttribute('download', `评估报告_${evaluationCode.value || '未命名'}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }
  
  // 导出对比CSV报告
  const exportComparisonCSV = () => {
    if (selectedReports.value.length < 2) return
    
    let headers = ['维度']
    selectedReports.value.forEach(reportId => {
      headers.push(getReportById(reportId).evaluation_code)
    })
    
    let csvContent = `${headers.join(',')}\n`
    
    // 添加角色扮演评分对比
    for (const [key, item] of Object.entries(rolePlayItems)) {
      let row = [item.label]
      selectedReports.value.forEach(reportId => {
        row.push(getDimensionScore(reportId, 'role_play', key))
      })
      csvContent += `${row.join(',')}\n`
    }
    
    csvContent += `\n对话体验维度\n`
    for (const [key, item] of Object.entries(dialogueItems)) {
      let row = [item.label]
      selectedReports.value.forEach(reportId => {
        row.push(getDimensionScore(reportId, 'dialogue_experience', key))
      })
      csvContent += `${row.join(',')}\n`
    }
    
    // 创建并下载CSV文件
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.setAttribute('href', url)
    link.setAttribute('download', `评估对比报告_${new Date().toISOString().slice(0,10)}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }
  
  // 添加打字声音效果
  const playTypeSound = () => {
    const audio = new Audio();
    audio.src = 'data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA/+M4wAAAAAAAAAAAAEluZm8AAAAPAAAAAwAAABQADw8PDw8PDw8PDw8PDw8PDw8PDw8PDw8VFRUVFRUVFRUVFRUVFRUVFRUVFRUVFR4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCT/wAARCAAIAAgDASIAAhEBAxEB/8QAWQABAQEAAAAAAAAAAAAAAAAAAAIEAQEBAQEAAAAAAAAAAAAAAAAAAgEF/8QAFwEBAQEBAAAAAAAAAAAAAAAAAAECA//EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AKpRqNQBX//Z';
    audio.volume = 0.05;
    audio.play().catch(() => {});
  };
  </script>
  
  <style scoped>
  /* 评估中心整体样式 */
  .evaluation-center {
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
    color: #44ff44;
    font-family: 'VT323', 'Courier New', monospace;
    background-color: #000;
    overflow: auto;
    padding: 1rem;
  }
  
  /* 空状态 */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    text-align: center;
  }
  
  .tv-logo {
  font-size: 3rem;
  font-weight: bold;
  letter-spacing: 5px;
  margin-bottom: 1rem;
  color: #44ff44;
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.7);
}

.instruction-text {
  font-size: 1.4rem;
  margin-bottom: 2rem;
  opacity: 0.8;
}

/* 屏幕内控制区域 */
.screen-controls {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  background: rgba(10, 20, 20, 0.5);
  border-radius: 10px;
  padding: 1.5rem;
  border: 1px solid rgba(68, 255, 68, 0.2);
  min-width: 500px;
}

.file-control {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.file-input {
  display: none;
}

.file-name {
  font-size: 0.9rem;
  color: #aaaaaa;
}

.tv-button {
  background: rgba(20, 40, 20, 0.8);
  border: 1px solid rgba(68, 255, 68, 0.5);
  color: #44ff44;
  padding: 0.7rem 1.2rem;
  cursor: pointer;
  transition: all 0.3s ease;
  font-family: 'VT323', 'Courier New', monospace;
  border-radius: 5px;
}

.tv-button:hover:not(:disabled) {
  background: rgba(30, 60, 30, 0.8);
  box-shadow: 0 0 10px rgba(68, 255, 68, 0.5);
}

.tv-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.tv-button.small {
  padding: 0.3rem 0.7rem;
  font-size: 0.8rem;
}

.tv-button.active {
  background: rgba(68, 255, 68, 0.3);
  border-color: #44ff44;
}

.tv-button.confirm {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
}

.confirm-indicator {
  color: #44ff44;
  font-weight: bold;
}

/* 字段选择器 */
.field-selector {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.field-selector-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.field-label {
  font-size: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.field-count {
  font-size: 0.8rem;
  color: #aaaaaa;
}

.field-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid rgba(68, 255, 68, 0.2);
  border-radius: 5px;
  padding: 0.5rem;
  background: rgba(0, 0, 0, 0.3);
}

.field-item {
  display: flex;
  align-items: center;
}

/* 评估类型选择器 */
.eval-type-selector {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.selector-label {
  font-size: 1rem;
}

.type-buttons {
  display: flex;
  gap: 1rem;
}

/* 评估代号和人设信息 */
.eval-code-group, .role-info-group {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.eval-code-input {
  display: flex;
  gap: 0.5rem;
}

.retro-input {
  background: rgba(0, 0, 0, 0.7);
  border: 1px solid rgba(68, 255, 68, 0.3);
  color: #44ff44;
  padding: 0.7rem;
  flex: 1;
  font-family: 'VT323', 'Courier New', monospace;
  border-radius: 5px;
}

.retro-textarea {
  background: rgba(0, 0, 0, 0.7);
  border: 1px solid rgba(68, 255, 68, 0.3);
  color: #44ff44;
  padding: 0.7rem;
  font-family: 'VT323', 'Courier New', monospace;
  border-radius: 5px;
  resize: vertical;
}

/* 操作按钮 */
.action-buttons {
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
}

/* 处理中状态 */
.processing-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
}

.processing-message {
  font-size: 1.5rem;
  margin: 1rem 0 2rem;
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

.progress-bar {
  width: 80%;
  height: 15px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 7px;
  overflow: hidden;
  margin-top: 1rem;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(to right, #44ff44, #88ff88);
  border-radius: 7px;
  transition: width 0.3s ease;
}

.progress-text {
  margin-top: 0.5rem;
  font-size: 1rem;
}

/* 评估结果状态 */
.result-state {
  height: 100%;
  overflow-y: auto;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.result-header h2 {
  color: #44ff44;
  text-shadow: 0 0 5px rgba(68, 255, 68, 0.5);
  margin: 0;
}

.header-actions {
  display: flex;
  gap: 0.8rem;
}

/* 标签导航 */
.result-tabs {
  display: flex;
  gap: 1px;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid rgba(68, 255, 68, 0.3);
}

.result-tab {
  padding: 0.7rem 1.2rem;
  cursor: pointer;
  background: rgba(10, 20, 10, 0.5);
  color: #aaaaaa;
  font-size: 1rem;
}

.result-tab.active {
  background: rgba(20, 40, 20, 0.8);
  color: #44ff44;
  border-top: 2px solid #44ff44;
}

/* 评估内容区域 */
.eval-content {
  padding: 1rem;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(68, 255, 68, 0.2);
  border-radius: 5px;
}

/* 总结栏 */
.summary-card {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding: 1.5rem;
  background: rgba(10, 20, 10, 0.5);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 10px;
  margin-bottom: 1.5rem;
}

.summary-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.summary-header h3 {
  margin: 0;
  color: #44ff44;
  font-size: 1.3rem;
}

.score-container {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.total-score {
  font-size: 1.5rem;
  font-weight: bold;
  color: #44ff44;
}

.score-label {
  font-size: 0.9rem;
  color: #aaaaaa;
}

/* 评分维度 */
.dimensions-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.dimension-category {
  margin-bottom: 1.5rem;
}

.category-title {
  font-size: 1.2rem;
  color: #44ff44;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid rgba(68, 255, 68, 0.3);
}

.dimension-list {
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

/* 频道信息 */
.channel-info {
  position: absolute;
  top: 10px;
  right: 15px;
  background: rgba(0, 0, 0, 0.7);
  color: #44ff44;
  padding: 0.3rem 0.6rem;
  border-radius: 3px;
  font-size: 0.8rem;
  border: 1px solid rgba(68, 255, 68, 0.3);
}
</style>