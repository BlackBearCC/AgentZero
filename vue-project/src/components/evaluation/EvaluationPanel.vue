<template>
  <div class="tv-container">
    <!-- 控制面板 -->
    <div class="control-panel">
      <div class="panel-title">控制中心</div>
      
      <!-- 电源控制 -->
      <div class="control-group">
        <div class="control-label">POWER</div>
        <button @click="togglePower" class="control-button">
          <div class="button-face">
            <span>{{ isPoweredOn ? 'ON' : 'OFF' }}</span>
            <div class="power-indicator" :class="{ 'active': isPoweredOn }"></div>
          </div>
        </button>
      </div>
      
      <!-- 频道控制 -->
      <div class="control-group">
        <div class="control-label">CHANNEL</div>
        <div class="channel-buttons">
          <button @click="changeChannel(1)" class="control-button channel-btn" :class="{ 'active': activeChannel === 1 }">1</button>
          <button @click="changeChannel(2)" class="control-button channel-btn" :class="{ 'active': activeChannel === 2 }">2</button>
          <button @click="changeChannel(3)" class="control-button channel-btn" :class="{ 'active': activeChannel === 3 }">3</button>
        </div>
      </div>
      
      <!-- 文件上传 -->
      <div class="control-group">
        <div class="control-label">INPUT</div>
        <label class="control-button file-input-button">
          <div class="button-face">
            <span>上传文件</span>
            <i class="upload-icon">↑</i>
          </div>
          <input type="file" @change="handleFileUpload" accept=".csv,.xls,.xlsx,.json" class="hidden-file-input" />
        </label>
        <div class="file-info" v-if="selectedFile">
          <div class="file-name">{{ selectedFile.name }}</div>
          <div class="file-size">{{ formatFileSize(selectedFile.size) }}</div>
        </div>
      </div>
      
      <!-- 字段选择 - 仅在有可用字段时显示 -->
      <div class="control-group field-selector" v-if="availableFields.length > 0">
        <div class="control-label">FIELDS <span class="field-count">{{ selectedFields.length }}/{{ availableFields.length }}</span></div>
        
        <!-- 字段列表 - 垂直排列 -->
        <div class="field-list">
          <div v-for="field in availableFields" :key="field" class="field-item">
            <label class="field-label">
              <input type="checkbox" v-model="selectedFields" :value="field">
              <span class="field-name">{{ field }}</span>
            </label>
          </div>
        </div>
        
        <!-- 操作按钮 -->
        <div class="field-actions">
          <button 
            @click="confirmFields" 
            class="control-button confirm-fields-btn"
            :disabled="selectedFields.length === 0"
          >
            <div class="button-face">
              <span>确认字段</span>
              <div v-if="fieldsConfirmed" class="confirm-indicator">✓</div>
            </div>
          </button>
        </div>
      </div>
      
      <!-- 评估类型选择 -->
      <div class="control-group">
        <div class="control-label">MODE</div>
        <div class="mode-selector">
          <button 
            @click="selectedEvalType = 'dialogue'" 
            class="control-button mode-btn" 
            :class="{ 'active': selectedEvalType === 'dialogue' }"
          >
            对话评估
          </button>
          <button 
            @click="selectedEvalType = 'memory'" 
            class="control-button mode-btn" 
            :class="{ 'active': selectedEvalType === 'memory' }"
          >
            记忆评估
          </button>
        </div>
      </div>
      
      <!-- 开始评估按钮之前添加新的控制组：评估代号和人设信息 -->
      <div v-if="selectedFile" class="control-group">
        <div class="control-label">评估代号</div>
        <div class="eval-code-input">
          <input 
            type="text" 
            v-model="evaluationCode" 
            placeholder="评估代号"
            class="code-input"
          >
          <button @click="generateRandomCode" class="control-button small-btn">
            <div class="button-face">
              <span>重新生成</span>
            </div>
          </button>
        </div>
      </div>
      
      <!-- 人设信息输入 -->
      <div class="control-group">
        <div class="control-label">人设信息</div>
        <textarea 
          v-model="roleInfo" 
          placeholder="输入角色人设信息（可选）"
          class="role-info-input"
          rows="4"
        ></textarea>
      </div>
      
      <!-- 开始评估按钮 -->
      <div class="control-group">
        <div class="control-label">OPERATION</div>
        <button 
          @click="startEvaluation" 
          class="control-button start-btn" 
          :disabled="!selectedFile || !fieldsConfirmed || isEvaluating"
        >
          <div class="button-face">
            <span>{{ isEvaluating ? '评估中...' : '开始评估' }}</span>
            <div class="operation-indicator" :class="{ 'active': isEvaluating }"></div>
          </div>
        </button>
      </div>
      
      <!-- 进度条 - 仅在评估过程中显示 -->
      <div class="control-group" v-if="isEvaluating">
        <div class="control-label">PROGRESS</div>
        <div class="progress-bar">
          <div class="progress-fill" :style="progressStyle"></div>
        </div>
        <div class="progress-text">{{ processed }}/{{ total }}</div>
      </div>
      
      <!-- 系统状态 -->
      <div class="system-status">
        <div class="status-label">SYSTEM STATUS</div>
        <div class="status-value">{{ systemStatus }}</div>
      </div>
    </div>

    <!-- 电视屏幕 -->
    <div class="tv-screen">
      <div class="screen-frame" :class="{ 'scanning': isScanning, 'changing-channel': isChangingChannel }">
        <div class="screen-content">
          <!-- 评估过程显示 - Channel 1 -->
          <div v-if="activeChannel === 1" class="chat-window" ref="chatWindow">
            <!-- 无数据时显示无信号 -->
            <div v-if="!evaluationText && !systemMessage" class="no-signal">
              <div class="static-effect"></div>
              <div class="no-signal-text">NO SIGNAL</div>
            </div>
            
            <!-- 有系统消息但无评估数据时显示待机画面 -->
            <div v-else-if="systemMessage && !evaluationText" class="standby-screen">
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
          
          <!-- 评估报告显示 - Channel 2 -->
          <div v-if="activeChannel === 2" class="chat-window report-view">
            <!-- 无数据时显示无信号 -->
            <div v-if="!evaluationStats" class="no-signal">
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
          </div>
          
          <!-- Channel 3 的报告对比视图 -->
          <div v-if="activeChannel === 3" class="chat-window report-comparison">
            <div class="comparison-header">
              <h2>报告对比</h2>
              <!-- 添加文件上传按钮 -->
              <label class="crt-button upload-button">
                <span class="button-text">[ 上传报告文件 ]</span>
                <div class="button-icon">📁</div>
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
                <h3>暂无保存的报告</h3>
                <p>在报告页面(频道2)点击"保存报告"按钮将报告保存到对比列表中</p>
                <p>或者上传已保存的报告文件</p>
              </div>
            </div>
            
            <!-- 有保存报告时显示报告列表和对比视图 -->
            <div v-else class="reports-container">
              <h2 class="report-title">报告对比</h2>
              
              <!-- 保存的报告列表 -->
              <div class="saved-reports-list">
                <h3>已保存报告 ({{ savedReports.length }})</h3>
                <div class="report-cards">
                  <div 
                    v-for="(report, index) in savedReports" 
                    :key="index"
                    class="report-card"
                    :class="{ 'selected': selectedReports.includes(report.id) }"
                    @click="toggleReportSelection(report.id)"
                  >
                    <div class="report-card-header">
                      <div class="report-code">{{ report.evaluation_code }}</div>
                      <div class="report-date">{{ formatDate(report.timestamp) }}</div>
                    </div>
                    <div class="report-score">{{ report.stats.overall_scores.final_score }}</div>
                    <div class="report-card-footer">
                      <button @click.stop="downloadReport(report)" class="mini-btn">下载</button>
                      <button @click.stop="removeReport(report.id)" class="mini-btn delete">删除</button>
                    </div>
                  </div>
                </div>
              </div>
              
              <!-- 对比视图 - 只在选择了2个及以上报告时显示 -->
              <div v-if="selectedReports.length >= 2" class="comparison-view">
                <h3>评分对比</h3>
                
                <!-- 总体评分对比 -->
                <div class="comparison-section">
                  <h4>总体评分</h4>
                  <div class="comparison-bars">
                    <div 
                      v-for="reportId in selectedReports" 
                      :key="`overall-${reportId}`"
                      class="comparison-bar-row"
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
                      <div class="comparison-value">{{ getReportById(reportId).stats.overall_scores.final_score }}</div>
                    </div>
                  </div>
                </div>
                
                <!-- 角色扮演评分对比 -->
                <div class="comparison-section">
                  <h4>角色扮演评分</h4>
                  <div class="dimension-tabs">
                    <button 
                      v-for="(item, key) in rolePlayItems" 
                      :key="`comp-role-${key}`"
                      @click="activeComparisonTab = key"
                      class="dimension-tab"
                      :class="{ 'active': activeComparisonTab === key }"
                    >
                      {{ item.label }}
                    </button>
                  </div>
                  
                  <div class="comparison-bars" v-if="activeComparisonTab">
                    <div 
                      v-for="reportId in selectedReports" 
                      :key="`role-${reportId}-${activeComparisonTab}`"
                      class="comparison-bar-row"
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
              <div v-if="savedReports.length > 0" class="report-actions">
                <button @click="exportComparisonCSV" class="crt-button export-btn">
                  <span class="button-text">[ 导出对比报告(CSV) ]</span>
                  <div class="button-icon">📊</div>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import axios from 'axios'
import { useRouter } from 'vue-router'

const API_BASE_URL = 'http://localhost:8000' // 修改为你的后端地址

const selectedFile = ref(null)
const selectedEvalType = ref('dialogue')
const isEvaluating = ref(false)
const results = ref([])
const processed = ref(0)
const total = ref(0)
const chatWindow = ref(null)
const systemMessage = ref('我是评估助手，请上传文件开始评估。')
const evaluationText = ref('')
const isScanning = ref(false)
const typingTimeout = ref(null)

const showFieldSelector = ref(false)
const availableFields = ref([])
const selectedFields = ref([])
const fieldsConfirmed = ref(false)

// 新增：控制是否显示报告
const showReport = ref(false)
// 新增：报告数据
const evaluationStats = ref(null)
const activeChannel = ref(1) // 当前频道
const isChangingChannel = ref(false) // 是否正在换台

const router = useRouter()

// 添加关键词标签页状态
const activeRoleKeywordTab = ref('consistency')
const activeDialogueKeywordTab = ref('response_quality')

// 缓存每个维度的关键词位置
const keywordPositions = ref({})

// 自动滚动到底部
const scrollToBottom = () => {
  if (chatWindow.value) {
    setTimeout(() => {
      chatWindow.value.scrollTop = chatWindow.value.scrollHeight
    }, 50)
  }
}

// 监听结果变化，自动滚动
watch(results, () => {
  scrollToBottom()
}, { deep: true })

const progressStyle = computed(() => ({
  width: `${(processed.value / total.value) * 100}%`
}))

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

// 生成随机评估代号
const generateRandomCode = () => {
  const randomGame = wordLists.games[Math.floor(Math.random() * wordLists.games.length)]
  const randomSuffix = wordLists.suffixes[Math.floor(Math.random() * wordLists.suffixes.length)]
  evaluationCode.value = `${randomGame}${randomSuffix}`
}

// 修改handleFileUpload方法，使用新的代号生成方式
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

// 修改确认字段方法，添加视觉反馈
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

const selectAllFields = () => {
  selectedFields.value = [...availableFields.value]
}

// 添加打字声音效果
const playTypeSound = () => {
  const audio = new Audio();
  audio.src = 'data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA/+M4wAAAAAAAAAAAAEluZm8AAAAPAAAAAwAAABQADw8PDw8PDw8PDw8PDw8PDw8PDw8PDw8VFRUVFRUVFRUVFRUVFRUVFRUVFRUVFR4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCT/wAARCAAIAAgDASIAAhEBAxEB/8QAWQABAQEAAAAAAAAAAAAAAAAAAAIEAQEBAQEAAAAAAAAAAAAAAAAAAgEF/8QAFwEBAQEBAAAAAAAAAAAAAAAAAAECA//EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AKpRqNQBX//Z';
  audio.volume = 0.05;
  audio.play().catch(() => {});
};

// 修改开始评估函数，自动添加报告到对比列表
const startEvaluation = async () => {
  if (!selectedFile.value || !fieldsConfirmed.value) return
  
  try {
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
    
    isEvaluating.value = true
    systemMessage.value = '正在评估，请稍候...'
    evaluationText.value = ''
    processed.value = 0
    total.value = 0
    
    const response = await fetch(`${API_BASE_URL}/api/v1/evaluate/stream`, {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) throw new Error('评估请求失败')
    
    // 开始接收数据时启动扫描
    isScanning.value = true

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
              if (!isScanning.value) {
                isScanning.value = true
              }
              currentEvaluation.content += data.content
              evaluationText.value = evaluationText.value.split(`评估项 ${data.index}:`)[0] + 
                                   `评估项 ${data.index}:\n原始数据:\n${currentEvaluation.originalData}\n\n评估结果:\n${currentEvaluation.content}`
              playTypeSound()
              scrollToBottom()
              
              // 重置打字状态的计时器
              clearTimeout(typingTimeout)
              typingTimeout = setTimeout(() => {
                isScanning.value = false
              }, 100)
              break

            case 'end':
              processed.value = data.index
              break

            case 'complete':
              // 处理评估统计数据
              showEvaluationReport(data.stats)
              break

            case 'error':
              systemMessage.value = `评估项 ${data.index} 错误: ${data.error}`
              break
          }
        } catch (e) {
          console.error('解析SSE数据失败:', e)
        }
      }
    }
    
    // 评估完成后自动添加到对比列表
    const report = {
      id: `report-${Date.now()}`,
      evaluation_code: evaluationCode.value,
      timestamp: new Date(),
      stats: JSON.parse(JSON.stringify(evaluationStats.value)),
      role_info: roleInfo.value
    }
    
    savedReports.value.push(report)
    
    // 如果选中报告少于3个，自动选中新生成的报告
    if (selectedReports.value.length < 3) {
      selectedReports.value.push(report.id)
    }
    
  } catch (error) {
    console.error('Error during evaluation:', error)
    systemMessage.value = '评估失败'
  } finally {
    isEvaluating.value = false
    systemMessage.value = '评估完成！'
    isScanning.value = false  // 确保扫描停止
  }
}

// 修改换台函数
const changeChannel = (channel) => {
  if (channel === activeChannel.value) return
  
  // 开始换台效果
  isChangingChannel.value = true
  isScanning.value = true
  
  // 延迟切换频道，模拟换台过程
  setTimeout(() => {
    activeChannel.value = channel
    
    // 如果切换到频道3，导出评估数据
    if (channel === 3 && evaluationStats.value) {
      exportEvaluationReport()
    }
    
    // 结束换台效果
    setTimeout(() => {
      isChangingChannel.value = false
      isScanning.value = false
    }, 500)
  }, 1000)
}

// 导出评估报告函数
const exportEvaluationReport = () => {
  if (evaluationStats.value) {
    const dataStr = JSON.stringify(evaluationStats.value, null, 2)
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr)
    
    const exportFileDefaultName = 'evaluation_report.json'
    
    const linkElement = document.createElement('a')
    linkElement.setAttribute('href', dataUri)
    linkElement.setAttribute('download', exportFileDefaultName)
    linkElement.click()
  } else {
    systemMessage.value = '没有评估数据可以导出'
  }
}

// 添加角色扮演评估项
const rolePlayItems = {
  consistency: { label: '角色一致性' },
  knowledge: { label: '角色知识' },
  language_style: { label: '语言风格' },
  emotional_expression: { label: '情感表达' },
  character_depth: { label: '角色深度' }
}

// 添加对话体验评估项
const dialogueItems = {
  response_quality: { label: '回应质量' },
  interaction_fluency: { label: '交互流畅度' },
  language_expression: { label: '语言表达' },
  context_adaptation: { label: '情境适应性' },
  personalization: { label: '个性化体验' }
}

// 获取评分值的辅助函数
const getScoreValue = (key, category) => {
  if (!evaluationStats.value || !evaluationStats.value[category] || !evaluationStats.value[category][key]) {
    return 0
  }
  return evaluationStats.value[category][key].avg || 0
}

// 修改显示统计报告的方法
const showEvaluationReport = (stats) => {
  if (!stats) {
    console.error('没有收到统计数据')
    systemMessage.value = '没有收到有效的统计数据，无法显示报告！'
    return
  }
  
  console.log('收到评估统计数据:', stats)
  
  // 直接使用统计数据
  evaluationStats.value = stats
  
  // 显示成功消息
  systemMessage.value = `评估完成！总体评分: ${stats.overall_scores?.final_score || 'N/A'}，角色评分: ${stats.overall_scores?.role_score || 'N/A'}，对话评分: ${stats.overall_scores?.dialogue_score || 'N/A'}`
  
  // 停止扫描效果
  isScanning.value = false
  
  // 自动切换到报告视图
  changeChannel(2)
}

// 添加电源状态变量
const isPoweredOn = ref(true)
const systemStatus = ref('系统就绪')

// 电源开关函数
const togglePower = () => {
  isPoweredOn.value = !isPoweredOn.value
  
  if (!isPoweredOn.value) {
    // 关闭电源
    activeChannel.value = 0 // 无频道
    systemStatus.value = '系统待机'
  } else {
    // 打开电源
    activeChannel.value = 1 // 默认频道1
    systemStatus.value = '系统就绪'
  }
}

// 格式化文件大小
const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// 获取关键词的辅助函数
const getKeywords = (key, category) => {
  if (!evaluationStats.value || 
      !evaluationStats.value[category] || 
      !evaluationStats.value[category][key] || 
      !evaluationStats.value[category][key].keywords) {
    return {}
  }
  return evaluationStats.value[category][key].keywords
}

// 计算关键词大小的辅助函数
const getKeywordSize = (count) => {
  // 根据关键词出现频率计算字体大小
  const baseSize = 0.9;
  const maxSize = 2.2;
  return Math.min(baseSize + (count / 5) * 0.5, maxSize);
}

// 获取格式化的关键词数组，包含样式
const getFormattedKeywords = (category, key) => {
  const keywords = getKeywords(key, category);
  if (Object.keys(keywords).length === 0) return [];
  
  // 创建缓存key
  const cacheKey = `${category}-${key}`;
  
  // 如果没有缓存，创建新的位置数据
  if (!keywordPositions.value[cacheKey]) {
    keywordPositions.value[cacheKey] = {};
  }
  
  // 获取最大计数值用于归一化
  const counts = Object.values(keywords);
  const maxCount = Math.max(...counts);
  
  // 按计数从大到小排序关键词
  const sortedKeywords = Object.entries(keywords).sort((a, b) => b[1] - a[1]);
  
  // 保存已放置的元素区域，用于避免重叠
  const placedAreas = [];
  
  // 从中心向外的分层布局配置
  const centerX = 50; // 中心点X坐标（百分比）
  const centerY = 50; // 中心点Y坐标（百分比）
  
  // 格式化关键词数组
  const result = [];
  for (let i = 0; i < sortedKeywords.length; i++) {
    const [text, count] = sortedKeywords[i];
    
    // 归一化权重 (0.1 - 1.0)
    const normalizedWeight = 0.3 + (count / maxCount) * 0.7;
    
    // 如果该关键词没有缓存位置，创建一个
    if (!keywordPositions.value[cacheKey][text]) {
      // 根据权重和索引计算极坐标
      // 权重越高，距离中心越近
      // 同等权重的词按索引顺序分布在不同角度
      
      // 索引角度 - 均匀分布在圆周上，但添加一些随机性
      const angle = (i * 137.5 + Math.random() * 20) % 360; // 黄金角分布 + 随机偏移
      
      // 距离 - 重要的词更靠近中心，不重要的词更远离中心
      // 1.0是最重要的词，会有一个最小距离
      // 0.1是最不重要的词，会有一个最大距离
      const minDistance = 5; // 最小距离（百分比）
      const maxDistance = 40; // 最大距离（百分比）
      const distance = minDistance + (1 - normalizedWeight) * (maxDistance - minDistance);
      
      // 将极坐标转换为笛卡尔坐标（百分比）
      const radians = angle * (Math.PI / 180);
      const x = centerX + distance * Math.cos(radians);
      const y = centerY + distance * Math.sin(radians);
      
      // 创建位置对象
      const position = {
        left: `${x}%`,
        top: `${y}%`,
        rotation: `${(Math.random() * 20 - 10) + angle * 0.1}deg`, // 旋转角度与位置角度相关
        delay: `${Math.random() * 2}s`,
        duration: `${3 + Math.random() * 2}s`
      };
      
      // 估算元素大小
      const fontSize = getKeywordSize(count);
      const estimatedWidth = text.length * fontSize * 0.6; // 粗略估算宽度
      const estimatedHeight = fontSize * 1.5; // 估算高度
      
      // 创建此元素的区域对象
      const area = {
        left: x - estimatedWidth/2,
        right: x + estimatedWidth/2,
        top: y - estimatedHeight/2,
        bottom: y + estimatedHeight/2
      };
      
      // 检查与已放置元素是否重叠
      let overlap = placedAreas.some(placed => {
        return !(
          area.right < placed.left || 
          area.left > placed.right || 
          area.bottom < placed.top || 
          area.top > placed.bottom
        );
      });
      
      // 如果重叠，尝试调整位置（最多20次）
      let attempts = 0;
      const maxAttempts = 20;
      
      while (overlap && attempts < maxAttempts) {
        // 尝试微调位置，保持相同方向但距离略有不同
        const adjustedAngle = angle + (Math.random() * 30 - 15);
        const adjustedDistance = distance * (0.9 + Math.random() * 0.2);
        
        const adjRadians = adjustedAngle * (Math.PI / 180);
        const adjX = centerX + adjustedDistance * Math.cos(adjRadians);
        const adjY = centerY + adjustedDistance * Math.sin(adjRadians);
        
        position.left = `${adjX}%`;
        position.top = `${adjY}%`;
        
        // 更新区域
        area.left = adjX - estimatedWidth/2;
        area.right = adjX + estimatedWidth/2;
        area.top = adjY - estimatedHeight/2;
        area.bottom = adjY + estimatedHeight/2;
        
        // 重新检查重叠
        overlap = placedAreas.some(placed => {
          return !(
            area.right < placed.left || 
            area.left > placed.right || 
            area.bottom < placed.top || 
            area.top > placed.bottom
          );
        });
        
        attempts++;
      }
      
      // 无论是否重叠，都添加区域和保存位置
      placedAreas.push(area);
      keywordPositions.value[cacheKey][text] = position;
    }
    
    // 从缓存获取位置
    const position = keywordPositions.value[cacheKey][text];
    
    // 创建样式对象
    const style = {
      fontSize: `${getKeywordSize(count)}rem`,
      left: position.left,
      top: position.top,
      transform: `rotate(${position.rotation})`,
      opacity: 0.7 + (normalizedWeight * 0.3), // 高频词更不透明
      animationDelay: position.delay,
      animationDuration: position.duration,
      // 基于词频调整发光效果
      textShadow: `0 0 ${3 + normalizedWeight * 7}px rgba(68, 255, 68, ${0.5 + normalizedWeight * 0.5})`
    };
    
    result.push({
      text,
      count,
      style,
      weight: normalizedWeight
    });
  }
  
  return result;
}

// 在script部分添加新的数据和方法
const evaluationCode = ref('')
const roleInfo = ref('')
const savedReports = ref([])
const selectedReports = ref([])
const activeComparisonTab = ref('consistency')
const comparisonColors = ref([
  '#44ff44', '#ff5252', '#52a2ff', '#ffbd52', 
  '#e552ff', '#52ffbd', '#ff52a2', '#bdff52'
])

// 切换报告选择状态
const toggleReportSelection = (reportId) => {
  const index = selectedReports.value.indexOf(reportId)
  if (index === -1) {
    // 最多只能选择3个报告进行对比
    if (selectedReports.value.length < 3) {
      selectedReports.value.push(reportId)
    } else {
      systemStatus.value = '最多只能选择3个报告进行对比'
    }
  } else {
    selectedReports.value.splice(index, 1)
  }
}

// 通过ID获取报告
const getReportById = (id) => {
  return savedReports.value.find(report => report.id === id) || {}
}

// 获取报告颜色（用于对比图）
const getReportColor = (reportId) => {
  const index = selectedReports.value.indexOf(reportId)
  return comparisonColors.value[index % comparisonColors.value.length]
}

// 修改获取维度评分的函数，添加安全访问
const getDimensionScore = (reportId, category, dimension) => {
  const report = getReportById(reportId)
  if (!report || !report.stats) return 0
  
  try {
    // 使用可选链操作符安全访问嵌套属性
    return report.stats[category]?.[dimension]?.avg || 0
  } catch (e) {
    console.error('Error getting dimension score:', e)
    return 0
  }
}

// 修改获取总分的函数
const getOverallScore = (reportId, scoreType) => {
  const report = getReportById(reportId)
  if (!report || !report.stats || !report.stats.overall_scores) return 0
  
  try {
    return report.stats.overall_scores[scoreType] || 0
  } catch (e) {
    console.error('Error getting overall score:', e)
    return 0
  }
}

// 格式化日期
const formatDate = (date) => {
  if (!date) return ''
  
  if (typeof date === 'string') {
    date = new Date(date)
  }
  
  return `${date.getMonth() + 1}/${date.getDate()} ${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`
}

// 下载报告
const downloadReport = (report) => {
  if (!report) return
  
  const reportData = JSON.stringify(report, null, 2)
  const blob = new Blob([reportData], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  
  const a = document.createElement('a')
  a.href = url
  a.download = `${report.evaluation_code}_${formatDateForFilename(report.timestamp)}.json`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// 下载当前报告
const downloadCurrentReport = () => {
  if (!evaluationStats.value) return
  
  const report = {
    evaluation_code: evaluationCode.value,
    timestamp: new Date(),
    stats: evaluationStats.value,
    role_info: roleInfo.value
  }
  
  downloadReport(report)
}

// 移除保存的报告
const removeReport = (reportId) => {
  const index = savedReports.value.findIndex(report => report.id === reportId)
  if (index !== -1) {
    savedReports.value.splice(index, 1)
    localStorage.setItem('savedReports', JSON.stringify(savedReports.value))
    
    // 如果已选中，也要从选中列表中移除
    const selectedIndex = selectedReports.value.indexOf(reportId)
    if (selectedIndex !== -1) {
      selectedReports.value.splice(selectedIndex, 1)
    }
  }
}

// 格式化用于文件名的日期
const formatDateForFilename = (date) => {
  if (typeof date === 'string') {
    date = new Date(date)
  }
  
  return `${date.getFullYear()}${String(date.getMonth() + 1).padStart(2, '0')}${String(date.getDate()).padStart(2, '0')}_${String(date.getHours()).padStart(2, '0')}${String(date.getMinutes()).padStart(2, '0')}`
}

// 保存当前报告
const saveReport = () => {
  if (!evaluationStats.value) return
  
  const reportId = Date.now().toString()
  const report = {
    id: reportId,
    evaluation_code: evaluationCode.value,
    timestamp: new Date(),
    stats: JSON.parse(JSON.stringify(evaluationStats.value)), // 深拷贝
    role_info: roleInfo.value
  }
  
  savedReports.value.push(report)
  
  // 保存到本地存储
  localStorage.setItem('savedReports', JSON.stringify(savedReports.value))
  
  // 显示成功消息
  systemStatus.value = `报告已保存: ${evaluationCode.value}`
  
  // 选中新保存的报告
  if (selectedReports.value.length < 2) {
    selectedReports.value.push(reportId)
  }
}

// 完全移除 watch(activeChannel) 中的下载相关逻辑
watch(activeChannel, (newChannel, oldChannel) => {
  isChangingChannel.value = true
  setTimeout(() => {
    isChangingChannel.value = false
  }, 1000)
})

// 添加报告文件上传处理函数
const handleReportFileUpload = async (event) => {
  const files = event.target.files
  if (!files || files.length === 0) return

  for (const file of files) {
    try {
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const reportData = JSON.parse(e.target.result)
          // 验证文件格式是否符合要求
          if (validateReportFormat(reportData)) {
            // 生成报告对象，与保存报告格式一致
            const report = {
              id: `imported-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
              evaluation_code: file.name.replace('.json', ''),
              timestamp: new Date(),
              stats: reportData.stats, // 直接使用上传的stats
              role_info: reportData.role_info || '' // 使用上传的role_info，如果没有则设为空字符串
            }
            
            // 添加到已保存报告列表
            savedReports.value.push(report)
            
            // 如果选中报告少于3个，自动选中新上传的报告
            if (selectedReports.value.length < 3) {
              selectedReports.value.push(report.id)
            }
            
            // 显示成功消息
            systemMessage.value = `报告已上传: ${file.name}`
          } else {
            throw new Error('文件格式不符合要求')
          }
        } catch (error) {
          console.error('Error parsing JSON:', error)
          systemMessage.value = `文件解析失败: ${file.name}`
        }
      }
      reader.readAsText(file)
    } catch (error) {
      console.error('Error reading file:', error)
      systemMessage.value = `文件读取失败: ${file.name}`
    }
  }
  // 清空input以允许重复上传相同文件
  event.target.value = ''
}

// 修改报告格式验证函数，添加更严格的检查
const validateReportFormat = (report) => {
  try {
    // 检查必要的字段
    const requiredFields = [
      'evaluation_code',
      'timestamp',
      'stats',
      'role_info'
    ]
    
    // 检查stats中的必要字段
    const requiredStatsFields = [
      'overall_scores',
      'role_play',
      'dialogue_experience'
    ]
    
    // 检查overall_scores中的必要字段
    const requiredOverallScoresFields = [
      'role_score',
      'dialogue_score',
      'final_score'
    ]
    
    // 检查role_play中的必要维度
    const requiredRolePlayDimensions = [
      'consistency',
      'knowledge',
      'language_style',
      'emotional_expression',
      'character_depth'
    ]
    
    // 检查dialogue_experience中的必要维度
    const requiredDialogueExperienceDimensions = [
      'response_quality',
      'interaction_fluency',
      'language_expression',
      'context_adaptation',
      'personalization'
    ]
    
    // 检查顶层字段
    if (!requiredFields.every(field => report?.hasOwnProperty(field))) {
      return false
    }
    
    // 检查stats字段
    if (!report.stats || !requiredStatsFields.every(field => report.stats?.hasOwnProperty(field))) {
      return false
    }
    
    // 检查overall_scores字段
    if (!report.stats.overall_scores || !requiredOverallScoresFields.every(field => report.stats.overall_scores?.hasOwnProperty(field))) {
      return false
    }
    
    // 检查role_play字段
    if (!report.stats.role_play || !requiredRolePlayDimensions.every(dimension => report.stats.role_play?.hasOwnProperty(dimension))) {
      return false
    }
    
    // 检查dialogue_experience字段
    if (!report.stats.dialogue_experience || !requiredDialogueExperienceDimensions.every(dimension => report.stats.dialogue_experience?.hasOwnProperty(dimension))) {
      return false
    }
    
    return true
  } catch (e) {
    console.error('Error validating report format:', e)
    return false
  }
}

// 添加导出图表函数
const exportReportChart = () => {
  const chartElement = document.querySelector('.report-container')
  if (!chartElement) {
    systemMessage.value = '未找到可导出的图表'
    return
  }
  
  html2canvas(chartElement).then(canvas => {
    const link = document.createElement('a')
    link.download = `report-chart-${Date.now()}.png`
    link.href = canvas.toDataURL('image/png')
    link.click()
  })
}

// 添加字体加载函数
const loadChineseFont = async () => {
  try {
    const response = await fetch('/fonts/SourceHanSansSC-Regular.otf')
    const fontBuffer = await response.arrayBuffer()
    return fontBuffer
  } catch (error) {
    console.error('加载字体失败:', error)
    throw error
  }
}

// 修改导出报告函数
const exportReportPDF = async () => {
  try {
    if (!evaluationStats.value) {
      systemMessage.value = '没有可导出的报告数据'
      return
    }

    // 创建PDF文档，使用内置的中文支持
    const doc = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4',
      putOnlyUsedFonts: true,
      language: 'zh-CN'
    })

    // 使用内置字体
    doc.setFont('helvetica', 'normal')

    // 确保所有文本内容都是字符串类型
    const safeText = (text) => String(text || '')
    
    // 确保数字是有效的
    const safeNumber = (num) => Number(num) || 0

    // 添加标题
    doc.setFontSize(16)
    doc.text(`评估报告 - ${safeText(evaluationCode.value)}`, 15, 20, { charSpace: 0.5 })

    // 添加基本信息
    doc.setFontSize(10)
    doc.text(`生成时间：${new Date().toLocaleString('zh-CN')}`, 15, 30)
    doc.text(`评估代号：${safeText(evaluationCode.value)}`, 15, 35)
    doc.text(`角色信息：${safeText(roleInfo.value)}`, 15, 40)

    // 准备总体评分数据
    const scores = evaluationStats.value.overall_scores || {}
    const overallScores = [
      ['总体评分', safeNumber(scores.final_score)],
      ['角色评分', safeNumber(scores.role_score)],
      ['对话评分', safeNumber(scores.dialogue_score)]
    ]

    // 添加总体评分表格
    doc.setFontSize(12)
    doc.text('总体评分', 15, 50)
    const overallTable = doc.autoTable({
      startY: 55,
      head: [['评分类型', '分数']],
      body: overallScores.map(([label, score]) => [
        label,
        score.toFixed(2)
      ]),
      theme: 'grid',
      styles: {
        fontSize: 10,
        font: 'helvetica',
        cellPadding: 3
      },
      headStyles: {
        fillColor: [68, 255, 68],
        textColor: [0, 0, 0],
        fontSize: 10,
        fontStyle: 'bold',
        halign: 'center'
      },
      columnStyles: {
        0: { halign: 'left' },
        1: { halign: 'center' }
      }
    })

    // 准备角色扮演评分数据
    const rolePlays = evaluationStats.value.role_play || {}
    const rolePlayData = Object.entries(rolePlays).map(([key, value]) => {
      const item = rolePlayItems[key] || { label: key }
      return [
        safeText(item.label),
        safeNumber(value?.avg),
        safeNumber(value?.min),
        safeNumber(value?.max)
      ]
    })

    // 添加角色扮演评分表格
    const rolePlayY = (overallTable.finalY || 55) + 10
    doc.text('角色扮演评分', 15, rolePlayY)
    const rolePlayTable = doc.autoTable({
      startY: rolePlayY + 5,
      head: [['维度', '平均分', '最低分', '最高分']],
      body: rolePlayData.map(row => row.map(val => 
        typeof val === 'number' ? val.toFixed(2) : val
      )),
      theme: 'grid',
      styles: {
        fontSize: 10,
        font: 'helvetica',
        cellPadding: 3
      },
      headStyles: {
        fillColor: [68, 255, 68],
        textColor: [0, 0, 0],
        fontSize: 10,
        fontStyle: 'bold',
        halign: 'center'
      },
      columnStyles: {
        0: { halign: 'left' },
        1: { halign: 'center' },
        2: { halign: 'center' },
        3: { halign: 'center' }
      }
    })

    // 准备对话体验评分数据
    const dialogues = evaluationStats.value.dialogue_experience || {}
    const dialogueData = Object.entries(dialogues).map(([key, value]) => {
      const item = dialogueItems[key] || { label: key }
      return [
        safeText(item.label),
        safeNumber(value?.avg),
        safeNumber(value?.min),
        safeNumber(value?.max)
      ]
    })

    // 添加对话体验评分表格
    const dialogueY = (rolePlayTable.finalY || rolePlayY + 50) + 10
    doc.text('对话体验评分', 15, dialogueY)
    doc.autoTable({
      startY: dialogueY + 5,
      head: [['维度', '平均分', '最低分', '最高分']],
      body: dialogueData.map(row => row.map(val => 
        typeof val === 'number' ? val.toFixed(2) : val
      )),
      theme: 'grid',
      styles: {
        fontSize: 10,
        font: 'helvetica',
        cellPadding: 3
      },
      headStyles: {
        fillColor: [68, 255, 68],
        textColor: [0, 0, 0],
        fontSize: 10,
        fontStyle: 'bold',
        halign: 'center'
      },
      columnStyles: {
        0: { halign: 'left' },
        1: { halign: 'center' },
        2: { halign: 'center' },
        3: { halign: 'center' }
      }
    })

    // 保存PDF
    const filename = `评估报告_${safeText(evaluationCode.value)}_${new Date().getTime()}.pdf`
    doc.save(filename)
    systemMessage.value = '报告已导出为PDF'

  } catch (error) {
    console.error('PDF生成错误:', error)
    systemMessage.value = '报告生成失败，请重试'
  }
}

// 添加导出对比报告函数
const exportComparisonCSV = () => {
  if (selectedReports.value.length === 0) {
    systemMessage.value = '请先选择要对比的报告'
    return
  }

  // 准备CSV数据
  const headers = [
    '报告代号',
    '时间',
    '总体评分',
    '角色评分', 
    '对话评分'
  ]

  const rows = selectedReports.value.map(reportId => {
    const report = getReportById(reportId)
    return [
      report.evaluation_code,
      formatDate(report.timestamp),
      report.stats.overall_scores.final_score,
      report.stats.overall_scores.role_score,
      report.stats.overall_scores.dialogue_score
    ]
  })

  // 生成CSV内容
  const csvContent = [
    headers.join(','),
    ...rows.map(row => row.join(','))
  ].join('\n')

  // 创建下载链接
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = `评估对比报告_${new Date().toISOString().slice(0,10)}.csv`
  link.click()
  URL.revokeObjectURL(link.href)

  systemMessage.value = '对比报告已导出为CSV文件'
}

// 添加导出单个报告为CSV的函数
const exportReportCSV = () => {
  if (!evaluationStats.value) {
    systemMessage.value = '没有可导出的报告数据'
    return
  }

  try {
    // 准备CSV数据
    const headers = [
      '评估项目',
      '平均分',
      '最低分',
      '最高分',
      '类别'
    ]

    const rows = []

    // 添加总体评分
    const overallScores = evaluationStats.value.overall_scores
    rows.push([
      '总体评分',
      overallScores.final_score,
      '',
      '',
      '总分'
    ])
    rows.push([
      '角色评分',
      overallScores.role_score,
      '',
      '',
      '总分'
    ])
    rows.push([
      '对话评分',
      overallScores.dialogue_score,
      '',
      '',
      '总分'
    ])

    // 添加角色扮演评分
    Object.entries(evaluationStats.value.role_play || {}).forEach(([key, value]) => {
      const item = rolePlayItems[key] || { label: key }
      rows.push([
        item.label,
        value.avg || 0,
        value.min || 0,
        value.max || 0,
        '角色扮演'
      ])
    })

    // 添加对话体验评分
    Object.entries(evaluationStats.value.dialogue_experience || {}).forEach(([key, value]) => {
      const item = dialogueItems[key] || { label: key }
      rows.push([
        item.label,
        value.avg || 0,
        value.min || 0,
        value.max || 0,
        '对话体验'
      ])
    })

    // 生成CSV内容
    const csvContent = [
      // 添加基本信息
      `评估代号,${evaluationCode.value}`,
      `评估时间,${formatDate(new Date())}`,
      `角色信息,${roleInfo.value.replace(/,/g, ';')}`, // 替换逗号以避免CSV格式问题
      '', // 空行分隔
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n')

    // 创建并下载文件
    const blob = new Blob([new Uint8Array([0xEF, 0xBB, 0xBF]), csvContent], { 
      type: 'text/csv;charset=utf-8'
    })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `评估报告_${evaluationCode.value}_${formatDateForFilename(new Date())}.csv`
    link.click()
    URL.revokeObjectURL(link.href)

    systemMessage.value = '报告已导出为CSV文件'
  } catch (error) {
    console.error('导出CSV错误:', error)
    systemMessage.value = '导出失败，请重试'
  }
}
</script>

<style scoped>
/* 组件特定的布局覆盖 */
.evaluation-content {
  height: 100%;
  overflow-y: auto;
  padding: 1rem;
}

/* 响应式设计覆盖 */
@media (max-width: 768px) {
  .tv-container {
    flex-direction: column;
    padding: 1rem;
    height: 100vh;
    overflow-y: auto;
  }

  .control-panel {
    width: 100%;
    height: auto;
    min-height: 200px;
  }

  .tv-screen {
    transform: none;
  }
}
</style>

/**
 * EvaluationPanel 组件
 * 
 * 这是一个模拟复古电视机的AI对话评估界面组件。
 * 
 * 特色功能:
 * 1. 复古CRT电视机外观 - 包括屏幕玻璃效果、扫描线、微光和反光效果
 * 2. 频道切换系统 - 模拟老式电视的换台效果，带有静态噪声和扫描线动画
 * 3. 三个频道功能:
 *    - 频道1: 评估过程显示，带有打字机效果的结果输出
 *    - 频道2: 评估报告显示，包含图表和详细分析
 *    - 频道3: 导出功能，触发评估报告下载
 * 4. 复古状态效果:
 *    - 待机画面: 显示系统消息，带有扫描线和脉冲动画
 *    - 无信号效果: 模拟老式电视无信号时的静态噪点和闪烁文字
 *    - 扫描效果: 模拟CRT电视的扫描线移动
 * 
 * 设计理念:
 * 通过怀旧的复古电视机界面，为AI评估工具增添趣味性和独特的用户体验，
 * 同时保持功能的完整性和数据的清晰展示。
 */ 