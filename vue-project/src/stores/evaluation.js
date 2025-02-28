import { defineStore } from 'pinia'
import { startEvaluation, exportReport } from '@/api/evaluation'
import { generateEvalCode } from '@/utils/formatters'

export const useEvaluationStore = defineStore('evaluation', {
  state: () => ({
    isPoweredOn: true,
    systemStatus: '系统就绪',
    activeChannel: 1,
    isChangingChannel: false,
    isScanning: false,
    availableFields: [],
    selectedFields: [],
    fieldsConfirmed: false,
    systemMessage: '',
    evaluationHistory: [],
    comparisonMode: false,
    selectedReportsForComparison: [],
    evaluationText: '',         // 评估文本
    evaluationCode: '',         // 评估代号
    roleInfo: '',              // 人设信息
    selectedEvalType: 'dialogue', // 评估类型
    processed: 0,              // 已处理数量
    total: 0,                  // 总数量
    evaluationInProgress: false,
    currentEvaluation: null,
    evaluationResults: null,
  }),

  getters: {
    progressPercentage: (state) => {
      if (state.total === 0) return 0
      return Math.round((state.processed / state.total) * 100)
    }
  },

  actions: {
    togglePower() {
      this.isPoweredOn = !this.isPoweredOn
      
      if (!this.isPoweredOn) {
        // 关闭电源
        this.activeChannel = 0 // 无频道
        this.systemStatus = '系统待机'
        this.isChangingChannel = false
        this.isScanning = false
      } else {
        // 打开电源
        this.activeChannel = 1 // 默认频道1
        this.systemStatus = '系统就绪'
      }
    },

    setActiveChannel(channel) {
      if (this.isPoweredOn && channel >= 0 && channel <= 3) {
        this.activeChannel = channel
        this.systemStatus = `当前频道: ${channel}`
      }
    },

    setChannelChanging(isChanging) {
      this.isChangingChannel = isChanging
    },

    setScanning(isScanning) {
      this.isScanning = isScanning
    },

    setAvailableFields(fields) {
      this.availableFields = fields
      this.selectedFields = []
      this.fieldsConfirmed = false
    },

    setSelectedFields(fields) {
      this.selectedFields = fields
    },

    setFieldsConfirmed(confirmed) {
      this.fieldsConfirmed = confirmed
    },

    setSystemMessage(message) {
      this.systemMessage = message
    },

    initializeSystem() {
      this.isPoweredOn = true
      this.activeChannel = 1
      this.systemStatus = '系统就绪'
      this.loadSavedReports()
    },

    resetAllStates() {
      this.isPoweredOn = false
      this.activeChannel = 0
      this.isScanning = false
      this.systemStatus = '系统待机'
      this.evaluationText = ''
      this.selectedFields = []
      this.fieldsConfirmed = false
    },

    saveReport(report) {
      this.evaluationHistory.push(report)
      // 保存到本地存储
      localStorage.setItem('evaluationReports', JSON.stringify(this.evaluationHistory))
    },

    loadSavedReports() {
      const savedReports = localStorage.getItem('evaluationReports')
      if (savedReports) {
        this.evaluationHistory = JSON.parse(savedReports)
      }
    },

    toggleReportSelection(reportId) {
      const index = this.selectedReportsForComparison.indexOf(reportId)
      if (index === -1) {
        if (this.selectedReportsForComparison.length < 3) {
          this.selectedReportsForComparison.push(reportId)
        }
      } else {
        this.selectedReportsForComparison.splice(index, 1)
      }
    },

    async startEvaluation(file, selectedFields) {
      if (!file || !this.fieldsConfirmed) return
      
      try {
        this.evaluationInProgress = true
        this.setScanning(true)
        this.setSystemMessage('正在评估，请稍候...')
        
        const formData = new FormData()
        formData.append('file', file)
        formData.append('eval_type', this.selectedEvalType)
        formData.append('selected_fields', JSON.stringify(selectedFields))
        formData.append('evaluation_code', this.evaluationCode || generateEvalCode())
        
        if (this.roleInfo?.trim()) {
          formData.append('role_info', this.roleInfo.trim())
        }

        const response = await fetch(`${API_BASE_URL}/api/v1/evaluate/stream`, {
          method: 'POST',
          body: formData
        })

        if (!response.ok) throw new Error('评估请求失败')
        
        const reader = response.body.getReader()
        let evaluationText = ''
        
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          
          const chunk = new TextDecoder().decode(value)
          const lines = chunk.split('\n').filter(Boolean)
          
          for (const line of lines) {
            const data = JSON.parse(line)
            
            switch (data.type) {
              case 'start':
                this.total = data.total
                break
                
              case 'progress':
                this.processed = data.index
                evaluationText += data.text
                this.setEvaluationText(evaluationText)
                break
                
              case 'complete':
                this.evaluationResults = data.stats
                // 保存报告
                this.saveReport({
                  id: Date.now().toString(),
                  evaluation_code: this.evaluationCode,
                  timestamp: new Date(),
                  stats: data.stats,
                  role_info: this.roleInfo
                })
                break
            }
          }
        }

        // 自动切换到报告频道
        this.setActiveChannel(2)
        
      } catch (error) {
        this.setSystemMessage(`评估失败: ${error.message}`)
        console.error('Evaluation error:', error)
      } finally {
        this.evaluationInProgress = false
        this.setScanning(false)
      }
    },

    async exportCurrentReport() {
      if (!this.currentReport) return
      try {
        await exportReport(this.currentReport.evaluationCode)
        this.setSystemMessage('报告导出成功')
      } catch (error) {
        this.setSystemMessage('报告导出失败：' + error.message)
      }
    },

    resetEvaluation() {
      this.evaluationCode = ''
      this.roleInfo = ''
      this.evaluationText = ''
      this.processed = 0
      this.total = 0
    }
  }
}) 