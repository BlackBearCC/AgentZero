// 评估配置接口
interface EvaluationConfig {
  evaluationCode: string
  roleInfo?: string
  evalType: 'dialogue' | 'memory'
  selectedFields: string[]
}

// 评估报告接口
interface EvaluationReport {
  id: number
  timestamp: string
  evaluationCode: string
  roleInfo?: string
  evalType: string
  results: {
    field: string
    score: number
    comments: string[]
  }[]
  summary: string
  totalScore: number
}

// 字段定义接口
interface FieldDefinition {
  name: string
  label: string
  description: string
  weight: number
}

// 评估状态接口
interface EvaluationStatus {
  isPoweredOn: boolean
  activeChannel: number
  isChangingChannel: boolean
  isScanning: boolean
  systemMessage: string
}

export {
  EvaluationConfig,
  EvaluationReport,
  FieldDefinition,
  EvaluationStatus
} 