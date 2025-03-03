import axios from 'axios'

const api = axios.create({
  baseURL: process.env.VUE_APP_API_BASE_URL || '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

/**
 * 上传评估文件
 * @param {File} file 文件对象
 * @returns {Promise} 包含字段信息的Promise
 */
export const uploadFile = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await api.post('/evaluation/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
  return response.data
}

/**
 * 开始评估
 * @param {EvaluationConfig} config 评估配置
 * @returns {Promise} 评估结果Promise
 */
export const startEvaluation = async (config) => {
  const response = await api.post('/evaluation/start', config)
  return response.data
}

/**
 * 获取评估报告
 * @param {string} evaluationCode 评估代号
 * @returns {Promise} 评估报告Promise
 */
export const getEvaluationReport = async (evaluationCode) => {
  const response = await api.get(`/evaluation/report/${evaluationCode}`)
  return response.data
}

/**
 * 导出评估报告
 * @param {string} evaluationCode 评估代号
 * @returns {Promise} 导出文件的Promise
 */
export const exportReport = async (evaluationCode) => {
  const response = await api.get(`/evaluation/export/${evaluationCode}`, {
    responseType: 'blob'
  })
  return response.data
}

/**
 * 获取字段定义
 * @returns {Promise} 字段定义列表Promise
 */
export const getFieldDefinitions = async () => {
  const response = await api.get('/evaluation/fields')
  return response.data
}

// 错误处理拦截器
api.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error)
    throw new Error(error.response?.data?.message || '请求失败')
  }
)

export default api 