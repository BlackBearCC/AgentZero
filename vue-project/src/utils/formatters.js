/**
 * 格式化文件大小
 * @param {number} bytes 文件大小（字节）
 * @returns {string} 格式化后的文件大小
 */
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`
}

/**
 * 格式化时间戳
 * @param {string} timestamp ISO时间字符串
 * @returns {string} 格式化后的时间
 */
export const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp)
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

/**
 * 生成随机评估代号
 * @returns {string} 随机生成的评估代号
 */
export const generateEvalCode = () => {
  const prefix = 'EVAL'
  const timestamp = Date.now().toString().slice(-6)
  const random = Math.random().toString(36).substring(2, 5).toUpperCase()
  return `${prefix}-${timestamp}-${random}`
}

/**
 * 格式化评估分数
 * @param {number} score 评估分数
 * @returns {string} 格式化后的分数
 */
export const formatScore = (score) => {
  return score.toFixed(1)
}

/**
 * 计算总分
 * @param {Array} results 评估结果数组
 * @returns {number} 总分
 */
export const calculateTotalScore = (results) => {
  if (!results || results.length === 0) return 0
  const total = results.reduce((sum, item) => sum + item.score, 0)
  return +(total / results.length).toFixed(1)
} 