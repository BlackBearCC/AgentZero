<template>
  <div class="control-group">
    <div class="control-label">INPUT</div>
    <label class="control-button file-input-button">
      <div class="button-face">
        <span>上传文件</span>
        <i class="upload-icon">↑</i>
      </div>
      <input 
        type="file" 
        @change="handleFileUpload" 
        accept=".csv,.xls,.xlsx,.json" 
        class="hidden-file-input" 
      />
    </label>
    
    <!-- 文件信息显示 -->
    <div class="file-info" v-if="selectedFile">
      <div class="file-name">{{ selectedFile.name }}</div>
      <div class="file-size">{{ formatFileSize(selectedFile.size) }}</div>
    </div>

    <!-- 评估代号输入 -->
    <div v-if="selectedFile" class="eval-code-input">
      <input 
        v-model="evaluationCode" 
        type="text" 
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
</template>

<script setup>
import { ref } from 'vue'
import { useEvaluationStore } from '@/stores/evaluation'

const store = useEvaluationStore()
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

// 状态
const selectedFile = ref(null)
const evaluationCode = ref('')

// 词组库
const wordLists = {
  games: ['魂斗罗', '双截龙', '坦克大战', '忍者龙剑传', '洛克人', '恶魔城', '冒险岛', '赤色要塞', 
          '超级马里奥', '塞尔达传说', '银河战士', '最终幻想', '勇者斗恶龙', '街头霸王', '快打旋风', 
          '魔界村', '绿色兵团', '沙罗曼蛇', '赤影战士', '忍者神龟', '超级魂斗罗', '热血物语'],
  suffixes: ['I', 'II', 'III', 'IV', 'V', 'EX', 'DX', 'PLUS', 'ULTRA', 'SPECIAL']
}

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
    store.setAvailableFields(data.columns)
  } catch (error) {
    console.error('Error:', error)
    store.setSystemMessage('文件处理失败')
  }
}

// 格式化文件大小
const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

// 导出必要的状态和方法
defineExpose({
  selectedFile,
  evaluationCode,
  generateRandomCode
})
</script>

<style scoped>
/* 移除 .control-group 和 .control-label 的样式定义,因为已在 main.css 中定义 */

.file-input-button {
  position: relative;
  overflow: hidden;
}

.hidden-file-input {
  position: absolute;
  top: 0;
  left: 0;
  opacity: 0;
  width: 100%;
  height: 100%;
  cursor: pointer;
}

.upload-icon {
  font-style: normal;
  font-size: 1.2em;
}

.file-info {
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
  font-size: 0.8rem;
}

.file-name {
  color: var(--primary-color);
  margin-bottom: 0.25rem;
  word-break: break-all;
}

.file-size {
  color: rgba(var(--primary-color-rgb), 0.7);
}

.eval-code-input {
  margin-top: 0.5rem;
  display: flex;
  gap: 0.5rem;
}

.code-input {
  flex: 1;
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid var(--border-color);
  color: var(--primary-color);
  padding: 0.5rem;
  border-radius: 4px;
  font-family: monospace;
}

.code-input:focus {
  outline: none;
  border-color: var(--primary-color);
  box-shadow: 0 0 5px var(--shadow-color);
}

.small-btn {
  padding: 0.25rem 0.5rem;
  font-size: 0.8rem;
}
</style> 