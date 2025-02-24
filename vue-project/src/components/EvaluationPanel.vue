<template>
  <div class="evaluation-container">
    <div class="file-selector">
      <input type="file" @change="handleFileUpload" accept=".csv,.xlsx" />
      <select v-model="selectedEvalType">
        <option value="dialogue">对话质量评估</option>
        <option value="memory">记忆相关性评估</option>
      </select>
      <button @click="startEvaluation" :disabled="isEvaluating">
        {{ isEvaluating ? '评估中...' : '开始评估' }}
      </button>
    </div>

    <div class="progress" v-if="isEvaluating">
      进度: {{ processed }}/{{ total }}
      <div class="progress-bar">
        <div :style="progressStyle"></div>
      </div>
    </div>

    <div class="results" v-if="results.length">
      <div class="result-item" v-for="(item, index) in results" :key="index">
        <h3>测试用例 {{ index + 1 }}</h3>
        <div class="input-text">{{ item.input }}</div>
        <div class="eval-result">
          <div>评分: {{ item.evaluation?.score ?? 0 }}/100</div>
          <div>评估理由: {{ item.evaluation?.reason }}</div>
          <div>建议: {{ item.evaluation?.suggestions?.join('，') }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000' // 修改为你的后端地址

const selectedFile = ref(null)
const selectedEvalType = ref('dialogue')
const isEvaluating = ref(false)
const results = ref([])
const processed = ref(0)
const total = ref(0)

const progressStyle = computed(() => ({
  width: `${(processed.value / total.value) * 100}%`
}))

const handleFileUpload = (event) => {
  selectedFile.value = event.target.files[0]
}

const startEvaluation = async () => {
  if (!selectedFile.value) return
  
  const formData = new FormData()
  formData.append('file', selectedFile.value)
  formData.append('eval_type', selectedEvalType.value)

  try {
    isEvaluating.value = true
    results.value = []
    processed.value = 0
    
    // 修改API路径，添加 v1 前缀
    const response = await fetch(`${API_BASE_URL}/api/v1/evaluate`, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value)
      try {
        const result = JSON.parse(chunk)
        results.value.push(result)
        processed.value = results.value.length
        total.value = results.value.length
      } catch (e) {
        console.error('Error parsing chunk:', e)
      }
    }

  } catch (error) {
    console.error('评估失败:', error)
  } finally {
    isEvaluating.value = false
  }
}
</script>

<style scoped>
.evaluation-container {
  max-width: 800px;
  margin: 20px auto;
}

.file-selector {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.progress-bar {
  height: 20px;
  background: #eee;
  border-radius: 10px;
  overflow: hidden;
}

.progress-bar div {
  height: 100%;
  background: #42b983;
  transition: width 0.3s ease;
}

.result-item {
  margin: 15px 0;
  padding: 15px;
  border: 1px solid #eee;
  border-radius: 8px;
}

.input-text {
  color: #666;
  margin: 10px 0;
}

.eval-result div {
  margin: 5px 0;
}
</style> 