<template>
  <div class="character-generator">
    <div v-if="!characterData && !isGenerating" class="empty-state">
      <div class="tv-logo">CHARACTER GENERATOR</div>
      <div class="channel-info">频道 1</div>
      <div class="instruction-text">请使用左侧控制面板上传角色资料并开始生成</div>
    </div>
    
    <div v-else-if="isGenerating" class="generating-state">
      <div class="tv-logo">CHARACTER GENERATOR</div>
      <div class="processing-message">正在生成角色信息...</div>
      <div class="processing-animation"></div>
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: `${generationProgress}%` }"></div>
      </div>
    </div>
    
    <div v-else-if="characterData" class="result-state">
      <div class="character-header">
        <h2>角色信息卡</h2>
        <div class="character-actions">
          <button @click="exportCharacterData" class="crt-button">
            <span class="button-text">[ 导出角色卡 ]</span>
          </button>
        </div>
      </div>
      
      <div class="character-card">
        <div class="character-name">{{ characterData.name }}</div>
        <div class="character-tags">
          <div v-for="(tag, index) in characterData.tags" :key="index" class="character-tag">
            {{ tag }}
          </div>
        </div>
        <div class="character-description">
          <pre class="typewriter-text">{{ characterData.description }}</pre>
        </div>
        <div class="character-traits">
          <h3>核心特质</h3>
          <div class="traits-list">
            <div v-for="(trait, index) in characterData.traits" :key="index" class="trait-item">
              <div class="trait-name">{{ trait.name }}</div>
              <div class="trait-bar-container">
                <div class="trait-bar" :style="{ width: `${trait.value}%` }"></div>
              </div>
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

// 角色生成相关状态
const isGenerating = ref(false)
const generationProgress = ref(0)
const characterData = ref(null)

// 导出角色数据
const exportCharacterData = () => {
  if (!characterData.value) return
  
  const dataStr = JSON.stringify(characterData.value, null, 2)
  const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr)
  
  const exportFileDefaultName = `character_${characterData.value.name.replace(/\s+/g, '_')}.json`
  
  const linkElement = document.createElement('a')
  linkElement.setAttribute('href', dataUri)
  linkElement.setAttribute('download', exportFileDefaultName)
  linkElement.click()
}

// 暴露给父组件的方法
defineExpose({
  startGeneration(sourceData) {
    isGenerating.value = true
    generationProgress.value = 0
    characterData.value = null
    emit('scanning:start')
    
    // 模拟生成进度
    const progressInterval = setInterval(() => {
      if (generationProgress.value < 95) {
        generationProgress.value += Math.random() * 5
      }
    }, 300)
    
    // 模拟生成完成
    setTimeout(() => {
      clearInterval(progressInterval)
      generationProgress.value = 100
      
      // 模拟生成结果
      characterData.value = {
        name: "测试角色",
        tags: ["友善", "聪明", "勇敢", "创造力强"],
        description: "这是一个测试角色的描述。在实际应用中，这里会显示由AI根据上传资料生成的详细角色描述。",
        traits: [
          { name: "智力", value: 85 },
          { name: "情感", value: 70 },
          { name: "社交", value: 60 },
          { name: "创造力", value: 90 }
        ]
      }
      
      setTimeout(() => {
        isGenerating.value = false
        emit('scanning:stop')
      }, 500)
    }, 3000)
  },
  
  reset() {
    isGenerating.value = false
    generationProgress.value = 0
    characterData.value = null
  }
})
</script>

<style scoped>
.character-generator {
  height: 100%;
  padding: 20px;
  overflow-y: auto;
  color: #e0e0e0;
}

.empty-state, .generating-state {
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

.instruction-text {
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

.processing-message {
  font-size: 1.2rem;
  color: #e0e0e0;
  margin-bottom: 2rem;
}

.processing-animation {
  width: 100px;
  height: 100px;
  border: 5px solid rgba(68, 255, 68, 0.3);
  border-top: 5px solid #44ff44;
  border-radius: 50%;
  animation: spin 1.5s linear infinite;
  margin-bottom: 2rem;
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
}

.progress-fill {
  height: 100%;
  background: linear-gradient(to right, #44ff44, #88ff88);
  border-radius: 7px;
  transition: width 0.3s ease;
}

.result-state {
  height: 100%;
  overflow-y: auto;
}

.character-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.character-header h2 {
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

.character-card {
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 10px;
  padding: 1.5rem;
}

.character-name {
  font-size: 1.8rem;
  font-weight: bold;
  color: #44ff44;
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.5);
  margin-bottom: 1rem;
  text-align: center;
}

.character-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
}

.character-tag {
  background: rgba(68, 255, 68, 0.2);
  border: 1px solid rgba(68, 255, 68, 0.5);
  color: #44ff44;
  padding: 0.3rem 0.8rem;
  border-radius: 15px;
  font-size: 0.8rem;
}

.character-description {
  margin-bottom: 1.5rem;
  line-height: 1.6;
}

.typewriter-text {
  font-family: 'Courier New', monospace;
  white-space: pre-wrap;
  margin: 0;
}

.character-traits h3 {
  color: #44ff44;
  margin-bottom: 1rem;
}

.traits-list {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.trait-item {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.trait-name {
  width: 100px;
  font-size: 0.9rem;
}

.trait-bar-container {
  flex: 1;
  height: 10px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 5px;
  overflow: hidden;
}

.trait-bar {
  height: 100%;
  background: linear-gradient(to right, #44ff44, #88ff88);
  border-radius: 5px;
}
</style> 