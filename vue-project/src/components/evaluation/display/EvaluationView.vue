<template>
  <div class="chat-window evaluation-view">
    <!-- 评估过程显示 -->
    <div class="chat-container">
      <div class="message system-message">
        <div class="message-header">
          <span class="system-badge">SYS</span>
          <span class="timestamp">{{ getCurrentTime() }}</span>
        </div>
        <div class="message-content">
          {{ systemMessage }}
        </div>
      </div>

      <!-- 评估进度 -->
      <ProgressBar 
        v-if="evaluationInProgress"
        :processed="processed"
        :total="total"
      />

      <!-- 评估结果 -->
      <div v-if="evaluationText" class="message evaluation-message">
        <div class="message-content typewriter">
          {{ evaluationText }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { storeToRefs } from 'pinia'
import { useEvaluationStore } from '@/stores/evaluation'
import ProgressBar from '@/components/utils/ProgressBar.vue'

const store = useEvaluationStore()
const { 
  systemMessage,
  evaluationText,
  evaluationInProgress,
  processed,
  total
} = storeToRefs(store)

const getCurrentTime = () => new Date().toLocaleTimeString()
</script>

<style scoped>
.chat-window {
  height: 100%;
  padding: 1rem;
  overflow-y: auto;
  color: #44ff44;
  font-family: monospace;
}

.chat-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.message {
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 4px;
  padding: 1rem;
}

.system-message {
  border-left: 3px solid #44ff44;
}

.message-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
}

.system-badge {
  background: rgba(68, 255, 68, 0.2);
  padding: 0.2rem 0.5rem;
  border-radius: 2px;
  font-weight: bold;
}

.timestamp {
  color: rgba(68, 255, 68, 0.7);
}

.message-content {
  line-height: 1.5;
}

/* 打字机效果 */
.typewriter {
  overflow: hidden;
  border-right: 2px solid #44ff44;
  white-space: pre-wrap;
  animation: typing 3s steps(40, end), blink-caret 0.75s step-end infinite;
}

@keyframes typing {
  from { width: 0 }
  to { width: 100% }
}

@keyframes blink-caret {
  from, to { border-color: transparent }
  50% { border-color: #44ff44 }
}
</style> 