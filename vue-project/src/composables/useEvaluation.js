import { ref, computed } from 'vue'
import { useEvaluationStore } from '@/stores/evaluation'

export function useEvaluation() {
  const store = useEvaluationStore()
  const isEvaluating = ref(false)
  const processed = ref(0)
  const total = ref(0)
  const evaluationText = ref('')
  const systemMessage = ref('')

  const progress = computed(() => {
    if (total.value === 0) return 0
    return (processed.value / total.value) * 100
  })

  const startEvaluation = async (data) => {
    isEvaluating.value = true
    store.setScanning(true)
    systemMessage.value = '评估开始...'
    
    try {
      // 评估逻辑
      processed.value = 0
      total.value = data.length
      
      for (const item of data) {
        // 处理每个评估项
        processed.value++
        // ... 评估逻辑
      }
      
      store.setEvaluationComplete(true)
      systemMessage.value = '评估完成'
    } catch (error) {
      systemMessage.value = '评估过程出错'
      console.error('Evaluation error:', error)
    } finally {
      isEvaluating.value = false
      store.setScanning(false)
    }
  }

  return {
    isEvaluating,
    processed,
    total,
    progress,
    evaluationText,
    systemMessage,
    startEvaluation
  }
} 