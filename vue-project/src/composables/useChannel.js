import { ref } from 'vue'
import { useEvaluationStore } from '@/stores/evaluation'

export function useChannel() {
  const store = useEvaluationStore()
  const isChangingChannel = ref(false)
  const channelChangeTimeout = ref(null)

  const changeChannel = async (newChannel) => {
    if (store.activeChannel === newChannel) return

    isChangingChannel.value = true
    store.setChannelChanging(true)

    // 清除之前的超时
    if (channelChangeTimeout.value) {
      clearTimeout(channelChangeTimeout.value)
    }

    // 模拟换台效果
    channelChangeTimeout.value = setTimeout(() => {
      store.setActiveChannel(newChannel)
      isChangingChannel.value = false
      store.setChannelChanging(false)
    }, 1000)
  }

  return {
    isChangingChannel,
    changeChannel
  }
} 