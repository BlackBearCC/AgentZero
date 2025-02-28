<template>
  <div class="control-group">
    <div class="control-label">POWER</div>
    <button @click="togglePower" class="control-button">
      <div class="button-face">
        <span>{{ isPoweredOn ? 'ON' : 'OFF' }}</span>
        <div class="power-indicator" :class="{ 'active': isPoweredOn }"></div>
      </div>
    </button>
  </div>
</template>

<script setup>
import { storeToRefs } from 'pinia'
import { useEvaluationStore } from '@/stores/evaluation'

const store = useEvaluationStore()
const { isPoweredOn, systemStatus } = storeToRefs(store)

const emit = defineEmits(['power-change'])

const togglePower = () => {
  store.togglePower()
  emit('power-change', isPoweredOn.value)
}
</script>

<style scoped>
/* 移除重复的通用样式,只保留组件特有的样式 */
</style>
