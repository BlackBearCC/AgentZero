<template>
  <div class="tv-container">
    <!-- 控制面板 -->
    <div class="control-panel">
      <div class="panel-title">控制中心</div>
      
      <!-- 电源控制 -->
      <div class="control-group">
        <div class="control-label">POWER</div>
        <button @click="togglePower" class="control-button">
          <div class="button-face">
            <span>{{ isPoweredOn ? 'ON' : 'OFF' }}</span>
            <div class="power-indicator" :class="{ 'active': isPoweredOn }"></div>
          </div>
        </button>
      </div>
      
      <!-- 频道控制 -->
      <div class="control-group">
        <div class="control-label">CHANNEL</div>
        <div class="channel-buttons">
          <button @click="changeChannel(1)" class="control-button channel-btn" :class="{ 'active': activeChannel === 1 }">1</button>
          <button @click="changeChannel(2)" class="control-button channel-btn" :class="{ 'active': activeChannel === 2 }">2</button>
          <button @click="changeChannel(3)" class="control-button channel-btn" :class="{ 'active': activeChannel === 3 }">3</button>
        </div>
        <div class="channel-labels">
          <div class="channel-label" :class="{ 'active': activeChannel === 1 }">角色生成</div>
          <div class="channel-label" :class="{ 'active': activeChannel === 2 }">批量对话</div>
          <div class="channel-label" :class="{ 'active': activeChannel === 3 }">评估中心</div>
        </div>
      </div>
      
      <!-- 动态加载当前频道的控制面板 -->
      <component 
        :is="currentControlComponent" 
        v-if="isPoweredOn" 
        @update:status="updateSystemStatus" 
      />
      
      <!-- 系统状态 -->
      <div class="system-status">
        <div class="status-label">SYSTEM STATUS</div>
        <div class="status-value">{{ systemStatus }}</div>
      </div>
    </div>

    <!-- 电视屏幕 -->
    <div class="tv-screen">
      <div class="screen-frame" :class="{ 'scanning': isScanning, 'changing-channel': isChangingChannel }">
        <div class="screen-content">
          <!-- 未开机状态 -->
          <div v-if="!isPoweredOn" class="power-off-screen">
            <div class="power-off-dot"></div>
          </div>
          
          <!-- 动态加载当前频道内容 -->
          <component 
            v-if="isPoweredOn" 
            :is="currentScreenComponent" 
            @scanning:start="startScanning" 
            @scanning:stop="stopScanning" 
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import CharacterGenerator from './CharacterGenerator.vue'
import BatchDialogue from './BatchDialogue.vue'
import EvaluationCenter from './EvaluationCenter.vue'
import CharacterGeneratorControls from './controls/CharacterGeneratorControls.vue'
import BatchDialogueControls from './controls/BatchDialogueControls.vue'
import EvaluationControls from './controls/EvaluationControls.vue'

// 频道相关变量
const activeChannel = ref(3) // 默认显示评估中心
const isChangingChannel = ref(false) // 是否正在换台
const isScanning = ref(false) // 扫描效果
const isPoweredOn = ref(true) // 电源状态
const systemStatus = ref('系统就绪') // 系统状态

// 计算当前应该显示的控制组件
const currentControlComponent = computed(() => {
  switch (activeChannel.value) {
    case 1: return CharacterGeneratorControls
    case 2: return BatchDialogueControls
    case 3: return EvaluationControls
    default: return null
  }
})

// 计算当前应该显示的屏幕组件
const currentScreenComponent = computed(() => {
  switch (activeChannel.value) {
    case 1: return CharacterGenerator
    case 2: return BatchDialogue
    case 3: return EvaluationCenter
    default: return null
  }
})

// 修改换台函数
const changeChannel = (channel) => {
  if (channel === activeChannel.value || !isPoweredOn.value) return
  
  // 开始换台效果
  isChangingChannel.value = true
  isScanning.value = true
  
  // 延迟切换频道，模拟换台过程
  setTimeout(() => {
    activeChannel.value = channel
    
    // 结束换台效果
    setTimeout(() => {
      isChangingChannel.value = false
      isScanning.value = false
    }, 500)
  }, 1000)
}

// 电源开关函数
const togglePower = () => {
  isPoweredOn.value = !isPoweredOn.value
  
  if (!isPoweredOn.value) {
    // 关闭电源
    systemStatus.value = '系统待机'
  } else {
    // 打开电源
    systemStatus.value = '系统就绪'
  }
}

// 扫描效果控制
const startScanning = () => {
  isScanning.value = true
}

const stopScanning = () => {
  isScanning.value = false
}

// 更新系统状态
const updateSystemStatus = (status) => {
  systemStatus.value = status
}
</script>

<style src="./styles/ControlPanel.css"></style>