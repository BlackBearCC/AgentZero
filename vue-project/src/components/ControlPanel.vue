<template>
  <div class="tv-container">
    <!-- 控制面板 -->
    <div class="control-panel">
      <div class="panel-title">CONTROL</div>
      
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
      
      <!-- 音量控制 -->
      <div class="control-group">
        <div class="control-label">VOLUME</div>
        <div class="volume-control">
          <input 
            type="range" 
            min="0" 
            max="100" 
            v-model="volume" 
            class="volume-slider"
            :disabled="!isPoweredOn"
          />
          <div class="volume-value">{{ volume }}</div>
        </div>
      </div>
      
      <!-- 亮度控制 -->
      <div class="control-group">
        <div class="control-label">BRIGHTNESS</div>
        <div class="brightness-control">
          <input 
            type="range" 
            min="0" 
            max="100" 
            v-model="brightness" 
            class="brightness-slider"
            :disabled="!isPoweredOn"
          />
          <div class="brightness-value">{{ brightness }}</div>
        </div>
      </div>
      <div class="action-buttons">
        <TvButton 
          primary 
        >
          [ 测试按钮 ]
        </TvButton>
      </div>
      <!-- 系统状态 -->
      <div class="system-status">
        <div class="status-label">SYSTEM STATUS</div>
        <div class="status-value">{{ systemStatus }}</div>
      </div>
    </div>

    <!-- 屏幕内容 - 简化结构 -->
    <div class="screen-content" :class="{ 'scanning': isScanning, 'changing-channel': isChangingChannel }" :style="{ filter: `brightness(${brightness}%)` }">
      <!-- 未开机状态 -->
      <div v-if="!isPoweredOn" class="power-off-screen">
        <div class="power-off-dot"></div>
      </div>
      
      <template v-if="isPoweredOn">
        <!-- 屏幕上方固定展示机器人 -->
        <div class="screen-header">
          <div class="robot-container">
            <RetroRobot 
              :mood="robotMood" 
              :talking="robotTalking" 
              :active="isPoweredOn"
            />
          </div>
        </div>
        
        <!-- 屏幕下方展示频道内容 -->
        <div class="screen-content-area">
          <component 
            :is="currentScreenComponent" 
            @scanning:start="startScanning" 
            @scanning:stop="stopScanning"
            @update:status="updateSystemStatus"
          />
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import {
  TvButton,
  TvPanel,
  TvScreen,
  TvTitle,
  TvCheckbox,
  TvControlGroup,
  TvSlider,
  TvFileInput,
  RetroRobot
} from '../design-system/components'
import { ref, computed, watch } from 'vue'
import CharacterGenerator from './CharacterGenerator.vue'
import BatchDialogue from './BatchDialogue.vue'
import EvaluationCenter from './EvaluationCenter.vue'

// 频道相关变量
const activeChannel = ref(1) // 默认显示角色生成器
const isChangingChannel = ref(false) // 是否正在换台
const isScanning = ref(false) // 扫描效果
const isPoweredOn = ref(true) // 电源状态
const systemStatus = ref('系统就绪') // 系统状态
const volume = ref(50) // 音量控制
const brightness = ref(100) // 亮度控制

// 机器人状态
const robotMood = ref('neutral')
const robotTalking = ref(false)

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
  systemStatus.value = `切换至频道 ${channel}`
  
  // 机器人状态变化
  robotMood.value = 'thinking'
  robotTalking.value = true
  
  // 延迟切换频道，模拟换台过程
  setTimeout(() => {
    activeChannel.value = channel
    
    // 结束换台效果
    setTimeout(() => {
      isChangingChannel.value = false
      isScanning.value = false
      systemStatus.value = '系统就绪'
      
      // 恢复机器人状态
      robotMood.value = 'neutral'
      robotTalking.value = false
    }, 500)
  }, 1000)
}

// 电源开关函数
const togglePower = () => {
  isPoweredOn.value = !isPoweredOn.value
  
  if (!isPoweredOn.value) {
    // 关闭电源
    systemStatus.value = '系统待机'
    robotTalking.value = false
  } else {
    // 打开电源
    systemStatus.value = '系统启动中'
    robotMood.value = 'happy'
    robotTalking.value = true
    
    // 模拟启动过程
    setTimeout(() => {
      systemStatus.value = '系统就绪'
      robotTalking.value = false
      robotMood.value = 'neutral'
    }, 2000)
  }
}

// 扫描效果控制
const startScanning = () => {
  isScanning.value = true
  robotMood.value = 'thinking'
}

const stopScanning = () => {
  isScanning.value = false
  robotMood.value = 'neutral'
}

// 更新系统状态
const updateSystemStatus = (status) => {
  systemStatus.value = status
  
  // 当状态更新时，让机器人短暂说话
  robotTalking.value = true
  setTimeout(() => {
    robotTalking.value = false
  }, 2000)
}

// 监听音量变化
watch(volume, (newVolume) => {
  if (newVolume > 80) {
    robotMood.value = 'happy'
  } else if (newVolume < 20) {
    robotMood.value = 'sad'
  } else {
    robotMood.value = 'neutral'
  }
})
</script>

<style>
/* 添加到现有的 ControlPanel.css 文件中 */
.screen-content {
  display: flex;
  flex-direction: column;
  height: 100%;
  position: relative;
}

.screen-header {
  height: 30%; /* 屏幕上方30%的区域用于展示机器人 */
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
  border-bottom: 1px solid rgba(0, 195, 255, 0.3);
}

.robot-container {
  width: 66.6%; /* 占屏幕的2/3 */
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.screen-content-area {
  height: 70%; /* 屏幕下方70%的区域用于展示频道内容 */
  width: 100%;
  overflow: auto;
  position: relative;
}

/* 移除之前的 robot-overlay 样式 */
.robot-overlay {
  display: none;
}
</style>

<style src="./styles/ControlPanel.css"></style>

/**
 * ControlPanel 组件
 * 
 * 这是一个模拟银翼杀手、重返未来1999的复古科幻电脑的主控界面组件。
 * 
 * 特色功能:
 * 1. 复古科幻CRT屏幕外观 - 包括屏幕玻璃效果、扫描线、微光和反光效果
 * 2. 频道切换系统 - 
 * 3. 三个频道功能:
 *    - 频道1: 角色生成器
 *    - 频道2: 批量对话
 *    - 频道3: 评估中心
 * 4. 复古控制面板:
 *    - 电源开关
 *    - 频道选择
 *    - 音量调节
 *    - 亮度调节
 * 
 * 设计理念:
 * 通过怀旧的复古科幻电脑界面，为AI工具增添趣味性和独特的用户体验。
 */