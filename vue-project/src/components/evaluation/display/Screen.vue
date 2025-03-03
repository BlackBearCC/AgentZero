<template>
  <div class="tv-screen">
    <div 
      class="screen-frame" 
      :class="{ 
        'scanning': isScanning, 
        'changing-channel': isChangingChannel 
      }"
    >
      <div class="screen-content">
        <!-- 待机画面 -->
        <div v-if="!isPoweredOn" class="standby-screen">
          <div class="tv-logo">AI EVAL</div>
          <div class="standby-message">系统待机中...</div>
          <div class="standby-animation"></div>
        </div>
        
        <!-- 无数据时显示无信号 -->
        <div v-else-if="showNoSignal" class="no-signal">
          <div class="static-effect"></div>
          <div class="no-signal-text">NO SIGNAL</div>
          <div class="scan-line"></div>
        </div>

        <!-- 有数据时显示内容 -->
        <slot v-else></slot>
      </div>
    </div>
  </div>
</template>

<script setup>
import { storeToRefs } from 'pinia'
import { useEvaluationStore } from '@/stores/evaluation'

const store = useEvaluationStore()
const { isPoweredOn, isScanning, isChangingChannel } = storeToRefs(store)

// 由父组件控制是否显示无信号
const props = defineProps({
  showNoSignal: {
    type: Boolean,
    default: false
  }
})
</script>

<style scoped>
.tv-screen {
  flex: 1;
  background: #000;
  border-radius: 20px;
  padding: 20px;
  position: relative;
  overflow: hidden;
  height: calc(100vh - 4rem);
  min-width: 0;
  /* 外壳立体效果 */
  box-shadow: 
    -5px -5px 15px rgba(255,255,255,0.1),
    5px 5px 15px rgba(0,0,0,0.4);
  border: 2px solid #2a2a3a;
  background: 
    linear-gradient(45deg, #1a1a2e, #2a2a3a) padding-box,
    linear-gradient(45deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)) border-box;
}

.screen-frame {
  background: linear-gradient(45deg, #1a1a2e, #2a2a3a);
  border-radius: 15px;
  padding: 15px;
  height: 100%;
  position: relative;
  overflow: hidden;
  /* 玻璃屏幕表面效果 */
  box-shadow: 
    inset 0 0 50px rgba(0,0,0,0.5),
    inset 0 0 20px rgba(0,0,0,0.3);
}

/* 玻璃表面反光效果 */
.screen-frame::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: 
    repeating-linear-gradient(
      0deg,
      rgba(255,255,255,0) 0%,
      rgba(255,255,255,0.03) 0.5%,
      rgba(255,255,255,0) 1%
    ),
    radial-gradient(
      circle at center,
      rgba(255,255,255,0.1) 0%,
      rgba(0,0,0,0.2) 100%
    );
  pointer-events: none;
  animation: scanlines 8s linear infinite;
  border-radius: 15px;
  z-index: 2;
  opacity: 0.6;
}

/* 移动扫描效果只在scanning时显示 */
.screen-frame::after {
  content: '';
  position: absolute;
  top: -100%;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    0deg,
    transparent 0%,
    rgba(255,255,255,0.05) 50%,
    transparent 100%
  );
  pointer-events: none;
  z-index: 2;
  opacity: 0;
  animation: scanning 3s linear infinite;
  animation-play-state: paused;
}

.screen-frame.scanning::after {
  opacity: 1;
  animation-play-state: running;
}

.screen-content {
  height: 100%;
  position: relative;
  z-index: 1;
}

/* 换台效果 */
.changing-channel::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    to bottom,
    transparent,
    rgba(255, 255, 255, 0.2) 20%,
    rgba(255, 255, 255, 0.2) 80%,
    transparent
  );
  z-index: 10;
  animation: channel-change 1s ease-in-out;
  pointer-events: none;
}

@keyframes channel-change {
  0% { transform: translateY(-100%); opacity: 0.8; }
  50% { transform: translateY(0); opacity: 1; }
  100% { transform: translateY(100%); opacity: 0.8; }
}

@keyframes scanning {
  0% { transform: translateY(-100%); }
  100% { transform: translateY(200%); }
}

@keyframes scanlines {
  0% { background-position: 0 0; }
  100% { background-position: 0 100%; }
}

/* CRT效果 */
.tv-screen::before {
  content: '';
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: 
    linear-gradient(
      rgba(18, 16, 16, 0) 50%, 
      rgba(0, 0, 0, 0.25) 50%
    ),
    linear-gradient(
      90deg,
      rgba(255, 0, 0, 0.06),
      rgba(0, 255, 0, 0.02),
      rgba(0, 0, 255, 0.06)
    );
  background-size: 100% 2px, 3px 100%;
  pointer-events: none;
  z-index: 10;
  opacity: 0.15;
}

/* 响应式设计优化 */
@media (max-width: 768px) {
  .tv-screen {
    transform: none;
  }
}

/* 待机画面样式 */
.standby-screen {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  position: relative;
  background: #000;
}

.standby-screen::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: 
    repeating-linear-gradient(
      0deg,
      rgba(0, 0, 0, 0.15) 0px,
      rgba(0, 0, 0, 0.15) 1px,
      transparent 1px,
      transparent 2px
    );
  pointer-events: none;
  z-index: 1;
}

.tv-logo {
  font-size: 3rem;
  font-weight: bold;
  letter-spacing: 5px;
  margin-bottom: 2rem;
  position: relative;
  z-index: 2;
  animation: pulse 2s infinite alternate;
}

.standby-message {
  font-size: 1.2rem;
  max-width: 80%;
  text-align: center;
  line-height: 1.6;
  background: rgba(0, 0, 0, 0.5);
  padding: 1rem;
  border-radius: 10px;
  border: 1px solid rgba(68, 255, 68, 0.3);
  position: relative;
  z-index: 2;
}

.standby-animation {
  position: absolute;
  bottom: 10%;
  width: 60%;
  height: 4px;
  background: rgba(68, 255, 68, 0.5);
  border-radius: 2px;
  overflow: hidden;
  z-index: 2;
}

.standby-animation::after {
  content: '';
  position: absolute;
  top: 0;
  left: -50%;
  width: 50%;
  height: 100%;
  background: rgba(68, 255, 68, 0.8);
  animation: loading 2s ease infinite;
}

@keyframes loading {
  0% { transform: translateX(0); }
  100% { transform: translateX(200%); }
}

@keyframes pulse {
  0% { text-shadow: 0 0 5px rgba(68, 255, 68, 0.5); }
  100% { text-shadow: 0 0 20px rgba(68, 255, 68, 0.8); }
}

/* 无信号效果 */
.no-signal {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background: #000;
}

.static-effect {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-image: repeating-radial-gradient(
    circle at 17% 32%,
    rgba(255,255,255,0.1),
    rgba(255,255,255,0.1) 1px,
    transparent 1px,
    transparent 2px
  );
  animation: static 0.5s steps(3, end) infinite;
  opacity: 0.3;
}

.no-signal-text {
  font-size: 3rem;
  font-weight: bold;
  color: #fff;
  text-shadow: 0 0 10px rgba(255,255,255,0.5);
  z-index: 2;
  animation: flicker 0.3s infinite;
  letter-spacing: 5px;
}

.scan-line {
  position: absolute;
  width: 100%;
  height: 2px;
  background: rgba(255,255,255,0.5);
  animation: scan 2s linear infinite;
}

@keyframes static {
  0% { transform: translate(0, 0); }
  25% { transform: translate(-1%, 1%); }
  50% { transform: translate(1%, -1%); }
  75% { transform: translate(-1%, -1%); }
  100% { transform: translate(1%, 1%); }
}

@keyframes flicker {
  0% { opacity: 1; }
  50% { opacity: 0.8; }
  100% { opacity: 1; }
}

@keyframes scan {
  0% { transform: translateY(-100%); }
  100% { transform: translateY(100%); }
}
</style> 