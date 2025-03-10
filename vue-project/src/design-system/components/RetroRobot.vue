<template>
  <div class="retro-robot" 
       :class="{ 'talking': isTalking, 'thinking': isThinking }"
       @mousemove="handleMouseMove"
       ref="robotContainer">
    <div class="robot-head">
      <div class="robot-face">
        <!-- 眼睛 -->
        <div class="robot-eyes">
          <div class="robot-eye left" 
               :class="{ 'blink': isBlinking }"
               :style="leftEyeStyle"></div>
          <div class="robot-eye right" 
               :class="{ 'blink': isBlinking }"
               :style="rightEyeStyle"></div>
        </div>
        
        <!-- 信号指示器 -->
        <div class="signal-indicators">
          <div class="signal-dot" :class="{ 'active': isActive }"></div>
          <div class="signal-dot delayed" :class="{ 'active': isActive }"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, computed } from 'vue';

const props = defineProps({
  mood: {
    type: String,
    default: 'neutral', // neutral, happy, sad, thinking
    validator: (value) => ['neutral', 'happy', 'sad', 'thinking'].includes(value)
  },
  active: {
    type: Boolean,
    default: true
  },
  talking: {
    type: Boolean,
    default: false
  }
});

const isBlinking = ref(false);
const isTalking = ref(props.talking);
const isActive = ref(props.active);
const isHappy = ref(props.mood === 'happy');
const isSad = ref(props.mood === 'sad');
const isThinking = ref(props.mood === 'thinking');

// 鼠标跟踪相关变量
const robotContainer = ref(null);
const mousePosition = ref({ x: 0, y: 0 });
const eyeMovementRange = 5; // 眼睛移动的最大范围（像素）

// 计算左眼样式
const leftEyeStyle = computed(() => {
  if (isBlinking.value) return {};
  
  return {
    transform: `translate(${mousePosition.value.x * eyeMovementRange}px, ${mousePosition.value.y * eyeMovementRange}px)`
  };
});

// 计算右眼样式
const rightEyeStyle = computed(() => {
  if (isBlinking.value) return {};
  
  return {
    transform: `translate(${mousePosition.value.x * eyeMovementRange}px, ${mousePosition.value.y * eyeMovementRange}px)`
  };
});

// 处理鼠标移动
const handleMouseMove = (event) => {
  if (!robotContainer.value) return;
  
  const rect = robotContainer.value.getBoundingClientRect();
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;
  
  // 计算鼠标相对于容器中心的位置（范围从-1到1）
  const relativeX = (event.clientX - centerX) / (window.innerWidth / 2);
  const relativeY = (event.clientY - centerY) / (window.innerHeight / 2);
  
  // 限制值在-1到1之间
  mousePosition.value = {
    x: Math.max(-1, Math.min(1, relativeX)),
    y: Math.max(-1, Math.min(1, relativeY))
  };
};

// 监听属性变化
watch(() => props.mood, (newMood) => {
  isHappy.value = newMood === 'happy';
  isSad.value = newMood === 'sad';
  isThinking.value = newMood === 'thinking';
});

watch(() => props.active, (newActive) => {
  isActive.value = newActive;
});

watch(() => props.talking, (newTalking) => {
  isTalking.value = newTalking;
});

// 眨眼动画
let blinkInterval;
const startBlinking = () => {
  blinkInterval = setInterval(() => {
    isBlinking.value = true;
    setTimeout(() => {
      isBlinking.value = false;
    }, 200);
  }, 3000 + Math.random() * 2000); // 随机间隔眨眼
};

// 组件挂载时启动动画
onMounted(() => {
  startBlinking();
});

// 组件卸载时清除定时器
onUnmounted(() => {
  if (blinkInterval) clearInterval(blinkInterval);
});
</script>
  
<style lang="scss" scoped>
.retro-robot {
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
  
  &.thinking {
    animation: thinking 2s infinite;
  }
  
  &.talking {
    animation: subtle-float 3s ease-in-out infinite;
  }
  
  .robot-head {
    width: 100%;
    height: 100%;
    background: transparent;
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
    animation: glow 4s infinite alternate;
    
    &::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(90deg, 
        rgba(0, 195, 255, 0) 0%, 
        rgba(0, 195, 255, 0.03) 30%,
        rgba(0, 195, 255, 0.05) 50%, 
        rgba(0, 195, 255, 0.03) 70%,
        rgba(0, 195, 255, 0) 100%
      );
      filter: blur(15px);
      pointer-events: none;
      z-index: -1;
    }
  }
  
  .robot-face {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    height: 100%;
    position: relative;
  }
  
  .robot-eyes {
    display: flex;
    gap: 120px;
    
    .robot-eye {
      width: 40px;
      height: 10px;
      background: #00c3ff;
      box-shadow: 0 0 8px #00c3ff;
      transition: transform 0.2s ease, height 0.2s ease;
      
      &.blink {
        height: 2px;
        transform: translateY(4px) !important;
      }
    }
  }
  
  .signal-indicators {
    position: absolute;
    top: 10px;
    right: 10px;
    display: flex;
    gap: 6px;
    
    .signal-dot {
      width: 4px;
      height: 4px;
      background: rgba(0, 195, 255, 0.3);
      
      &.active {
        background: #00c3ff;
        box-shadow: 0 0 5px #00c3ff;
        animation: pulse 1s infinite;
      }
      
      &.delayed {
        animation-delay: 0.5s;
      }
    }
  }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

@keyframes thinking {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-2px); }
  75% { transform: translateX(2px); }
}

@keyframes subtle-float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-3px); }
}

@keyframes glow {
  0%, 100% { 
    box-shadow: 
      0 0 30px rgba(0, 195, 255, 0.02),
      0 0 50px rgba(0, 195, 255, 0.01),
      inset 0 0 20px rgba(0, 195, 255, 0.02);
  }
  50% { 
    box-shadow: 
      0 0 40px rgba(0, 195, 255, 0.03),
      0 0 60px rgba(0, 195, 255, 0.02),
      inset 0 0 25px rgba(0, 195, 255, 0.03);
  }
}
</style>