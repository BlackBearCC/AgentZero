<template>
  <div class="retro-robot" :class="{ 'talking': isTalking, 'thinking': isThinking }">
    <div class="robot-head">
      <div class="robot-face">
        <!-- 眼睛 -->
        <div class="robot-eyes">
          <div class="robot-eye left" :class="{ 'blink': isBlinking }"></div>
          <div class="robot-eye right" :class="{ 'blink': isBlinking }"></div>
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
    animation: slight-shift 8s ease-in-out infinite;
    
    .robot-eye {
      width: 40px;
      height: 10px;
      background: #00c3ff;
      box-shadow: 0 0 8px #00c3ff;
      transition: all 0.2s ease;
      
      &.blink {
        height: 2px;
        transform: translateY(4px);
      }
      
      &.left {
        animation: eye-move-left 6s ease-in-out infinite;
      }
      
      &.right {
        animation: eye-move-right 6s ease-in-out infinite;
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

@keyframes slight-shift {
  0%, 100% { transform: translateY(0); }
  30% { transform: translateY(-2px); }
  70% { transform: translateY(2px); }
}

@keyframes eye-move-left {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-2px); }
  75% { transform: translateX(1px); }
}

@keyframes eye-move-right {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(2px); }
  75% { transform: translateX(-1px); }
}
</style>