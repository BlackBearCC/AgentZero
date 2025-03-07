<template>
    <div :class="['tv-screen', { 'scanning': scanning, 'changing-channel': changing }]">
      <div class="scan-line"></div>
      <div class="horizontal-scan"></div>
      <div class="screen-glow"></div>
      <div class="hologram-lines" v-if="hologram"></div>
      <div class="power-off-screen" v-if="powerOff">
        <div class="power-off-dot"></div>
      </div>
      <div class="screen-content" v-else>
        <slot></slot>
      </div>
    </div>
  </template>
  
  <script setup>
  defineProps({
    scanning: {
      type: Boolean,
      default: false
    },
    changing: {
      type: Boolean,
      default: false
    },
    hologram: {
      type: Boolean,
      default: false
    },
    powerOff: {
      type: Boolean,
      default: false
    },
    brightness: {
      type: Number,
      default: 100
    }
  })
  </script>
  
  <style lang="scss" scoped>
  .tv-screen {
    background: #1a1a2a;
    background: linear-gradient(135deg, #1a1a2a 0%, #2a2a3a 100%);
    border: 1px solid rgba(0, 195, 255, 0.3);
    border-radius: 5px;
    box-shadow: inset 0 0 15px rgba(0, 195, 255, 0.3);
    position: relative;
    overflow-y: auto;
    padding: 20px;
    min-height: 200px;
    font-family: 'Share Tech Mono', monospace;
    flex: 1;
    width: 100%;
    height: 100%;
    color: #00c3ff;
    text-shadow: 0 0 5px rgba(0, 195, 255, 0.5);
    
    // 自定义滚动条
    &::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
  
    &::-webkit-scrollbar-track {
      background: rgba(0, 20, 40, 0.5);
    }
  
    &::-webkit-scrollbar-thumb {
      background: rgba(0, 195, 255, 0.3);
      border-radius: 4px;
    }
      
    &::-webkit-scrollbar-thumb:hover {
      background: rgba(0, 195, 255, 0.5);
    }
    
    &::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.05'/%3E%3C/svg%3E");
      opacity: 0.08;
      pointer-events: none;
      z-index: 1;
      animation: noiseAnimation 0.5s infinite;
    }
    
    .scan-line {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: linear-gradient(
        to bottom,
        transparent 0%, 
        rgba(0, 195, 255, 0.05) 50%, 
        transparent 100%
      );
      background-size: 100% 4px;
      animation: scan 8s linear infinite;
      pointer-events: none;
      z-index: 2;
    }
    
    .horizontal-scan {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 2px;
      background: rgba(0, 195, 255, 0.05);
      box-shadow: 0 0 10px rgba(0, 195, 255, 0.5);
      animation: horizontalScan 3s linear infinite;
      pointer-events: none;
      z-index: 3;
    }
    
    .screen-glow {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      box-shadow: 
        inset 0 0 50px rgba(0, 195, 255, 0.3),
        inset 0 0 100px rgba(0, 0, 0, 0.8);
      pointer-events: none;
      z-index: 4;
    }
    
    .hologram-lines {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      pointer-events: none;
      background: repeating-linear-gradient(
        0deg,
        rgba(0, 195, 255, 0.03) 0px,
        rgba(0, 195, 255, 0.03) 1px,
        transparent 1px,
        transparent 2px
      );
      z-index: 5;
      opacity: 0.5;
    }
    
    &.scanning::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: linear-gradient(to bottom, 
        transparent 0%, 
        rgba(0, 195, 255, 0.05) 50%, 
        transparent 100%);
      background-size: 100% 4px;
      animation: scan 8s linear infinite;
      pointer-events: none;
      z-index: 10;
    }
    
    &.scanning::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 2px;
      background: rgba(0, 195, 255, 0.15);
      box-shadow: 0 0 10px rgba(0, 195, 255, 0.5);
      animation: horizontalScan 3s linear infinite;
      pointer-events: none;
      z-index: 11;
    }
    
    &.changing-channel::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: linear-gradient(
        135deg,
        rgba(0, 195, 255, 0.2),
        rgba(255, 0, 195, 0.2)
      );
      opacity: 0;
      animation: channel-change 1s ease;
      pointer-events: none;
      z-index: 20;
    }
    
    &.changing-channel {
      animation: channelFlicker 0.5s ease-in-out;
    }
    
    .power-off-screen {
      width: 100%;
      height: 100%;
      background-color: #000;
      display: flex;
      justify-content: center;
      align-items: center;
      position: relative;
      animation: powerOffFade 1s forwards;
      
      &::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(
          ellipse at center,
          rgba(0, 195, 255, 0.1) 0%,
          rgba(0, 0, 0, 0) 70%
        );
        animation: powerOffGlow 2s forwards;
      }
      
      .power-off-dot {
        width: 5px;
        height: 5px;
        background-color: #00c3ff;
        border-radius: 50%;
        box-shadow: 0 0 15px rgba(0, 195, 255, 0.5);
        animation: fade-out 2s forwards;
        position: relative;
        
        &::after {
          content: '';
          position: absolute;
          top: -10px;
          left: -10px;
          right: -10px;
          bottom: -10px;
          border-radius: 50%;
          border: 1px solid rgba(0, 195, 255, 0.3);
          opacity: 0.8;
          animation: dot-pulse 2s forwards;
        }
      }
    }
    
    .screen-content {
      position: relative;
      z-index: 1;
      height: 100%;
      width: 100%;
    }
  }
  
  @keyframes noiseAnimation {
    0% { opacity: 0.08; }
    50% { opacity: 0.06; }
    100% { opacity: 0.08; }
  }
  
  @keyframes scan {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100%); }
  }
  
  @keyframes horizontalScan {
    0% { transform: translateY(-100%); opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { transform: translateY(100vh); opacity: 0; }
  }
  
  @keyframes channelFlicker {
    0% { filter: brightness(1); }
    20% { filter: brightness(1.5); }
    40% { filter: brightness(0.8); }
    60% { filter: brightness(1.2); }
    80% { filter: brightness(0.9); }
    100% { filter: brightness(1); }
  }
  
  @keyframes channel-change {
    0% { opacity: 0; }
    50% { opacity: 0.5; }
    100% { opacity: 0; }
  }
  
  @keyframes powerOffFade {
    0% { background-color: #000022; }
    100% { background-color: #000000; }
  }
  
  @keyframes powerOffGlow {
    0% { opacity: 1; }
    100% { opacity: 0; }
  }
  
  @keyframes fade-out {
    0% { transform: scale(1); opacity: 1; }
    20% { transform: scale(0.8); opacity: 0.8; }
    100% { transform: scale(0); opacity: 0; }
  }
  
  @keyframes dot-pulse {
    0% { transform: scale(1); opacity: 0.8; }
    100% { transform: scale(3); opacity: 0; }
  }
  </style>