<template>
    <div 
      :class="['tv-title', size]" 
      :data-text="glitch ? $slots.default()[0].children : ''"
    >
      <slot></slot>
    </div>
  </template>
  
  <script setup>
  defineProps({
    size: {
      type: String,
      default: 'md',
      validator: (value) => ['sm', 'md', 'lg', 'xl'].includes(value)
    },
    glitch: {
      type: Boolean,
      default: false
    }
  })
  </script>
  
  <style lang="scss" scoped>
  .tv-title {
    font-family: 'Blade Runner', 'Eurostile', 'Orbitron', sans-serif;
    color: #00c3ff;
    text-transform: uppercase;
    letter-spacing: 3px;
    text-shadow: 
      0 0 10px rgba(0, 195, 255, 0.8),
      0 0 20px rgba(0, 195, 255, 0.4),
      0 0 30px rgba(0, 195, 255, 0.2);
    position: relative;
    animation: logoGlow 4s infinite;
    font-weight: bold;
    margin-bottom: 20px;
    text-align: center;
    
    &.sm {
      font-size: 1.2rem;
      letter-spacing: 2px;
    }
    
    &.md {
      font-size: 1.5rem;
    }
    
    &.lg {
      font-size: 2rem;
    }
    
    &.xl {
      font-size: 2.5rem;
      letter-spacing: 5px;
    }
    
    &[data-text] {
      position: relative;
      
      &::before,
      &::after {
        content: attr(data-text);
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.2);
      }
      
      &::before {
        left: 2px;
        text-shadow: -2px 0 #ff00c3;
        clip-path: inset(0 0 0 0);
        animation: glitch-1 2s infinite linear alternate-reverse;
      }
      
      &::after {
        left: -2px;
        text-shadow: 2px 0 #00c3ff;
        clip-path: inset(0 0 0 0);
        animation: glitch-2 3s infinite linear alternate-reverse;
      }
    }
  }
  
  @keyframes logoGlow {
    0%, 100% { 
      text-shadow: 
        0 0 10px rgba(0, 195, 255, 0.8),
        0 0 20px rgba(0, 195, 255, 0.4),
        0 0 30px rgba(0, 195, 255, 0.2);
    }
    50% { 
      text-shadow: 
        0 0 15px rgba(0, 195, 255, 0.9),
        0 0 25px rgba(0, 195, 255, 0.5),
        0 0 35px rgba(0, 195, 255, 0.3);
    }
  }
  
  @keyframes titlePulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
  }
  
  @keyframes glitch-1 {
    0% { clip-path: inset(20% 0 30% 0); }
    100% { clip-path: inset(10% 0 40% 0); }
  }
  
  @keyframes glitch-2 {
    0% { clip-path: inset(40% 0 10% 0); }
    100% { clip-path: inset(30% 0 20% 0); }
  }
  </style>