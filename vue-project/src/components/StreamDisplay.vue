<template>
  <div class="stream-display">
    <div class="stream-content" ref="contentRef">
      <div v-if="loading" class="loading-indicator">
        <div class="loading-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
      <div class="stream-text" :class="{ 'typing': typingEffect }">
        <pre v-if="formattedContent">{{ formattedContent }}</pre>
        <span v-else class="placeholder">{{ placeholder }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, computed, onMounted, nextTick } from 'vue';

const props = defineProps({
  // 流式内容
  content: {
    type: String,
    default: ''
  },
  // 是否显示加载状态
  loading: {
    type: Boolean,
    default: false
  },
  // 占位文本
  placeholder: {
    type: String,
    default: '等待数据流...'
  },
  // 是否启用打字机效果
  typingEffect: {
    type: Boolean,
    default: true
  },
  // 打字机效果速度 (ms)
  typingSpeed: {
    type: Number,
    default: 30
  },
  // 是否自动滚动到底部
  autoScroll: {
    type: Boolean,
    default: true
  },
  // 是否尝试格式化JSON
  formatJson: {
    type: Boolean,
    default: true
  }
});

const displayContent = ref('');
const contentRef = ref(null);
const typingTimer = ref(null);
const buffer = ref(''); // 用于存储待打印的内容

// 监听内容变化
watch(() => props.content, (newContent) => {
  if (!newContent) {
    displayContent.value = '';
    buffer.value = '';
    return;
  }

  // 将新内容添加到缓冲区
  const newChunk = newContent.substring(buffer.value.length);
  if (newChunk) {
    buffer.value = newContent;
    if (props.typingEffect) {
      // 如果正在打字，不需要重新开始
      if (!typingTimer.value) {
        typeNextChar();
      }
    } else {
      // 不需要打字效果，直接显示
      displayContent.value = newContent;
      scrollToBottom();
    }
  }
});

// 打字效果 - 逐字打印
const typeNextChar = () => {
  if (displayContent.value.length < buffer.value.length) {
    // 添加下一个字符
    displayContent.value = buffer.value.substring(0, displayContent.value.length + 1);
    scrollToBottom();
    
    // 设置下一个字符的定时器
    typingTimer.value = setTimeout(() => {
      typingTimer.value = null;
      typeNextChar();
    }, props.typingSpeed);
  }
};

// 滚动到底部
const scrollToBottom = () => {
  if (props.autoScroll && contentRef.value) {
    nextTick(() => {
      contentRef.value.scrollTop = contentRef.value.scrollHeight;
    });
  }
};

// 格式化显示内容
const formattedContent = computed(() => {
  if (!displayContent.value) return '';
  
  if (props.formatJson) {
    try {
      const jsonObj = JSON.parse(displayContent.value);
      return JSON.stringify(jsonObj, null, 2);
    } catch (e) {
      return displayContent.value;
    }
  }
  
  return displayContent.value;
});

// 组件卸载时清理
onMounted(() => {
  if (props.content) {
    buffer.value = props.content;
    if (props.typingEffect) {
      typeNextChar();
    } else {
      displayContent.value = props.content;
    }
  }
});
</script>

<style scoped>
.stream-display {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.stream-content {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 5px;
  position: relative;
}

.stream-text {
  font-family: 'Courier New', monospace;
  color: #44ff44;
  font-size: 1rem;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}

.stream-text.typing::after {
  content: '|';
  animation: blink 1s step-end infinite;
}

.placeholder {
  color: rgba(68, 255, 68, 0.5);
  font-style: italic;
}

.loading-indicator {
  position: absolute;
  top: 10px;
  right: 10px;
  background: rgba(0, 0, 0, 0.7);
  border-radius: 4px;
  padding: 5px 10px;
  z-index: 10;
}

.loading-dots {
  display: flex;
  align-items: center;
  justify-content: center;
}

.loading-dots span {
  width: 8px;
  height: 8px;
  margin: 0 3px;
  background-color: #44ff44;
  border-radius: 50%;
  display: inline-block;
  animation: dot-pulse 1.5s infinite ease-in-out;
}

.loading-dots span:nth-child(2) {
  animation-delay: 0.2s;
}

.loading-dots span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

@keyframes dot-pulse {
  0%, 100% { transform: scale(0.8); opacity: 0.5; }
  50% { transform: scale(1.2); opacity: 1; }
}

pre {
  margin: 0;
  font-family: 'Courier New', monospace;
}
</style>
