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
        <div v-if="formattedContent" class="content-wrapper">{{ formattedContent }}</div>
        <span v-else class="placeholder">{{ placeholder }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue';

const props = defineProps({
  content: {
    type: String,
    default: ''
  },
  loading: {
    type: Boolean,
    default: false
  },
  placeholder: {
    type: String,
    default: ''
  },
  typingEffect: {
    type: Boolean,
    default: false
  },
  typingSpeed: {
    type: Number,
    default: 30 // 每个字符的打字速度(ms)
  }
});

const formattedContent = ref('');
const contentRef = ref(null);

// 改进的打字机效果实现
const typeContent = async (text) => {
  if (!props.typingEffect) {
    formattedContent.value = text;
    return;
  }

  // 如果新内容比当前内容短，直接替换避免抖动
  if (text.length < formattedContent.value.length) {
    formattedContent.value = text;
    return;
  }

  // 只添加新字符，避免重新渲染整个内容
  let currentIndex = formattedContent.value.length;
  const targetLength = text.length;
  
  while (currentIndex < targetLength) {
    formattedContent.value = text.slice(0, currentIndex + 1);
    currentIndex++;
    // 使用requestAnimationFrame来优化渲染
    await new Promise(resolve => {
      setTimeout(() => {
        requestAnimationFrame(resolve);
      }, props.typingSpeed);
    });
  }
};

// 监听content变化
watch(() => props.content, async (newContent, oldContent) => {
  if (newContent) {
    // 检查是否只是追加内容，避免重新打印已有内容
    if (oldContent && newContent.startsWith(oldContent)) {
      // 只打印新增部分
      await typeContent(newContent);
    } else {
      // 完全不同的内容，重新打印
      formattedContent.value = '';
      await typeContent(newContent);
    }
    
    // 自动滚动到底部
    if (contentRef.value) {
      contentRef.value.scrollTop = contentRef.value.scrollHeight;
    }
  } else {
    formattedContent.value = '';
  }
}, { immediate: true });
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
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  text-align: left; /* 确保内容左对齐 */
}

.stream-text {
  font-family: 'Courier New', monospace;
  color: #44ff44;
  font-size: 1rem;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
  width: 100%;
  min-height: 1.5em; /* 确保最小高度，防止抖动 */
  text-align: left; /* 确保文本左对齐 */
}

.content-wrapper {
  display: inline-block; /* 改为inline-block以支持文本对齐 */
  white-space: pre-wrap;
  word-break: break-word;
  text-align: left; /* 确保包装器内容左对齐 */
  width: 100%; /* 确保宽度占满 */
}

/* 使用伪元素实现光标，避免影响文本布局 */
.stream-text.typing::after {
  content: '';
  display: inline-block;
  width: 0.6em;
  height: 1em;
  background-color: #44ff44;
  vertical-align: text-bottom;
  margin-left: 2px;
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
</style>
