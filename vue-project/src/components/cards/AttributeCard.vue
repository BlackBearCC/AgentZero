<template>
  <div 
    class="attribute-card" 
    :class="{ 
      'is-generating': loading,
      'mode-compact': displayMode === 'compact',
      'mode-list': displayMode === 'list'
    }"
  >
    <div class="card-header">
      <h3>{{ title }}</h3>
      <div class="header-actions">
        <!-- 只在有数据且不在加载状态时显示刷新按钮 -->
        <button 
          v-if="attributes && attributes.length > 0 && !loading" 
          @click="handleRefresh" 
          class="refresh-button"
        >
          <span class="button-icon">↻</span>
        </button>
        <!-- 移除了这里的状态指示器 -->
      </div>
    </div>
    
    <div class="card-content-wrapper">
      <div class="card-content" v-if="attributes && attributes.length && !loading">
        <div 
          v-for="(attr, index) in attributes" 
          :key="index"
          class="attribute-item"
          :style="{ 
            '--delay': `${index * 0.1}s`,
            '--importance': attr.强度
          }"
        >
          <div class="attribute-header">
            <span class="attribute-title">{{ formatContent(attr.内容) }}</span>
            <div class="importance-indicator" v-if="displayMode !== 'list'">
              <div 
                v-for="n in 5" 
                :key="n"
                class="importance-dot"
                :class="{ active: n <= attr.强度 }"
              ></div>
            </div>
          </div>
          
          <div class="keywords-container" v-if="displayMode !== 'list'">
            <span 
              v-for="(keyword, kidx) in attr.关键词"
              :key="kidx"
              class="keyword-tag"
            >
              {{ keyword }}
            </span>
          </div>
        </div>
      </div>
      
      <div class="card-placeholder" v-else-if="!loading">
        <span>等待生成...</span>
      </div>
      
      <div class="card-placeholder" v-else>
        <span>正在生成...</span>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  title: {
    type: String,
    required: true
  },
  attributes: {
    type: Array,
    default: () => []
  },
  loading: {
    type: Boolean,
    default: false
  },
  displayMode: {
    type: String,
    default: 'default', // 'default', 'compact', 'list'
    validator: (value) => ['default', 'compact', 'list'].includes(value)
  }
});

// 格式化内容，移除占位符
function formatContent(content) {
  if (!content) return '';
  return content.replace(/{{char}}/g, '').replace(/{{user}}/g, '').trim();
}

// 添加 emit 定义
const emit = defineEmits(['refresh']);

// 添加刷新处理函数
function handleRefresh() {
  console.log('刷新按钮被点击，标题:', props.title);
  emit('refresh', props.title);
}
</script>

<style scoped>
.attribute-card {
  background: rgba(0, 0, 0, 0.7);
  border: 1px solid rgba(68, 255, 68, 0.2);
  border-radius: 10px;
  padding: 20px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  height: 40vh; /* 固定高度为视口高度的40% */
  display: flex;
  flex-direction: column;
}

.attribute-card:hover {
  border-color: rgba(68, 255, 68, 0.4);
  box-shadow: 0 0 15px rgba(68, 255, 68, 0.2);
}

.card-content-wrapper {
  flex: 1;
  overflow: hidden;
  position: relative;
}

.card-content {
  height: 100%;
  overflow-y: auto;
  padding-right: 10px; /* 为滚动条留出空间 */
  scrollbar-width: thin;
  scrollbar-color: rgba(68, 255, 68, 0.3) rgba(0, 0, 0, 0.2);
}

/* 自定义滚动条样式 */
.card-content::-webkit-scrollbar {
  width: 6px;
}

.card-content::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 3px;
}

.card-content::-webkit-scrollbar-thumb {
  background: rgba(68, 255, 68, 0.3);
  border-radius: 3px;
}

.card-content::-webkit-scrollbar-thumb:hover {
  background: rgba(68, 255, 68, 0.5);
}

/* 紧凑模式样式 */
.mode-compact {
  padding: 15px;
}

.mode-compact .attribute-item {
  padding: 10px;
  margin-bottom: 10px;
}

.mode-compact .attribute-title {
  font-size: 1rem;
}

.mode-compact .keywords-container {
  margin-top: 5px;
}

.mode-compact .keyword-tag {
  padding: 2px 8px;
  font-size: 0.8rem;
}

/* 列表模式样式 */
.mode-list .attribute-item {
  padding: 8px 12px;
  margin-bottom: 8px;
  background: rgba(68, 255, 68, 0.03);
}

.mode-list .attribute-header {
  margin-bottom: 0;
}

.mode-list .attribute-title {
  font-size: 0.95rem;
  color: #c0c0c0;
}

.mode-list .attribute-item:hover {
  background: rgba(68, 255, 68, 0.08);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  flex-shrink: 0; /* 防止头部被压缩 */
}

.card-header h3 {
  color: #44ff44;
  margin: 0;
  font-size: 1.2rem;
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.3);
}

.attribute-item {
  background: rgba(68, 255, 68, 0.05);
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 15px;
  animation: fadeIn 0.5s ease forwards;
  animation-delay: var(--delay);
  opacity: 0;
}

.attribute-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.attribute-title {
  color: #e0e0e0;
  font-size: 1.1rem;
  flex: 1;
  margin-right: 10px;
}

.importance-indicator {
  display: flex;
  gap: 4px;
  flex-shrink: 0;
}

.importance-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: rgba(68, 255, 68, 0.2);
  transition: all 0.3s ease;
}

.importance-dot.active {
  background: #44ff44;
  box-shadow: 0 0 8px rgba(68, 255, 68, 0.5);
}

.keywords-container {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.keyword-tag {
  background: rgba(68, 255, 68, 0.1);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 12px;
  padding: 4px 10px;
  font-size: 0.9rem;
  color: #44ff44;
  transition: all 0.3s ease;
}

.keyword-tag:hover {
  background: rgba(68, 255, 68, 0.2);
  transform: translateY(-2px);
}

.card-placeholder {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(68, 255, 68, 0.5);
  font-style: italic;
}

@keyframes scanning {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

@keyframes fadeIn {
  from { 
    opacity: 0;
    transform: translateY(10px);
  }
  to { 
    opacity: 1;
    transform: translateY(0);
  }
}

.is-generating {
  position: relative;
}

.is-generating::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    45deg,
    transparent 0%,
    rgba(68, 255, 68, 0.1) 50%,
    transparent 100%
  );
  animation: shine 2s linear infinite;
}

@keyframes shine {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

/* 响应式调整 */
@media (max-width: 768px) {
  .attribute-card {
    padding: 15px;
    height: 50vh; /* 在小屏幕上稍微增加高度 */
  }
  
  .card-header h3 {
    font-size: 1.1rem;
  }
  
  .attribute-title {
    font-size: 1rem;
  }
  
  .keyword-tag {
    font-size: 0.8rem;
    padding: 3px 8px;
  }
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 15px;
}

.refresh-button {
  background: transparent;
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 4px;
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #44ff44;
  cursor: pointer;
  transition: all 0.3s ease;
  padding: 0;
}

.refresh-button:hover {
  background: rgba(68, 255, 68, 0.1);
  border-color: rgba(68, 255, 68, 0.5);
  transform: rotate(180deg);
}

.refresh-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.button-icon {
  font-size: 1.2rem;
  transition: transform 0.3s ease;
}

.refresh-button:hover .button-icon {
  transform: rotate(180deg);
}
</style>