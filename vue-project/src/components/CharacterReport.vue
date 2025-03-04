<template>
  <div class="character-report">
    <div class="report-header">
      <div class="header-left">
        <div class="report-badge">角色档案</div>
        <h2>{{ character.name || '未命名角色' }}</h2>
      </div>
      <div class="header-actions">
        <button @click="$emit('reset')" class="tv-button">
          <span class="button-text">[ 重新生成 ]</span>
        </button>
        <button @click="$emit('export')" class="tv-button primary">
          <span class="button-text">[ 导出角色 ]</span>
        </button>
      </div>
    </div>

    <div class="report-grid">
      <!-- 基础信息卡片 -->
      <div class="report-card basic-info">
        <h3>基础信息</h3>
        <div class="info-grid">
          <div v-for="(info, index) in basicInfo" :key="index" class="info-item">
            <div class="info-label">{{ info.label }}</div>
            <div class="info-value">{{ info.value }}</div>
          </div>
        </div>
      </div>

      <!-- 性格特质卡片 -->
      <div class="report-card traits">
        <h3>性格特质</h3>
        <div class="traits-list">
          <div v-for="trait in character.traits" :key="trait.name" class="trait-item">
            <div class="trait-header">
              <span class="trait-name">{{ trait.name }}</span>
              <span class="trait-value">{{ trait.value }}%</span>
            </div>
            <div class="trait-bar-container">
              <div class="trait-bar" :style="{ width: `${trait.value}%` }">
                <div class="trait-glow"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 关键词卡片 -->
      <div class="report-card keywords">
        <h3>关键特征</h3>
        <div class="keywords-cloud">
          <span 
            v-for="(keyword, index) in character.keywords" 
            :key="index" 
            class="keyword-tag"
            :style="{ 
              fontSize: `${1 + Math.random() * 0.5}rem`,
              opacity: 0.7 + Math.random() * 0.3
            }"
          >
            {{ keyword }}
          </span>
        </div>
      </div>

      <!-- 背景故事卡片 -->
      <div class="report-card background">
        <h3>背景故事</h3>
        <div class="background-content">
          <p v-for="(paragraph, index) in backgroundParagraphs" :key="index">
            {{ paragraph }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  character: {
    type: Object,
    required: true
  }
});

defineEmits(['reset', 'export']);

// 处理背景故事分段
const backgroundParagraphs = computed(() => {
  if (!props.character.background) return [];
  return props.character.background.split('\n').filter(p => p.trim());
});

// 处理基础信息
const basicInfo = computed(() => {
  const { name, identity, age } = props.character;
  return [
    { label: '姓名', value: name || '未知' },
    { label: '身份', value: identity || '未知' },
    { label: '年龄', value: age || '未知' }
  ];
});
</script>

<style scoped>
.character-report {
  height: 100%;
  padding: 20px;
  background: rgba(0, 0, 0, 0.8);
  border-radius: 15px;
  border: 2px solid rgba(68, 255, 68, 0.3);
  overflow-y: auto;
}

.report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 25px;
  padding-bottom: 15px;
  border-bottom: 1px solid rgba(68, 255, 68, 0.3);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 15px;
}

.report-badge {
  background: rgba(68, 255, 68, 0.2);
  color: #44ff44;
  padding: 5px 10px;
  border-radius: 4px;
  font-size: 0.9rem;
  border: 1px solid rgba(68, 255, 68, 0.5);
  text-transform: uppercase;
  letter-spacing: 1px;
}

.report-header h2 {
  color: #44ff44;
  margin: 0;
  font-size: 1.8rem;
  text-shadow: 0 0 10px rgba(68, 255, 68, 0.5);
}

.report-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}

.report-card {
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(68, 255, 68, 0.2);
  border-radius: 10px;
  padding: 20px;
  transition: all 0.3s ease;
}

.report-card:hover {
  border-color: rgba(68, 255, 68, 0.4);
  box-shadow: 0 0 15px rgba(68, 255, 68, 0.2);
}

.report-card h3 {
  color: #44ff44;
  margin: 0 0 15px 0;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(68, 255, 68, 0.2);
  font-size: 1.2rem;
}

/* 基础信息样式 */
.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 15px;
}

.info-item {
  padding: 10px;
  background: rgba(68, 255, 68, 0.1);
  border-radius: 5px;
}

.info-label {
  color: #a0a0a0;
  font-size: 0.9rem;
  margin-bottom: 5px;
}

.info-value {
  color: #44ff44;
  font-size: 1.1rem;
}

/* 性格特质样式 */
.traits-list {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.trait-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.trait-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.trait-name {
  color: #e0e0e0;
  font-size: 1rem;
}

.trait-value {
  color: #44ff44;
  font-size: 0.9rem;
}

.trait-bar-container {
  height: 8px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 4px;
  overflow: hidden;
}

.trait-bar {
  height: 100%;
  background: linear-gradient(90deg, rgba(68, 255, 68, 0.3), #44ff44);
  border-radius: 4px;
  position: relative;
  transition: width 1s ease-out;
}

.trait-glow {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(68, 255, 68, 0.5),
    transparent
  );
  animation: glow 2s linear infinite;
}

/* 关键词样式 */
.keywords-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  padding: 10px;
}

.keyword-tag {
  background: rgba(68, 255, 68, 0.15);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 15px;
  padding: 5px 12px;
  transition: all 0.3s ease;
  cursor: default;
}

.keyword-tag:hover {
  background: rgba(68, 255, 68, 0.25);
  transform: translateY(-2px);
  box-shadow: 0 2px 8px rgba(68, 255, 68, 0.2);
}

/* 背景故事样式 */
.background {
  grid-column: 1 / -1;
}

.background-content {
  line-height: 1.6;
  color: #d0d0d0;
}

.background-content p {
  margin-bottom: 15px;
  text-indent: 2em;
}

/* 按钮样式 */
.tv-button {
  background: rgba(40, 40, 60, 0.8);
  border: 1px solid rgba(68, 255, 68, 0.5);
  border-radius: 5px;
  padding: 8px 20px;
  color: #44ff44;
  cursor: pointer;
  transition: all 0.3s ease;
}

.tv-button:hover {
  background: rgba(60, 60, 80, 0.8);
  box-shadow: 0 0 10px rgba(68, 255, 68, 0.5);
}

.tv-button.primary {
  background: rgba(68, 255, 68, 0.3);
}

/* 动画效果 */
@keyframes glow {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

/* 响应式调整 */
@media (max-width: 768px) {
  .report-grid {
    grid-template-columns: 1fr;
  }
  
  .report-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 15px;
  }
  
  .header-actions {
    width: 100%;
    display: flex;
    justify-content: space-between;
  }
}
</style>