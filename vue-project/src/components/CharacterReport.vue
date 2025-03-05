<template>
  <div class="character-report">
    <div class="report-header">
      <div class="header-left">
        <div class="report-badge">角色档案</div>
        <h2>{{ characterName }}</h2>
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

    <div class="report-content">
      <!-- 基础信息区域 - 特殊处理 -->
      <div class="report-section basic-info-section" v-if="character['基础信息']">
        <h3 class="section-title">基础信息</h3>
        <div class="basic-info-grid">
          <div 
            v-for="(attr, index) in character['基础信息']" 
            :key="index"
            class="basic-info-item"
          >
            <div class="info-label">
              {{ getMainKeyword(attr.关键词) }}
            </div>
            <div class="info-value">
              {{ formatContent(attr.内容) }}
            </div>
          </div>
        </div>
      </div>
      
      <!-- 主要特性区域 - 使用卡片布局 -->
      <div class="report-section main-traits-section">
        <div class="traits-grid">
          <AttributeCard
            v-for="category in mainCategories"
            :key="category.key"
            :title="category.title"
            :attributes="character[category.key] || []"
            :loading="loadingCategories.includes(category.key)"
            display-mode="compact"
            class="trait-card"
          />
        </div>
      </div>
      
      <!-- 次要特性区域 - 使用列表布局 -->
      <div class="report-section secondary-traits-section" v-if="secondaryCategories.length > 0">
        <h3 class="section-title">其他特性</h3>
        <div class="secondary-traits-container">
          <div 
            v-for="category in secondaryCategories" 
            :key="category.key"
            class="secondary-category"
          >
            <AttributeCard
              :title="category.title"
              :attributes="character[category.key] || []"
              :loading="loadingCategories.includes(category.key)"
              display-mode="list"
              class="secondary-card"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import AttributeCard from './cards/AttributeCard.vue';

const props = defineProps({
  character: {
    type: Object,
    required: true
  },
  loadingCategories: {
    type: Array,
    default: () => []
  }
});

defineEmits(['reset', 'export']);

// 获取角色名称
const characterName = computed(() => {
  if (props.character['基础信息'] && props.character['基础信息'].length > 0) {
    const nameItem = props.character['基础信息'].find(item => 
      item.关键词 && item.关键词.some(k => k.includes('姓名') || k.includes('名字'))
    );
    return nameItem ? formatContent(nameItem.内容) : '未命名角色';
  }
  return '未命名角色';
});

// 格式化内容，移除占位符
function formatContent(content) {
  if (!content) return '';
  return content.replace(/{{char}}/g, '').replace(/{{user}}/g, '').trim();
}

// 从关键词数组中获取主要关键词
function getMainKeyword(keywords) {
  if (!keywords || keywords.length === 0) return '未知';
  // 优先返回较短的关键词作为标签
  const sortedKeywords = [...keywords].sort((a, b) => a.length - b.length);
  return sortedKeywords[0];
}

// 定义所有类别及其显示标题
const categoryConfig = [
  { key: '基础信息', title: '基础信息', priority: 0 },
  { key: '性格特征', title: '性格特征', priority: 1 },
  { key: '能力特征', title: '能力特征', priority: 1 },
  { key: '情感特质', title: '情感特质', priority: 1 },
  { key: '兴趣爱好', title: '兴趣爱好', priority: 1 },
  { key: '喜好厌恶', title: '喜好与厌恶', priority: 2 },
  { key: '成长经历', title: '成长经历', priority: 2 },
  { key: '价值观念', title: '价值观念', priority: 2 },
  { key: '社交关系', title: '社交关系', priority: 2 },
  { key: '禁忌话题', title: '禁忌话题', priority: 2 },
  { key: '行为模式', title: '行为模式', priority: 2 },
  { key: '隐藏设定', title: '隐藏设定', priority: 2 },
  { key: '目标动机', title: '目标动机', priority: 2 },
  { key: '弱点缺陷', title: '弱点缺陷', priority: 2 },
  { key: '特殊习惯', title: '特殊习惯', priority: 2 },
  { key: '语言风格', title: '语言风格', priority: 2 }
];

// 获取已生成的类别
const availableCategories = computed(() => {
  return categoryConfig.filter(category => 
    props.character[category.key] && 
    props.character[category.key].length > 0
  );
});

// 主要特性类别（优先级1）
const mainCategories = computed(() => {
  return availableCategories.value.filter(category => category.priority === 1);
});

// 次要特性类别（优先级2）
const secondaryCategories = computed(() => {
  return availableCategories.value.filter(category => category.priority === 2);
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
  display: flex;
  flex-direction: column;
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

.report-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 30px;
}

.report-section {
  background: rgba(0, 0, 0, 0.4);
  border: 1px solid rgba(68, 255, 68, 0.2);
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 0 15px rgba(0, 0, 0, 0.3);
}

.section-title {
  color: #44ff44;
  margin: 0 0 20px 0;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(68, 255, 68, 0.2);
  font-size: 1.3rem;
  text-shadow: 0 0 8px rgba(68, 255, 68, 0.3);
}

/* 基础信息区域样式 */
.basic-info-section {
  background: rgba(0, 0, 0, 0.5);
  border-color: rgba(68, 255, 68, 0.3);
}

.basic-info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 15px;
}

.basic-info-item {
  background: rgba(68, 255, 68, 0.05);
  border-radius: 8px;
  padding: 12px;
  transition: all 0.3s ease;
}

.basic-info-item:hover {
  background: rgba(68, 255, 68, 0.1);
  transform: translateY(-2px);
}

.info-label {
  color: #a0a0a0;
  font-size: 0.9rem;
  margin-bottom: 5px;
}

.info-value {
  color: #e0e0e0;
  font-size: 1.1rem;
}

/* 主要特性区域样式 */
.traits-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}

.trait-card {
  height: 100%;
  transition: all 0.3s ease;
}

.trait-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 15px rgba(68, 255, 68, 0.1);
}

/* 次要特性区域样式 */
.secondary-traits-container {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
}

.secondary-category {
  flex: 1 1 300px;
  min-width: 0;
}

.secondary-card {
  height: 100%;
  background: rgba(0, 0, 0, 0.3);
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

/* 响应式调整 */
@media (max-width: 768px) {
  .traits-grid {
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