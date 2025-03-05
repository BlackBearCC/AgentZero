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
        <button @click="handleExport" class="tv-button primary">
          <span class="button-text">[ 导出角色 ]</span>
        </button>
      </div>
    </div>

    <div class="report-grid">
      <AttributeCard
        v-for="(category, index) in visibleCategories"
        :key="index"
        :title="category.title"
        :attributes="category.data"
        :loading="category.loading"
        @refresh="handleRefresh"
        @update="handleUpdate"
        @aiOptimize="handleAiOptimize"
        @aiGenerate="handleAiGenerate"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import AttributeCard from './cards/AttributeCard.vue';

// 修改 emit 定义，添加 refresh 事件
// 修改 emit 定义，添加 AI 相关事件
const emit = defineEmits(['reset', 'refresh', 'update', 'aiOptimize', 'aiGenerate']);

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

// 获取角色名称
const characterName = computed(() => {
  if (props.character.基础信息 && props.character.基础信息.length > 0) {
    const nameItem = props.character.基础信息.find(item => 
      item.关键词 && item.关键词.some(k => k.includes('姓名') || k.includes('名字'))
    );
    return nameItem ? nameItem.内容.replace(/{{char}}/g, '').trim() : '未命名角色';
  }
  return '未命名角色';
});

// 定义所有类别及其显示标题
const categoryConfig = [
  { key: '基础信息', title: '基础信息' },
  { key: '性格特征', title: '性格特征' },
  { key: '能力特征', title: '能力特征' },
  { key: '兴趣爱好', title: '兴趣爱好' },
  { key: '情感特质', title: '情感特质' },
  { key: '喜好厌恶', title: '喜好与厌恶' },
  { key: '成长经历', title: '成长经历' },
  { key: '价值观念', title: '价值观念' },
  { key: '社交关系', title: '社交关系' },
  { key: '禁忌话题', title: '禁忌话题' },
  { key: '行为模式', title: '行为模式' },
  { key: '隐藏设定', title: '隐藏设定' },
  { key: '目标动机', title: '目标动机' },
  { key: '弱点缺陷', title: '弱点缺陷' },
  { key: '特殊习惯', title: '特殊习惯' },
  { key: '语言风格', title: '语言风格' }
];

// 计算属性，只生成有数据或正在加载的类别
const visibleCategories = computed(() => {
  return categoryConfig
    .map(category => {
      const hasData = props.character[category.key] && props.character[category.key].length > 0;
      const isLoading = props.loadingCategories.includes(category.key);
      
      if (hasData || isLoading) {
        return {
          title: category.title,
          data: props.character[category.key] || [],
          loading: isLoading
        };
      }
      return null;
    })
    .filter(category => category !== null);
});

// 添加导出相关的函数
function exportToCSV(data) {
  // CSV 表头
  let csvContent = "类别,内容,关键词,重要程度\n";
  
  // 遍历所有类别数据
  Object.entries(data).forEach(([category, attributes]) => {
    if (Array.isArray(attributes)) {
      attributes.forEach(attr => {
        const keywords = Array.isArray(attr.关键词) ? attr.关键词.join('|') : '';
        const content = attr.内容.replace(/{{char}}/g, '').replace(/{{user}}/g, '').replace(/"/g, '""');
        csvContent += `"${category}","${content}","${keywords}",${attr.强度}\n`;
      });
    }
  });
  
  // 创建 Blob 并下载
  const blob = new Blob(["\uFEFF" + csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `${characterName.value || '未命名角色'}_角色设定.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function exportToJSON(data) {
  const jsonStr = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `${characterName.value || '未命名角色'}_角色设定.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

// 导出处理函数
function handleExport() {
  // 导出 JSON
  exportToJSON(props.character);
  // 导出 CSV
  exportToCSV(props.character);
}



// 添加刷新处理函数
function handleRefresh(categoryTitle) {
  console.log('AttributeCard 触发刷新:', categoryTitle);
  
  // 找到对应的类别配置
  const category = categoryConfig.find(cat => cat.title === categoryTitle);
  console.log('找到类别配置:', category);
  
  if (category) {
    // 触发父组件的刷新事件
    console.log('触发父组件刷新事件:', category.key);
    emit('refresh', category.key);
  } else {
    console.error('未找到对应的类别配置:', categoryTitle);
  }
}

// 添加更新处理函数
function handleUpdate(categoryTitle, updatedData) {
  console.log('处理卡片更新:', categoryTitle, updatedData);
  
  // 找到对应的类别配置
  const category = categoryConfig.find(cat => cat.title === categoryTitle);
  
  if (category) {
    // 触发父组件的更新事件
    emit('update', category.key, updatedData);
  } else {
    console.error('未找到对应的类别配置:', categoryTitle);
  }
}

// 添加 AI 优化处理函数
function handleAiOptimize(data) {
  emit('aiOptimize', data);
}

// 添加 AI 生成处理函数
function handleAiGenerate(data) {
  console.log('CharacterReport 处理 AI 生成:', data);
  emit('aiGenerate', data);
}
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
  grid-template-columns: repeat(2, 1fr); /* 每行最多2个卡片 */
  gap: 20px;
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
  .report-grid {
    grid-template-columns: 1fr; /* 小屏幕上每行只显示1个卡片 */
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