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
        <!-- 编辑按钮 -->
        <button 
          v-if="attributes && attributes.length > 0 && !loading" 
          @click="toggleEditing" 
          class="edit-button"
          :class="{ 'active': isEditing }"
        >
          <span class="button-icon">✎</span>
        </button>
        
        <!-- 刷新按钮 -->
        <button 
          v-if="attributes && attributes.length > 0 && !loading" 
          @click="handleRefresh" 
          class="refresh-button"
        >
          <span class="button-icon">↻</span>
        </button>
      </div>
    </div>
    
    <div class="card-content-wrapper">
      <!-- 非编辑模式 -->
      <div class="card-content" v-if="attributes && attributes.length && !loading && !isEditing">
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
      
      <!-- 编辑模式 -->
      <div class="card-content edit-mode" v-else-if="isEditing">
        <div class="edit-actions top">
          <button @click="addNewAttribute" class="add-button">
            <span>+ 添加新属性</span>
          </button>
          <button @click="aiAddNewAttribute" class="add-button ai">
            <span>🤖 AI 生成新属性</span>
          </button>
        </div>
        
        <div 
          v-for="(attr, index) in editingAttributes" 
          :key="index"
          class="attribute-item editing"
        >
          <div class="edit-item-header">
            <div class="edit-content-wrapper">
              <textarea 
                v-model="attr.内容" 
                class="edit-content"
                placeholder="输入内容..."
                rows="2"
              ></textarea>
              <button 
                @click="aiOptimizeAttribute(index)" 
                class="ai-optimize-button"
                :disabled="attr.isOptimizing"
              >
                <span class="button-icon">🤖</span>
                <span class="button-text">AI 优化</span>
              </button>
            </div>
            
            <div class="edit-importance">
              <span class="importance-label">重要程度:</span>
              <div class="importance-selector">
                <div 
                  v-for="n in 5" 
                  :key="n"
                  class="importance-dot selectable"
                  :class="{ active: n <= attr.强度 }"
                  @click="attr.强度 = n"
                ></div>
              </div>
            </div>
          </div>
          
          <div class="edit-keywords">
            <div class="keywords-header">
              <span>关键词:</span>
              <button @click="addKeyword(index)" class="small-button">+</button>
            </div>
            
            <div class="keywords-list">
              <div 
                v-for="(keyword, kidx) in attr.关键词" 
                :key="kidx"
                class="keyword-edit-item"
              >
                <input 
                  v-model="attr.关键词[kidx]" 
                  class="keyword-input"
                  placeholder="关键词..."
                />
                <button @click="removeKeyword(index, kidx)" class="small-button remove">×</button>
              </div>
            </div>
          </div>
          
          <div class="edit-item-footer">
            <button @click="removeAttribute(index)" class="remove-button">删除</button>
          </div>
        </div>
        
        <div class="edit-actions bottom">
          <button @click="saveChanges" class="save-button">保存更改</button>
          <button @click="cancelEditing" class="cancel-button">取消</button>
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
import { ref, computed, watch } from 'vue';
import { ElMessageBox } from 'element-plus';

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

// 添加 emit 定义
const emit = defineEmits(['refresh', 'update', 'aiOptimize', 'aiGenerate']);

// 编辑状态
const isEditing = ref(false);
const editingAttributes = ref([]);

// 监听属性变化，更新编辑状态
watch(() => props.attributes, (newAttributes) => {
  if (isEditing.value) {
    // 如果正在编辑，更新编辑中的属性
    editingAttributes.value = JSON.parse(JSON.stringify(newAttributes));
  }
}, { deep: true });

// 格式化内容，移除占位符
function formatContent(content) {
  if (!content) return '';
  return content.replace(/{{char}}/g, '').replace(/{{user}}/g, '').trim();
}

// 添加刷新处理函数
function handleRefresh() {
  console.log('刷新按钮被点击，标题:', props.title);
  emit('refresh', props.title);
}

// 切换编辑模式
async function toggleEditing() {
  if (isEditing.value) {
    // 如果已经在编辑模式，询问是否保存
    try {
      await ElMessageBox.confirm(
        '是否保存当前的修改？',
        '提示',
        {
          confirmButtonText: '保存',
          cancelButtonText: '不保存',
          type: 'warning',
          distinguishCancelAndClose: true,
          showClose: true,
        }
      );
      // 用户点击保存
      await saveChanges();
    } catch (action) {
      if (action === 'cancel') {
        // 用户点击不保存
        cancelEditing();
      }
      // 用户点击关闭按钮，保持编辑状态
      return;
    }
  } else {
    // 进入编辑模式，复制一份数据进行编辑
    editingAttributes.value = JSON.parse(JSON.stringify(props.attributes));
  }
  isEditing.value = !isEditing.value;
}

// AI 优化属性
async function aiOptimizeAttribute(index) {
  const attr = editingAttributes.value[index];
  attr.isOptimizing = true;
  
  try {
    // 向父组件发送优化请求
    emit('aiOptimize', {
      category: props.title,
      index,
      attribute: attr
    });
    
    // 注意：实际的优化逻辑在父组件中处理
    // 这里只需要发送事件
  } catch (error) {
    console.error('AI 优化失败:', error);
  } finally {
    attr.isOptimizing = false;
  }
}

// AI 添加新属性
async function aiAddNewAttribute() {
  // 向父组件发送生成请求
  emit('aiGenerate', {
    category: props.title,
    existingAttributes: editingAttributes.value
  });
  
  // 注意：实际的生成逻辑在父组件中处理
}

// 处理取消编辑
async function handleCancelEditing() {
  if (hasChanges.value) {
    try {
      await ElMessageBox.confirm(
        '确定要取消编辑？未保存的修改将会丢失。',
        '警告',
        {
          confirmButtonText: '确定',
          cancelButtonText: '返回编辑',
          type: 'warning'
        }
      );
      cancelEditing();
    } catch {
      // 用户取消操作，继续编辑
    }
  } else {
    cancelEditing();
  }
}

// 检查是否有未保存的更改
const hasChanges = computed(() => {
  if (!isEditing.value) return false;
  return JSON.stringify(editingAttributes.value) !== JSON.stringify(props.attributes);
});

// 添加新属性
function addNewAttribute() {
  editingAttributes.value.push({
    内容: '',
    关键词: [''],
    强度: 3
  });
}

// 删除属性
function removeAttribute(index) {
  editingAttributes.value.splice(index, 1);
}

// 添加关键词
function addKeyword(attrIndex) {
  editingAttributes.value[attrIndex].关键词.push('');
}

// 删除关键词
function removeKeyword(attrIndex, keywordIndex) {
  editingAttributes.value[attrIndex].关键词.splice(keywordIndex, 1);
}

// 保存更改
function saveChanges() {
  // 过滤掉空内容的属性
  const validAttributes = editingAttributes.value.filter(attr => attr.内容.trim() !== '');
  
  // 过滤每个属性中的空关键词
  validAttributes.forEach(attr => {
    attr.关键词 = attr.关键词.filter(k => k.trim() !== '');
    if (attr.关键词.length === 0) {
      attr.关键词 = [''];
    }
  });
  
  // 发送更新事件
  emit('update', props.title, validAttributes);
  isEditing.value = false;
}

// 取消编辑
function cancelEditing() {
  isEditing.value = false;
  editingAttributes.value = [];
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

/* 添加编辑相关样式 */
.edit-button {
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
  margin-right: 8px;
}

.edit-button:hover, .edit-button.active {
  background: rgba(68, 255, 68, 0.1);
  border-color: rgba(68, 255, 68, 0.5);
}

.edit-button.active {
  background: rgba(68, 255, 68, 0.2);
}

.is-editing {
  border-color: rgba(68, 255, 68, 0.5);
}

.edit-mode {
  height: 100%;
  overflow-y: auto;
  padding-right: 10px;
}

.attribute-item.editing {
  background: rgba(68, 255, 68, 0.08);
  opacity: 1;
  animation: none;
  padding: 15px;
  margin-bottom: 15px;
}

.edit-item-header {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 15px;
}

.edit-content {
  width: 100%;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 5px;
  color: #e0e0e0;
  padding: 8px;
  resize: vertical;
  font-family: inherit;
  font-size: 1rem;
}

.edit-importance {
  display: flex;
  align-items: center;
  gap: 10px;
}

.importance-label {
  color: #a0a0a0;
  font-size: 0.9rem;
}

.importance-selector {
  display: flex;
  gap: 5px;
}

.importance-dot.selectable {
  cursor: pointer;
  width: 12px;
  height: 12px;
}

.importance-dot.selectable:hover {
  background: rgba(68, 255, 68, 0.4);
}

.edit-keywords {
  margin-bottom: 15px;
}

.keywords-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  color: #a0a0a0;
  font-size: 0.9rem;
}

.keywords-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.keyword-edit-item {
  display: flex;
  gap: 8px;
}

.keyword-input {
  flex: 1;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 5px;
  color: #e0e0e0;
  padding: 5px 8px;
  font-size: 0.9rem;
}

.small-button {
  background: rgba(68, 255, 68, 0.1);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 4px;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #44ff44;
  cursor: pointer;
  font-size: 1rem;
  padding: 0;
}

.small-button:hover {
  background: rgba(68, 255, 68, 0.2);
}

.small-button.remove {
  color: #ff4444;
  border-color: rgba(255, 68, 68, 0.3);
  background: rgba(255, 68, 68, 0.1);
}

.small-button.remove:hover {
  background: rgba(255, 68, 68, 0.2);
}

.edit-item-footer {
  display: flex;
  justify-content: flex-end;
}

.remove-button {
  background: rgba(255, 68, 68, 0.1);
  border: 1px solid rgba(255, 68, 68, 0.3);
  border-radius: 4px;
  color: #ff4444;
  padding: 5px 10px;
  cursor: pointer;
  font-size: 0.9rem;
}

.remove-button:hover {
  background: rgba(255, 68, 68, 0.2);
}

.edit-actions {
  display: flex;
  justify-content: center;
  margin-bottom: 15px;
}

.edit-actions.bottom {
  margin-top: 20px;
  margin-bottom: 0;
  gap: 15px;
}

.add-button {
  background: rgba(68, 255, 68, 0.1);
  border: 1px solid rgba(68, 255, 68, 0.3);
  border-radius: 4px;
  color: #44ff44;
  padding: 8px 15px;
  cursor: pointer;
  font-size: 0.9rem;
}

.add-button:hover {
  background: rgba(68, 255, 68, 0.2);
}

.save-button {
  background: rgba(68, 255, 68, 0.2);
  border: 1px solid rgba(68, 255, 68, 0.4);
  border-radius: 4px;
  color: #44ff44;
  padding: 8px 20px;
  cursor: pointer;
  font-size: 1rem;
}

.save-button:hover {
  background: rgba(68, 255, 68, 0.3);
}

.cancel-button {
  background: rgba(150, 150, 150, 0.1);
  border: 1px solid rgba(150, 150, 150, 0.3);
  border-radius: 4px;
  color: #c0c0c0;
  padding: 8px 20px;
  cursor: pointer;
  font-size: 1rem;
}

.cancel-button:hover {
  background: rgba(150, 150, 150, 0.2);
}

.edit-content-wrapper {
  position: relative;
  width: 100%;
}

.ai-optimize-button {
  position: absolute;
  right: 8px;
  bottom: 8px;
  background: rgba(68, 68, 255, 0.1);
  border: 1px solid rgba(68, 68, 255, 0.3);
  border-radius: 4px;
  color: #4444ff;
  padding: 4px 8px;
  cursor: pointer;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  gap: 4px;
  transition: all 0.3s ease;
}

.ai-optimize-button:hover {
  background: rgba(68, 68, 255, 0.2);
}

.ai-optimize-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.add-button.ai {
  background: rgba(68, 68, 255, 0.1);
  border-color: rgba(68, 68, 255, 0.3);
  color: #4444ff;
}

.add-button.ai:hover {
  background: rgba(68, 68, 255, 0.2);
}

.edit-actions.top {
  display: flex;
  gap: 10px;
  margin-bottom: 15px;
}

.button-icon {
  font-size: 1.1rem;
}
</style>