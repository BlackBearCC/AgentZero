<template>
  <!-- <TvScreen class="character-generator" :scanning="isGenerating"> -->
    <!-- 未上传文件时的空状态 -->
    <div v-if="!hasGeneratedData && !isGenerating" class="empty-state">
      <TvTitle size="xl">CHARACTER GENERATOR</TvTitle>
      <div class="channel-info">频道 1</div>
      <div class="instruction-text">请上传角色资料文件开始生成</div>
      <!-- 控制区域整合到屏幕中 -->
      <TvPanel>
        <!-- <TvControlGroup label="文件控制"> -->
          <div class="file-control">
            <TvFileInput
              id="character-file"
              accept=".txt,.pdf,.docx,.csv,.json"
              @file-change="handleFileChange"
            >
              [ 选择角色文件 ]
            </TvFileInput>
          </div>

          <div class="import-control">
            <TvFileInput
              id="import-character"
              accept=".csv,.json"
              @file-change="handleImportFile"
            >
              [ 导入角色 ]
            </TvFileInput>
          </div>
        <!-- </TvControlGroup> -->

        <!-- <TvControlGroup label="生成选项"> -->
          <div class="option-group">
            <TvCheckbox 
              v-model="showGenerationProcess"
              id="show-generation-process"
            >
              显示生成过程
            </TvCheckbox>
          </div>
        <!-- </TvControlGroup> -->
        
        <!-- 类别选择 -->
        <!-- <TvControlGroup v-if="batchGenerate" label="选择要生成的属性类别"> -->
          <!-- 类别选择 -->
          <TvCheckboxGroup
            v-model="selectedCategories"
            :options="categoryOptionsFormatted"
            selectAllLabel="全选所有类别"
            id="category-selection"
            class="grid-layout"
            :descriptions="categoryDescriptions"
          />
        <!-- </TvControlGroup> -->
      
        <!-- 操作按钮 -->
        <div class="action-buttons">
          <TvButton 
            primary 
            @click="generateCharacter" 
            :disabled="!canGenerate"
          >
            [ 开始生成 ]
          </TvButton>
        </div>
      </TvPanel>
    </div>
    
    <!-- 生成中状态 -->
    <div v-else-if="isGenerating && showGenerationProcess" class="processing-state">
      <TvTitle size="xl">CHARACTER GENERATOR</TvTitle>
      <TvScreen scanning>
        <StreamDisplay 
          :content="streamContent"
          :loading="isGenerating"
          placeholder="正在生成角色配置..."
          :typing-effect="true"
          :typing-speed="3"
        />
      </TvScreen>
    </div>
    
    <!-- 生成结果 -->
    <div v-else class="result-container">
      <CharacterReport 
        :character="characterData"
        :loadingCategories="loadingCategories"
        @reset="resetGenerator"
        @refresh="refreshCategory"
        @update="updateCategory"
        @aiOptimizeContent="handleAiOptimizeContent"
        @aiOptimizeKeywords="handleAiOptimizeKeywords"
        @aiGenerate="handleAiGenerate"
      />
    </div>
  <!-- </TvScreen> -->
</template>

<script setup>
import { ref, computed, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import StreamDisplay from './StreamDisplay.vue'
import CharacterReport from './CharacterReport.vue'
import {
  TvButton,
  TvPanel,
  TvScreen,
  TvTitle,
  TvCheckbox,
  TvCheckboxGroup,
  TvFileInput,
} from '../design-system/components'

// 修改 emit 定义，确保包含所有需要的事件
const emit = defineEmits(['reset', 'refresh', 'update', 'aiOptimizeContent', 'aiOptimizeKeywords', 'aiGenerate'])

// 文件状态
const fileName = ref('')
const selectedFile = ref(null)

// 生成选项
const showGenerationProcess = ref(true)
const selectedCategories = ref([
  '基础信息', '性格特征', '能力特征', '兴趣爱好', '情感特质', 
  '喜好厌恶', '成长经历', '价值观念', '社交关系', '禁忌话题', 
  '行为模式', '隐藏设定', '目标动机', '弱点缺陷', '特殊习惯', 
  '语言风格'
])

// 类别配置
const categoryOptions = [
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
]

// 生成状态
const isGenerating = ref(false)
const characterData = reactive({})
const streamContent = ref('')
const loadingCategories = ref([])

// 计算属性
const canGenerate = computed(() => {
  return selectedFile.value && !isGenerating.value && selectedCategories.value.length > 0
})

const hasGeneratedData = computed(() => {
  return Object.keys(characterData).length > 0
})

// 格式化类别选项以适配 TvCheckboxGroup
const categoryOptionsFormatted = computed(() => {
  return categoryOptions.map(category => ({
    label: category.title,
    value: category.key
  }))
})

// 为每个类别添加描述信息
const categoryDescriptions = {
  '基础信息': '角色的基本资料，如姓名、年龄、性别等基础属性',
  '性格特征': '角色的性格特点和心理特征',
  '能力特征': '角色的技能、专长和特殊能力',
  '兴趣爱好': '角色喜欢从事的活动和爱好',
  '情感特质': '角色的情感表达方式和情绪特点',
  '喜好厌恶': '角色喜欢和讨厌的事物',
  '成长经历': '角色的成长过程和重要经历',
  '价值观念': '角色的信仰、价值观和人生态度',
  '社交关系': '角色与他人的关系和社交模式',
  '禁忌话题': '角色不愿提及或敏感的话题',
  '行为模式': '角色在不同情境下的行为习惯',
  '隐藏设定': '角色不为人知的秘密或隐藏特质',
  '目标动机': '角色的目标和行动动机',
  '弱点缺陷': '角色的弱点、缺陷和不足',
  '特殊习惯': '角色独特的习惯和小癖好',
  '语言风格': '角色的说话方式和语言特点'
}

// 处理文件选择
const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (file) {
    selectedFile.value = file
    fileName.value = file.name
  } else {
    selectedFile.value = null
    fileName.value = ''
  }
}

// 生成角色
const generateCharacter = async () => {
  if (!canGenerate.value) return
  
  isGenerating.value = true
  streamContent.value = ''  // 重置流内容
  
  
  try {
    const fileContent = await selectedFile.value.text()
    
    // 构建请求体
    const requestBody = {
      reference: fileContent,
      categories: selectedCategories.value

    }
    

    
    const response = await fetch('/api/v1/generate_role_config/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify(requestBody)
    })
    
    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      
      buffer += decoder.decode(value, { stream: true })
      
      // 处理SSE事件
      let eventEndIndex
      while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
        const event = buffer.slice(0, eventEndIndex)
        buffer = buffer.slice(eventEndIndex + 2)
        
        if (event.startsWith('data: ')) {
          const data = JSON.parse(event.slice(6).trim())
          
          // 根据事件类型处理数据
          switch (data.type) {
            case 'start':
              if (data.category) {
                // 添加到正在加载的类别
                if (!loadingCategories.value.includes(data.category)) {
                  loadingCategories.value.push(data.category)
                }
              }
              break
            case 'chunk':
              streamContent.value += data.content
              break
            case 'complete':
              if (data.category && data.content) {
                // 更新特定类别的数据
                characterData[data.category] = data.content
                
                // 从加载列表中移除
                const index = loadingCategories.value.indexOf(data.category)
                if (index !== -1) {
                  loadingCategories.value.splice(index, 1)
                }
              } else if (!batchGenerate.value && data.content) {
                // 处理完整数据（非批量模式）
                Object.keys(data.content).forEach(key => {
                  characterData[key] = data.content[key]
                })
              }
              break
            case 'error':
              console.error('生成错误:', data.content)
              ElMessage.error(data.content)
              
              // 如果是特定类别的错误，从加载列表中移除
              if (data.category) {
                const index = loadingCategories.value.indexOf(data.category)
                if (index !== -1) {
                  loadingCategories.value.splice(index, 1)
                }
              }
              break
            case 'end':
              // 如果是特定类别的结束，不做特殊处理
              break
          }
        }
      }
    }
  } catch (e) {
    console.error('生成失败:', e)
    ElMessage.error('生成过程出现错误，请重试')
    loadingCategories.value = [] // 清空加载状态
  } finally {
    // 只有在没有正在加载的类别时才完全结束生成状态
    if (loadingCategories.value.length === 0) {
      isGenerating.value = false
    }
  }
}

// 导出角色
const exportCharacter = () => {
  if (Object.keys(characterData).length === 0) return
  
  const charData = JSON.stringify(characterData, null, 2)
  const blob = new Blob([charData], { type: 'application/json' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  
  // 尝试从基础信息中获取角色名称
  let fileName = '未命名角色'
  if (characterData.基础信息 && characterData.基础信息.length > 0) {
    const nameItem = characterData.基础信息.find(item => 
      item.关键词 && item.关键词.some(k => k.includes('姓名') || k.includes('名字'))
    )
    if (nameItem && nameItem.内容) {
      fileName = nameItem.内容.replace(/[（(].*[)）]/, '').trim()
    }
  }
  
  link.download = `${fileName.replace(/\s+/g, '_')}.json`
  link.click()
  URL.revokeObjectURL(link.href)
}

// 重置生成器
const resetGenerator = () => {
  // 清空所有数据
  Object.keys(characterData).forEach(key => {
    delete characterData[key]
  })
  selectedFile.value = null
  fileName.value = ''
  isGenerating.value = false
  loadingCategories.value = []
}

// 添加单个类别刷新函数
async function refreshCategory(categoryKey) {
  console.log('刷新类别被触发:', categoryKey);
  
  if (!selectedFile.value) {
    console.error('没有选择文件');
    ElMessage.warning('没有可用的角色资料文件');
    return;
  }
  
  // 检查该类别是否已经在加载中
  if (loadingCategories.value.includes(categoryKey)) {
    console.warn('该类别正在生成中');
    ElMessage.warning('该类别正在生成中，请稍后再试');
    return;
  }
  
  // 将类别添加到加载状态
  loadingCategories.value.push(categoryKey);
  console.log('加载状态更新:', loadingCategories.value);
  
  // 清除该类别的现有数据
  if (characterData[categoryKey]) {
    characterData[categoryKey] = [];
  }
  
  try {
    console.log('开始读取文件');
    const fileContent = await selectedFile.value.text();
    
    // 构建请求体
    const requestBody = {
      reference: fileContent,
      categories: [categoryKey] // 只包含要刷新的类别
    };
    console.log('请求体:', requestBody);
    
    console.log('发送请求');
    const response = await fetchWithTimeout('/api/v1/generate_role_config/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify(requestBody)
    });
    
    if (!response.ok) {
      console.error('请求失败:', response.status, response.statusText);
      throw new Error(`请求失败: ${response.status} ${response.statusText}`);
    }
    
    console.log('开始处理响应流');
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log('响应流读取完成');
        break;
      }
      
      const chunk = decoder.decode(value, { stream: true });
      console.log('收到数据块:', chunk);
      buffer += chunk;
      
      // 处理SSE事件
      let eventEndIndex;
      while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
        const event = buffer.slice(0, eventEndIndex);
        buffer = buffer.slice(eventEndIndex + 2);
        
        if (event.startsWith('data: ')) {
          try {
            const data = JSON.parse(event.slice(6).trim());
            console.log('解析的事件数据:', data);
            
            switch (data.type) {
              case 'complete':
                if (data.category && data.content) {
                  console.log(`更新类别 ${data.category} 的数据`);
                  characterData[data.category] = data.content;
                  ElMessage.success(`${data.category} 刷新成功`);
                }
                break;
              case 'error':
                console.error('生成错误:', data.content);
                ElMessage.error(data.content);
                break;
            }
          } catch (e) {
            console.error('解析事件数据失败:', e, event.slice(6).trim());
          }
        }
      }
    }
  } catch (e) {
    console.error('刷新失败:', e);
    ElMessage.error(`刷新过程出现错误: ${e.message}`);
  } finally {
    // 从加载状态中移除该类别
    const index = loadingCategories.value.indexOf(categoryKey);
    if (index !== -1) {
      loadingCategories.value.splice(index, 1);
    }
    console.log('刷新完成，加载状态更新:', loadingCategories.value);
  }
}
// 添加更新类别数据的函数
function updateCategory(categoryKey, updatedData) {
  console.log('更新类别数据:', categoryKey, updatedData);
  
  // 直接更新对应类别的数据
  characterData[categoryKey] = updatedData;
  
  // 可以选择是否要保存到本地存储
  try {
    const savedData = JSON.parse(localStorage.getItem('characterData') || '{}');
    savedData[categoryKey] = updatedData;
    localStorage.setItem('characterData', JSON.stringify(savedData));
  } catch (e) {
    console.error('保存到本地存储失败:', e);
  }
  
  // 显示更新成功提示
  ElMessage.success(`${categoryKey} 更新成功`);
}

// 处理 AI 生成请求
async function handleAiGenerate({ category, existingAttributes }) {
  loadingCategories.value.push(category);
  
  try {
    // 找到对应的类别配置
    const categoryConfig = categoryOptions.find(cat => cat.title === category);
    if (!categoryConfig) {
      throw new Error(`未找到类别: ${category}`);
    }
    
    console.log('发送生成请求:', {
      category: categoryConfig.key,
      existingAttributes: existingAttributes
    });
    
    const response = await fetchWithTimeout('/api/v1/generate_new_attribute', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        category: categoryConfig.key,
        existingAttributes: existingAttributes,
        reference: await selectedFile.value.text() // 添加参考资料
      })
    });
    
    if (!response.ok) {
      throw new Error(`生成请求失败: ${response.status} ${response.statusText}`);
    }
    
    const result = await response.json();
    console.log('生成结果:', result);
    
    // 在数组开头添加新生成的属性 - 直接使用返回的中文键名
    if (!characterData[categoryConfig.key]) {
      characterData[categoryConfig.key] = [];
    }
    
    // 使用服务端返回的数据结构（中文键名）
    characterData[categoryConfig.key].unshift({
      内容: result.内容,
      关键词: result.关键词,
      强度: result.强度
    });
    
    // 触发更新事件，通知 AttributeCard 组件更新编辑状态
    emit('update', categoryConfig.key, characterData[categoryConfig.key]);
    
    ElMessage.success('AI 生成完成');
  } catch (error) {
    console.error('AI 生成失败:', error);
    ElMessage.error(`AI 生成失败: ${error.message}`);
  } finally {
    const index = loadingCategories.value.indexOf(category);
    if (index !== -1) {
      loadingCategories.value.splice(index, 1);
    }
  }
}
// 处理 AI 优化内容请求
async function handleAiOptimizeContent({ category, index, attribute }) {
  console.log('CharacterGenerator: 处理优化内容请求', { category, index, attribute });
  
  // 添加到加载状态
  loadingCategories.value.push(category);
  
  try {
    // 找到对应的类别配置
    const categoryConfig = categoryOptions.find(cat => cat.title === category);
    if (!categoryConfig) {
      throw new Error(`未找到类别: ${category}`);
    }
    
    if (!selectedFile.value) {
      throw new Error('没有选择文件');
    }
    
    const fileContent = await selectedFile.value.text();
    
    const response = await fetchWithTimeout('/api/v1/optimize_content', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        category: categoryConfig.key,
        content: attribute.内容,
        reference: fileContent,
        user_id: 'web'  // 添加 user_id
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `优化内容请求失败: ${response.status} ${response.statusText}`);
    }
    
    const result = await response.json();
    console.log('优化内容结果:', result);
    
    // 更新编辑中的属性
    if (characterData[categoryConfig.key] && characterData[categoryConfig.key][index]) {
      // 只更新内容
      characterData[categoryConfig.key][index].内容 = result.内容;
    }
    
    ElMessage.success('内容优化完成');
  } catch (error) {
    console.error('AI 优化内容失败:', error);
    ElMessage.error(`AI 优化内容失败: ${error.message}`);
  } finally {
    // 移除加载状态
    const index = loadingCategories.value.indexOf(category);
    if (index !== -1) {
      loadingCategories.value.splice(index, 1);
    }
  }
}

// 处理 AI 优化关键词请求
async function handleAiOptimizeKeywords({ category, index, attribute }) {
  console.log('CharacterGenerator: 处理优化关键词请求', { category, index, attribute });
  
  // 添加到加载状态
  loadingCategories.value.push(category);
  
  try {
    // 找到对应的类别配置
    const categoryConfig = categoryOptions.find(cat => cat.title === category);
    if (!categoryConfig) {
      throw new Error(`未找到类别: ${category}`);
    }
    
    console.log('发送优化关键词请求:', {
      category: categoryConfig.key,
      content: attribute.内容,
      keywords: attribute.关键词
    });
    
    if (!selectedFile.value) {
      throw new Error('没有选择文件');
    }
    
    const fileContent = await selectedFile.value.text();
    console.log('文件内容长度:', fileContent.length);
    
    const response = await fetchWithTimeout('/api/v1/optimize_keywords', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        category: categoryConfig.key,
        content: attribute.内容,
        keywords: attribute.关键词,
        reference: fileContent
      })
    });
    
    console.log('优化关键词响应状态:', response.status);
    
    if (!response.ok) {
      throw new Error(`优化关键词请求失败: ${response.status} ${response.statusText}`);
    }
    
    const result = await response.json();
    console.log('优化关键词结果:', result);
    
    // 更新编辑中的属性
    if (characterData[categoryConfig.key] && characterData[categoryConfig.key][index]) {
      // 只更新关键词
      console.log('更新关键词:', result.关键词);
      characterData[categoryConfig.key][index].关键词 = result.关键词;
    }
    
    ElMessage.success('关键词优化完成');
  } catch (error) {
    console.error('AI 优化关键词失败:', error);
    ElMessage.error(`AI 优化关键词失败: ${error.message}`);
  } finally {
    // 移除加载状态
    const index = loadingCategories.value.indexOf(category);
    if (index !== -1) {
      loadingCategories.value.splice(index, 1);
    }
  }
}
</script>


<style lang="scss" scoped>
.character-generator {
  height: 100%;
  
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: var(--spacing-xl);
  }
  
  .channel-info {
    font-family: var(--font-family-secondary);
    font-size: var(--font-size-base);
    color: var(--color-text);
    margin-bottom: var(--spacing-xl);
    opacity: 0.8;
    text-shadow: 0 0 5px var(--shadow-secondary);
  }
  
  .instruction-text {
    font-size: var(--font-size-lg);
    color: var(--color-text-light);
    margin-bottom: var(--spacing-xxl);
    text-align: center;
    text-shadow: 0 0 8px var(--shadow-primary);
  }
  
  .file-control {
    width: 100%;
    margin-bottom: var(--spacing-lg);
    
    .file-input {
      display: none;
    }
    
    .file-name {
      margin-top: var(--spacing-sm);
      font-size: var(--font-size-sm);
      color: var(--color-text);
      opacity: 0.8;
    }
  }
  
  .category-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: var(--spacing-md);
    margin-top: var(--spacing-sm);
  }
  
  .action-buttons {
    margin-top: var(--spacing-xl);
    display: flex;
    justify-content: center;
  }
  
  .processing-state {
    padding: var(--spacing-xl);
  }
  
  .result-container {
    height: 100%;
  }
}
</style>